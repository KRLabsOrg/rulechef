#!/usr/bin/env python3
"""HITL demo: plain-English feedback on learned rules -> measurable repair.

The qualitative audit of the TAB ruleset found three readable defects
(possible to find precisely BECAUSE rules are readable):
  - QUANTITY: fraction pattern matches case numbers like "1432/03" (F1 0.06)
  - CODE: misses bare application numbers ("no. 36244/06") -> low recall
  - PERSON initials rule fires on abbreviations -> low precision (0.35)

This demo plays the human: it attaches rule-level feedback in plain English
via chef.add_feedback(level="rule"), runs ONE incremental learning round,
and reports per-type test F1 before vs after. This exercises the HITL
functionality end-to-end: inspect -> critique in English -> repair.

Usage:
    OPENAI_API_KEY=... python benchmarks/demo_hitl_feedback.py \
        --base-url https://inference.baseten.co/v1 --model moonshotai/Kimi-K2.6
"""

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path

from benchmark_extract import TAB_FORMAT, TAB_SEMANTIC, group_f1, load_tab_ds, prf, score

# (rule name -> plain-English feedback a domain reviewer would write)
FEEDBACK = {
    "physical_measurements": (
        "This QUANTITY rule misses most quantities in the corpus. Cover amounts of "
        "money with currency (e.g. '10,000 euros', 'EUR 4,000', '300,000 Turkish "
        "liras'), areas ('1,000 square metres'), and distances. Never match "
        "number/number patterns like '1432/03' — those are case numbers, not quantities."
    ),
    "case_and_echr_numbers": (
        "This CODE rule misses bare application numbers. It must match patterns like "
        "'no. 36244/06', 'nos. 6210/73 and 6877/75', and application numbers in "
        "parentheses '(no. 36244/06)'. The slash format digits/2-digit-year is the "
        "strongest signal. Do not match dates or monetary amounts."
    ),
    "initials_with_period": (
        "This PERSON rule has too many false positives: it fires on abbreviations "
        "that are not people. Only match initials when adjacent to a name or title "
        "context (e.g. 'Mr A.B.', 'A.B. lodged the application'), and never match "
        "single common abbreviations like 'p.', 'No.', 'art.', 'cf.'."
    ),
}
TASK_FEEDBACK = (
    "Focus repairs ONLY on the rules with feedback (QUANTITY, CODE, PERSON initials). "
    "Do not modify the well-performing DATETIME, court, and country rules."
)


def rebuild_rules(rule_dicts):
    from rulechef.core import Rule, RuleFormat

    rules = []
    for d in rule_dicts:
        rules.append(
            Rule(
                id=str(uuid.uuid4())[:8],
                name=d["name"],
                description=d.get("description", ""),
                format=RuleFormat(d["format"]),
                content=d["content"],
                priority=d.get("priority", 5),
                output_template=d.get("output_template"),
                output_key=d.get("output_key"),
                validated_precision=d.get("validated_precision"),
                validated_support=d.get("validated_support", 0),
            )
        )
    return rules


def eval_rules(chef, rules, test):
    from rulechef.core import TaskType

    preds = [
        chef.learner._apply_rules(rules, {"text": r["text"]}, TaskType.NER, "text").get(
            "entities", []
        )
        or []
        for r in test
    ]
    return score(preds, [r["entities"] for r in test])


def main():
    from openai import OpenAI

    from rulechef import RuleChef
    from rulechef.core import Task, TaskType

    p = argparse.ArgumentParser(description="HITL rule-feedback demo on TAB")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None)
    p.add_argument("--ckpt", default="benchmarks/results/results_extract_tab.ckpt_rulechef.json")
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--max-iterations", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="benchmarks/results/results_hitl_feedback.json")
    args = p.parse_args()

    types = TAB_FORMAT + TAB_SEMANTIC
    train, test = load_tab_ds(args.train, args.test, args.seed, types)
    ckpt = json.loads(Path(args.ckpt).read_text())
    rules = rebuild_rules(ckpt["result"]["rules"])
    print(f"Loaded {len(rules)} learned rules from checkpoint; test={len(test)}")

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url,
        timeout=240.0,
        max_retries=1,
    )
    task = Task(
        name="tab extraction",
        description=(
            "Extract entity spans. Entity types: " + ", ".join(types) + ". "
            "Use precise patterns; only match unambiguous spans."
        ),
        input_schema={"text": "str"},
        output_schema={"entities": "List[{text,start,end,type}]"},
        type=TaskType.NER,
        text_field="text",
    )
    storage = tempfile.mkdtemp(prefix="rulechef_hitl_")
    chef = RuleChef(
        task=task,
        client=client,
        dataset_name="hitl",
        storage_path=storage,
        allowed_formats=["regex", "code"],
        model=args.model,
        max_rules=30,
        max_samples=60,
        synthesis_output_tokens=6000,
        patch_output_tokens=4096,
    )
    for r in train:
        chef.add_example({"text": r["text"]}, {"entities": r["entities"]})
    # BEFORE
    before = eval_rules(chef, rules, test)
    print("\nBEFORE feedback:")
    for t in ("QUANTITY", "CODE", "PERSON"):
        pr, rc, f1 = prf(before[t])
        print(f"  {t:<10} P={pr:.2f} R={rc:.2f} F1={f1:.2f}")

    # Install rules into the dataset, then attach the human feedback
    chef.dataset.rules = rules
    by_name = {r.name: r for r in rules}
    attached = 0
    for name, text in FEEDBACK.items():
        rule = by_name.get(name)
        if rule is None:
            print(f"  (rule '{name}' not in set — skipping its feedback)")
            continue
        chef.add_feedback(text, level="rule", target_id=rule.id)
        attached += 1
    chef.add_feedback(TASK_FEEDBACK, level="task")
    print(f"\nAttached {attached} rule-level + 1 task-level feedback items. Re-learning...")

    result = chef.learn_rules(
        run_evaluation=True,
        max_refinement_iterations=args.max_iterations,
        incremental_only=True,
        holdout_fraction=0.2,
        split_seed=args.seed,
    )
    if result is None:
        print("incremental learning failed")
        return
    new_rules, _ = result

    after = eval_rules(chef, new_rules, test)
    print(f"\nAFTER feedback ({len(new_rules)} rules):")
    print(f"{'type':<10}{'before':>16}{'after':>16}{'ΔF1':>8}")
    rows = {}
    for t in types:
        pb, rb, fb = prf(before[t])
        pa, ra, fa = prf(after[t])
        rows[t] = {"before": fb, "after": fa}
        flag = "  <- feedback" if t in ("QUANTITY", "CODE", "PERSON") else ""
        print(
            f"{t:<10}  P{pb:.2f}/R{rb:.2f}/F{fb:.2f}  P{pa:.2f}/R{ra:.2f}/F{fa:.2f} {fa - fb:>+7.2f}{flag}"
        )
    bf = group_f1(before, TAB_FORMAT)[2]
    af = group_f1(after, TAB_FORMAT)[2]
    bs = group_f1(before, TAB_SEMANTIC)[2]
    a_s = group_f1(after, TAB_SEMANTIC)[2]
    print(f"\nFORMAT micro-F1   {bf:.3f} -> {af:.3f} ({af - bf:+.3f})")
    print(f"SEMANTIC micro-F1 {bs:.3f} -> {a_s:.3f} ({a_s - bs:+.3f})")

    Path(args.output).write_text(
        json.dumps(
            {
                "feedback": FEEDBACK,
                "per_type": rows,
                "format_micro_f1": {"before": bf, "after": af},
                "semantic_micro_f1": {"before": bs, "after": a_s},
                "num_rules": {"before": len(rules), "after": len(new_rules)},
            },
            indent=2,
        )
    )
    print(f"\nSaved to {args.output}")

    import shutil

    shutil.rmtree(storage, ignore_errors=True)


if __name__ == "__main__":
    main()
