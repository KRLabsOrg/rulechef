#!/usr/bin/env python3
"""LLM-free search synthesis on TAB format types (rulechef.search).

Same data, split and scorer as ``benchmark_extract.py --dataset tab``:
train = the first 1,000 chunks, test = the next 600 (seed 42); the search
learns on RuleChef's own ``split_dataset(holdout 0.2, seed 42)`` train part
and configs are compared on its dev part. The unlabelled pool is a seeded
4,000-chunk sample of the chunks after test, text only. No model and no API
key; test is scored only with ``--test``.

Usage:
    python benchmarks/benchmark_search.py --min-support 3
    # hybrid: an existing (e.g. LLM-written) ruleset joins the candidate pool
    python benchmarks/benchmark_search.py --seed-rules \
        benchmarks/results/results_extract_tab.ckpt_rulechef.json
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_extract import TAB_FORMAT, TAB_NL, TAB_SEMANTIC, group_f1, load_tab_ds, score

from rulechef.core import Dataset, Example, Rule, Task, TaskType
from rulechef.executor import RuleExecutor
from rulechef.search import SearchConfig, SearchSynthesizer
from rulechef.splitting import split_dataset


def to_dataset(records, name):
    task = Task(
        name=name,
        description="TAB format-type entities",
        input_schema={"text": "str"},
        output_schema={"entities": "List[Entity]"},
        type=TaskType.NER,
        text_field="text",
    )
    ds = Dataset(name=name, task=task)
    for i, r in enumerate(records):
        ds.examples.append(
            Example(
                id=str(i),
                input={"text": r["text"]},
                expected_output={"entities": r["entities"]},
                source="benchmark",
            )
        )
    return ds


def predict(rules, examples):
    ex = RuleExecutor()
    return [
        ex.apply_rules(rules, e.input, TaskType.NER, "text").get("entities", []) for e in examples
    ]


def report(name, rules, examples):
    per = score(predict(rules, examples), [e.expected_output["entities"] for e in examples])
    p, r, f = group_f1(per, TAB_FORMAT)
    by_type = {t: round(group_f1(per, [t])[2], 4) for t in TAB_FORMAT}
    print(f"{name}: format P={p:.4f} R={r:.4f} F1={f:.4f} {by_type}")
    return {"P": p, "R": r, "F1": f, "per_type": by_type}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--seed-rules", default=None, help="ruleset JSON whose rules join the pool")
    ap.add_argument("--test", action="store_true", help="also score the 600 test chunks")
    ap.add_argument("--output", default="benchmarks/results_search_tab.json")
    args = ap.parse_args()

    types = TAB_FORMAT + TAB_SEMANTIC
    train, rest = load_tab_ds(1000, 10**9, 42, types)
    test, pool = rest[:600], rest[600:]
    unlabeled = [r["text"] for r in random.Random(42).sample(pool, 4000)]
    synth_ds, dev_ds = split_dataset(to_dataset(train, "tab_train"), 0.2, seed=42)
    print(f"synth {len(synth_ds.examples)} / dev {len(dev_ds.examples)} / test {len(test)}")

    seeds = None
    if args.seed_rules:
        with open(args.seed_rules) as f:
            d = json.load(f)
        raw = d["rules"] if "rules" in d else d["result"]["rules"]
        seeds = [Rule.from_dict(r) for r in raw if r.get("format", "regex") == "regex"]

    labels = ["DATETIME", "CODE", "QUANTITY"]
    cfg = SearchConfig(labels=labels, min_support=args.min_support)
    t0 = time.time()
    synth = SearchSynthesizer(cfg)
    rules = synth.synthesize(synth_ds, unlabeled, seeds)
    secs = time.time() - t0
    print(f"{len(rules)} rules in {secs:.0f}s (CPU, no model)")
    for rule in rules:
        print(f"  {rule.output_template['type']:9s} {rule.content}")

    out = {
        "config": {
            "min_support": args.min_support,
            "seed_rules": args.seed_rules,
            "labels": labels,
        },
        "label_names": {t: TAB_NL[t] for t in labels},
        "synthesis_s": round(secs, 1),
        "report": synth.report,
        "dev": report("dev", rules, dev_ds.examples),
    }
    if args.test:
        out["test"] = report("test", rules, to_dataset(test, "tab_test").examples)
    out["rules"] = [r.to_dict() for r in rules]
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
