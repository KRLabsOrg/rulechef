#!/usr/bin/env python3
"""Ablation matrix for RuleChef on extraction (TAB by default).

Isolates the framework's two core design choices, holding data and seed fixed:
  - coordinator: AgenticCoordinator (critic + audit + guided refinement)
    vs SimpleCoordinator (heuristics only)
  - dev-holdout: patch acceptance on a held-out dev split vs on train itself

Each cell learns rules and evaluates on the same held-out test split,
reporting FORMAT/SEMANTIC micro-F1, rule count, and heavy-LLM-call count
(synthesis + patch + critic + audit), so gains are weighed against cost.
Cells are checkpointed: a finished cell is never recomputed.

Usage:
    OPENAI_API_KEY=... python benchmarks/ablation_extract.py \
        --base-url https://inference.baseten.co/v1 --model moonshotai/Kimi-K2.6
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from benchmark_extract import (
    DATASETS,
    group_f1,
    score,
)


def run_cell(
    train, test, types, fmt_types, sem_types, client, args, *, agentic, holdout, iterations=None
):
    from rulechef import RuleChef
    from rulechef.core import Task, TaskType
    from rulechef.training_logger import TrainingDataLogger

    task = Task(
        name=f"{args.dataset} extraction",
        description=(
            "Extract entity spans. Entity types: " + ", ".join(types) + ". "
            "Use precise patterns; only match unambiguous spans."
        ),
        input_schema={"text": "str"},
        output_schema={"entities": "List[{text,start,end,type}]"},
        type=TaskType.NER,
        text_field="text",
    )
    storage = tempfile.mkdtemp(prefix="rulechef_abl_")
    log_path = Path(tempfile.mkdtemp()) / "abl.jsonl"
    logger = TrainingDataLogger(str(log_path))

    coordinator = None
    if agentic:
        from rulechef.coordinator import AgenticCoordinator

        coordinator = AgenticCoordinator(
            client,
            model=args.model,
            prune_after_learn=True,
            enable_critic=True,
            critic_interval=2,
            audit_interval=3,
        )

    chef = RuleChef(
        task=task,
        client=client,
        dataset_name="abl",
        storage_path=storage,
        allowed_formats=["regex", "code"],
        model=args.model,
        max_rules=args.max_rules,
        max_samples=args.max_samples,
        synthesis_strategy="per_class",
        coordinator=coordinator,
        training_logger=logger,
        synthesis_output_tokens=6000,
        patch_output_tokens=4096,
    )
    for r in train:
        chef.add_example({"text": r["text"]}, {"entities": r["entities"]})

    iters = args.max_iterations if iterations is None else iterations
    t0 = time.time()
    result = chef.learn_rules(
        run_evaluation=iters > 0,
        max_refinement_iterations=max(iters, 1),
        holdout_fraction=(0.2 if holdout else 0.0),
        split_seed=args.seed,
    )
    learn_s = time.time() - t0
    if result is None:
        return None
    rules, _ = result

    preds = [
        chef.learner._apply_rules(rules, {"text": r["text"]}, TaskType.NER, "text").get(
            "entities", []
        )
        or []
        for r in test
    ]
    per = score(preds, [r["entities"] for r in test])

    import shutil

    shutil.rmtree(storage, ignore_errors=True)
    return {
        "format_f1": round(group_f1(per, fmt_types)[2], 4),
        "semantic_f1": round(group_f1(per, sem_types)[2], 4),
        "per_type_f1": {
            t: round(2 * c["tp"] / max(1, 2 * c["tp"] + c["fp"] + c["fn"]), 4)
            for t, c in ((t, per[t]) for t in types)
        },
        "num_rules": len(rules),
        "learn_s": round(learn_s, 1),
        "heavy_llm_calls": logger.count,
        "rules": [r.to_dict() for r in rules],
    }


def main():
    from openai import OpenAI

    p = argparse.ArgumentParser(description="RuleChef extraction ablation (coordinator x holdout)")
    p.add_argument("--dataset", choices=list(DATASETS), default="tab")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None)
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--max-rules", type=int, default=60)
    p.add_argument("--max-samples", type=int, default=80)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    loader, fmt_types, sem_types, _nl, _syn = DATASETS[args.dataset]
    types = fmt_types + sem_types
    train, test = loader(args.train, args.test, args.seed, types)
    print(f"{args.dataset}: train {len(train)} test {len(test)}")

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url,
        timeout=240.0,
        max_retries=1,
    )
    out_path = Path(args.output or f"benchmarks/results/results_ablation_extract_{args.dataset}.json")

    cells = [
        ("oneshot_prompting", False, False, 0),  # single synthesis call, no refinement
        ("simple", False, False, None),
        ("simple+holdout", False, True, None),
        ("agentic", True, False, None),
        ("agentic+holdout", True, True, None),
        # horizon test: does agentic coordination catch up given more iterations?
        # (critic fires every 2, audit every 3 — at 3 iters the machinery barely runs)
        ("agentic+holdout_8iter", True, True, 8),
    ]
    results = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text()).get("cells", {})
        except json.JSONDecodeError:
            results = {}

    for name, agentic, holdout, iters in cells:
        if name in results:
            print(f"\n=== cell {name}: cached, skipping ===")
            continue
        print(f"\n=== cell {name} (agentic={agentic}, holdout={holdout}, iters={iters}) ===")
        m = run_cell(
            train,
            test,
            types,
            fmt_types,
            sem_types,
            client,
            args,
            agentic=agentic,
            holdout=holdout,
            iterations=iters,
        )
        if m is None:
            print("  learning failed")
            continue
        results[name] = m
        out_path.write_text(json.dumps({"config": vars(args), "cells": results}, indent=2))
        print(
            f"  FORMAT={m['format_f1']:.3f} SEMANTIC={m['semantic_f1']:.3f} "
            f"rules={m['num_rules']} heavy_calls={m['heavy_llm_calls']} ({m['learn_s']:.0f}s)"
        )

    print(f"\n{'=' * 76}\nABLATION — {args.dataset}\n{'=' * 76}")
    print(f"{'cell':<18}{'FORMAT F1':>10}{'SEMANTIC F1':>13}{'#rules':>8}{'heavy calls':>12}")
    print("-" * 76)
    for name, _, _, _ in cells:
        m = results.get(name)
        if m:
            print(
                f"{name:<18}{m['format_f1']:>10.3f}{m['semantic_f1']:>13.3f}"
                f"{m['num_rules']:>8}{m['heavy_llm_calls']:>12}"
            )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
