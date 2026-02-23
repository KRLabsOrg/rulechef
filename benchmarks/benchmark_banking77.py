#!/usr/bin/env python3
"""BANKING77 Intent Classification Benchmark for RuleChef

Evaluates rulechef's ability to learn regex classification rules from few examples.
Uses the BANKING77 dataset (77 banking intent classes, 13K examples).

Usage:
    # Groq (fast, free-ish)
    export OPENAI_API_KEY=gsk_...
    python benchmarks/benchmark_banking77.py --shots 5 \
        --base-url https://api.groq.com/openai/v1 \
        --model llama-3.3-70b-versatile

    # OpenAI
    export OPENAI_API_KEY=sk-...
    python benchmarks/benchmark_banking77.py --shots 5 --model gpt-4o-mini

    # Quick test (100 test examples)
    python benchmarks/benchmark_banking77.py --shots 2 --test-limit 100 --max-iterations 1

Requirements:
    pip install datasets rulechef openai
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

# ── Dataset loading ─────────────────────────────────────────


def load_banking77():
    """Load BANKING77 from HuggingFace and return (train, test, label_names)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    ds = load_dataset("legacy-datasets/banking77")
    label_names = ds["train"].features["label"].names

    def to_records(split):
        return [{"text": row["text"], "label": label_names[row["label"]]} for row in split]

    return to_records(ds["train"]), to_records(ds["test"]), label_names


def sample_few_shot(train_data, shots_per_class, seed=42, num_classes=None, classes=None):
    """Sample K examples per intent class, stratified. Optionally limit to N classes.

    Returns (train_sample, remaining, selected_classes).
    remaining = unused training examples from selected classes (for eval set).
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for ex in train_data:
        by_label[ex["label"]].append(ex)

    labels = sorted(by_label.keys())
    if classes:
        missing = set(classes) - set(labels)
        if missing:
            print(f"WARNING: classes not found in dataset: {missing}")
        labels = sorted(c for c in classes if c in by_label)
    elif num_classes and num_classes < len(labels):
        rng.shuffle(labels)
        labels = sorted(labels[:num_classes])

    sampled = []
    remaining = []
    for label in labels:
        examples = by_label[label]
        rng.shuffle(examples)
        sampled.extend(examples[:shots_per_class])
        remaining.extend(examples[shots_per_class:])

    return sampled, remaining, set(labels)


# ── Benchmark runner ────────────────────────────────────────


def run_benchmark(args):
    from openai import OpenAI

    from rulechef import RuleChef
    from rulechef.core import (
        Dataset,
        Example,
        RuleFormat,
        Task,
        TaskType,
    )
    from rulechef.evaluation import evaluate_dataset

    # 1. Load data
    print("Loading BANKING77 dataset...")
    train_all, test_all, label_names = load_banking77()
    print(f"  Train: {len(train_all)}, Test: {len(test_all)}, Classes: {len(label_names)}")

    # 2. Few-shot sample (optionally limited to N classes)
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    train_sample, train_remaining, selected_classes = sample_few_shot(
        train_all,
        args.shots,
        seed=args.seed,
        num_classes=args.num_classes,
        classes=classes,
    )
    num_classes = len(selected_classes)
    print(f"  Few-shot: {args.shots}-shot x {num_classes} classes = {len(train_sample)} examples")
    print(f"  Eval pool (unused train): {len(train_remaining)} examples")
    if args.num_classes:
        print(f"  Selected classes: {', '.join(sorted(selected_classes))}")

    # Filter test set to only selected classes (held out — never seen during learning)
    test_data = [ex for ex in test_all if ex["label"] in selected_classes]
    if args.test_limit:
        rng = random.Random(args.seed)
        test_data = list(test_data)
        rng.shuffle(test_data)
        test_data = test_data[: args.test_limit]
        print(f"  Test subset: {len(test_data)} (limited, held out)")
    else:
        print(f"  Test: {len(test_data)} (filtered to selected classes, held out)")

    # 3. Configure rulechef
    active_labels = sorted(selected_classes)
    task = Task(
        name="Banking Intent Classification",
        description=(
            f"Classify banking customer queries into one of {num_classes} intent categories. "
            f"Return the most specific matching intent label.\n"
            f"Intent labels: {', '.join(active_labels)}"
        ),
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )

    format_map = {
        "code": [RuleFormat.CODE],
        "regex": [RuleFormat.REGEX],
        "both": [RuleFormat.REGEX, RuleFormat.CODE],
    }
    allowed_formats = format_map.get(args.format, [RuleFormat.CODE])

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url,
    )

    import tempfile

    from rulechef.training_logger import TrainingDataLogger

    storage_dir = tempfile.mkdtemp(prefix="rulechef_bench_")

    # Training logger
    log_path = Path(args.output).with_suffix(".training.jsonl")
    logger = TrainingDataLogger(
        str(log_path),
        run_metadata={
            "benchmark": "banking77",
            "model": args.model,
            "format": args.format,
            "num_classes": num_classes,
        },
    )
    print(f"  Training log: {log_path}")

    coordinator = None
    if args.agentic:
        from rulechef.coordinator import AgenticCoordinator

        coordinator = AgenticCoordinator(client, model=args.model)
        print("  Agentic coordinator: enabled")

    chef = RuleChef(
        task=task,
        client=client,
        dataset_name="banking77_bench",
        storage_path=storage_dir,
        allowed_formats=allowed_formats,
        model=args.model,
        use_grex=not args.no_grex,
        max_rules=args.max_rules,
        max_samples=args.max_samples,
        coordinator=coordinator,
        training_logger=logger,
    )

    # 4. Add training examples (suppress per-example prints)
    print(f"\nAdding {len(train_sample)} training examples...")
    t0 = time.time()
    for ex in train_sample:
        chef.add_example(
            {"text": ex["text"]},
            {"label": ex["label"]},
        )
    t_add = time.time() - t0
    print(f"  Done ({t_add:.1f}s)")

    # 6. Build eval Dataset from unused training data (for refinement)
    eval_dataset = Dataset(name="banking77_eval", task=task)
    for ex in train_remaining:
        eval_dataset.examples.append(
            Example(
                id=str(uuid.uuid4())[:8],
                input={"text": ex["text"]},
                expected_output={"label": ex["label"]},
                source="benchmark",
            )
        )

    # 7. Learn rules (synthesis only, no refinement yet)
    print("\nLearning rules...")
    print(f"  model={args.model}")
    print(f"  format={args.format}")
    print(f"  max_rules={args.max_rules}")
    print(f"  max_samples={args.max_samples}")
    print(f"  max_iterations={args.max_iterations}")
    t0 = time.time()
    result = chef.learn_rules(
        run_evaluation=False,
    )
    t_learn = time.time() - t0

    if result is None:
        print("ERROR: Learning failed!")
        return

    rules, _ = result
    print(f"\nSynthesis complete ({t_learn:.1f}s)")
    print(f"  Rules generated: {len(rules)}")

    # 8. Refine against eval set (unused training data), test set stays held out
    iteration_metrics = []

    def on_iteration(iteration, iter_rules, eval_result):
        """Callback to log per-iteration metrics on the dev set."""
        iteration_metrics.append(
            {
                "iteration": iteration,
                "num_rules": len(iter_rules),
                "exact_match": eval_result.exact_match,
                "micro_precision": eval_result.micro_precision,
                "micro_recall": eval_result.micro_recall,
                "micro_f1": eval_result.micro_f1,
                "macro_f1": eval_result.macro_f1,
                "per_class": [
                    {
                        "label": cm.label,
                        "f1": cm.f1,
                        "precision": cm.precision,
                        "recall": cm.recall,
                    }
                    for cm in (eval_result.per_class or [])
                ],
            }
        )

    if args.max_iterations > 0:
        print(
            f"\nRefining against eval set ({len(train_remaining)} examples, max {args.max_iterations} iterations)..."
        )
        t0_refine = time.time()
        rules, refine_eval = chef.learner.evaluate_and_refine(
            rules,
            eval_dataset,
            max_iterations=args.max_iterations,
            coordinator=chef.coordinator,
            iteration_callback=on_iteration,
        )
        t_refine = time.time() - t0_refine
        t_learn += t_refine
        print(f"Refinement complete ({t_refine:.1f}s)")
        if refine_eval:
            print(
                f"  Eval accuracy: {refine_eval.exact_match:.1%}, eval micro F1: {refine_eval.micro_f1:.1%}"
            )

    # 9. Print rule summary
    print(f"\n{'─' * 70}")
    print("RULES LEARNED:")
    print(f"{'─' * 70}")
    for r in sorted(rules, key=lambda r: -r.priority):
        content_preview = r.content.replace("\n", " ")[:100]
        print(f"  [p={r.priority}] {r.name}: {content_preview}")
    print(f"{'─' * 70}")

    # Build held-out test Dataset for final evaluation
    test_dataset = Dataset(name="banking77_test", task=task)
    for ex in test_data:
        test_dataset.examples.append(
            Example(
                id=str(uuid.uuid4())[:8],
                input={"text": ex["text"]},
                expected_output={"label": ex["label"]},
                source="benchmark",
            )
        )

    print(f"\nEVALUATING ON HELD-OUT TEST SET ({len(test_data)} examples)...")
    t0 = time.time()
    test_eval = evaluate_dataset(
        rules,
        test_dataset,
        chef.learner._apply_rules,
    )
    t_eval = time.time() - t0

    # Coverage = what % of test queries got ANY prediction (TP + FP) / total
    coverage = (test_eval.total_tp + test_eval.total_fp) / len(test_data) if test_data else 0

    # 9. Print results
    print(f"\n{'=' * 70}")
    print("BANKING77 BENCHMARK RESULTS")
    print(f"{'=' * 70}")
    print("  Configuration:")
    print(f"    Shots per class:          {args.shots}")
    print(f"    Training examples:        {len(train_sample)}")
    print(f"    Test examples:            {len(test_data)}")
    print(f"    Model:                    {args.model}")
    print(f"    Max rules:                {args.max_rules}")
    print(f"    Max samples in prompt:    {args.max_samples}")
    print(f"    Refinement iterations:    {args.max_iterations}")
    print(f"    Seed:                     {args.seed}")
    print()
    print("  Results:")
    print(f"    Accuracy (exact match):   {test_eval.exact_match:.1%}")
    print(
        f"    Coverage:                 {coverage:.1%} ({test_eval.total_tp + test_eval.total_fp}/{len(test_data)} got a label)"
    )
    print(f"    Micro Precision:          {test_eval.micro_precision:.1%}")
    print(f"    Micro Recall:             {test_eval.micro_recall:.1%}")
    print(f"    Micro F1:                 {test_eval.micro_f1:.1%}")
    print(f"    Macro F1:                 {test_eval.macro_f1:.1%}")
    print()
    print("  Timing:")
    print(f"    Learning:                 {t_learn:.1f}s")
    print(f"    Evaluation:               {t_eval:.1f}s")
    print(f"    Per-query:                {t_eval / len(test_data) * 1000:.2f}ms")
    print()
    print(f"  Rules:                      {len(rules)} total")
    print(f"{'=' * 70}")

    # 10. Per-class breakdown
    if test_eval.per_class:
        sorted_classes = sorted(test_eval.per_class, key=lambda c: c.f1, reverse=True)

        print("\n  Top 10 classes by F1:")
        for cm in sorted_classes[:10]:
            print(
                f"    {cm.label:50s} F1={cm.f1:.0%} P={cm.precision:.0%} R={cm.recall:.0%} "
                f"(TP={cm.tp} FP={cm.fp} FN={cm.fn})"
            )

        print("\n  Bottom 10 classes by F1:")
        for cm in sorted_classes[-10:]:
            print(
                f"    {cm.label:50s} F1={cm.f1:.0%} P={cm.precision:.0%} R={cm.recall:.0%} "
                f"(TP={cm.tp} FP={cm.fp} FN={cm.fn})"
            )

        zero_recall = sum(1 for cm in sorted_classes if cm.recall == 0)
        covered = len(sorted_classes) - zero_recall
        print(
            f"\n  Intent coverage: {covered}/{len(sorted_classes)} intents have at least one correct match"
        )
        print(f"  Uncovered intents: {zero_recall}/{len(sorted_classes)}")

    # 11. Save results
    output_path = Path(args.output)
    results = {
        "config": {
            "shots": args.shots,
            "model": args.model,
            "format": args.format,
            "max_rules": args.max_rules,
            "max_samples": args.max_samples,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "train_size": len(train_sample),
            "test_size": len(test_data),
            "use_grex": not args.no_grex,
            "agentic": args.agentic,
        },
        "results": {
            "accuracy": test_eval.exact_match,
            "coverage": coverage,
            "micro_precision": test_eval.micro_precision,
            "micro_recall": test_eval.micro_recall,
            "micro_f1": test_eval.micro_f1,
            "macro_f1": test_eval.macro_f1,
            "num_rules": len(rules),
            "learning_time_s": round(t_learn, 1),
            "eval_time_s": round(t_eval, 3),
            "per_query_ms": round(t_eval / len(test_data) * 1000, 2),
        },
        "per_class": [
            {
                "label": cm.label,
                "precision": cm.precision,
                "recall": cm.recall,
                "f1": cm.f1,
                "tp": cm.tp,
                "fp": cm.fp,
                "fn": cm.fn,
            }
            for cm in (test_eval.per_class or [])
        ],
        "iteration_metrics": iteration_metrics,
        "rules": [
            {
                "name": r.name,
                "format": r.format.value,
                "content": r.content,
                "priority": r.priority,
                "output_template": r.output_template,
                "output_key": r.output_key,
            }
            for r in rules
        ],
    }
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")

    # Cleanup temp storage
    import shutil

    shutil.rmtree(storage_dir, ignore_errors=True)


# ── CLI ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="BANKING77 Intent Classification Benchmark for RuleChef"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=5,
        help="Examples per intent class for training (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for rule synthesis (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL (e.g. https://api.groq.com/openai/v1)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="regex",
        choices=["regex", "code", "both"],
        help="Rule format (default: regex)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Limit to N random classes (default: all 77)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of specific class names to use (overrides --num-classes)",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=100,
        help="Max rules to generate per synthesis (default: 100)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max training examples in LLM prompt (default: 200)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max refinement iterations (default: 3)",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Limit test set size for quick runs (default: full 3080)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results_banking77.json",
        help="Save results to JSON file (default: benchmarks/results_banking77.json)",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use AgenticCoordinator for LLM-guided refinement",
    )
    parser.add_argument(
        "--no-grex",
        action="store_true",
        help="Disable grex regex pattern suggestions (for ablation)",
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
