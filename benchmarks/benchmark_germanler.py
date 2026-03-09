import argparse
import json
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

from ner_datasets import get_dataset_class_definitions
from ner_datasets.ner_dataset import NERData, NERDataset, NERSample
from ner_datasets.preprocessing_ler import create_sample, download_dataset

# ── Dataset loading ─────────────────────────────────────────


def load_germanler():
    """Load GERMANLER from HuggingFace and return (train, test, label_names)."""

    ds = download_dataset("elenanereiss/german-ler")

    ner_data = NERData(samples=[])

    for d in ds:
        sample = create_sample(d)
        ner_data.samples.append(sample)

    train = [sample for sample in ner_data.samples if sample.split == "train"]
    val = [sample for sample in ner_data.samples if sample.split == "test"]

    entity_definitions = get_dataset_class_definitions("ler")
    entity_names = set(entity_definitions.keys())

    return train, val, entity_names


def select_windows(text, entities, window_size=200):
    merged_windows = []
    for ent in entities:
        start = max(0, ent["start"] - window_size)
        end = min(len(text), ent["end"] + window_size)
        if merged_windows and start <= merged_windows[-1][1]:
            merged_windows[-1][1] = max(end, merged_windows[-1][1])
            merged_windows[-1][2].append(ent)
        else:
            merged_windows.append([start, end, [ent]])
    return merged_windows


def build_examples(text, merged_windows):
    examples = []
    for start, end, window_entities in merged_windows:
        snippet = text[start:end]
        adjusted_entities = [
            {
                "text": e["text"],
                "start": e["start"] - start,
                "end": e["end"] - start,
                "type": e["type"],
            }
            for e in window_entities
        ]
        examples.append({"text": snippet, "entities": adjusted_entities})
    return examples


def sample_few_shot(
    train_data, shots_per_class, seed=42, num_classes=None, classes=None, window_size=200
):
    rng = random.Random(seed)
    if not classes:
        all_labels = sorted({l["type"] for sample in train_data for l in sample.labels})
        if num_classes and num_classes < len(all_labels):
            rng.shuffle(all_labels)
            classes = set(sorted(all_labels[:num_classes]))
        else:
            classes = set(all_labels)

    classes = set(classes)

    all_examples = []
    for sample in train_data:
        text = sample.text
        entities = sorted(
            [l for l in sample.labels if l["type"] in classes],
            key=lambda x: x["start"],
        )
        if not entities:
            continue
        all_examples.extend(build_examples(text, select_windows(text, entities, window_size)))

    by_label = defaultdict(list)
    for ex in all_examples:
        for ent in ex["entities"]:
            by_label[ent["type"]].append(ex)

    labels = sorted(classes & set(by_label.keys()))
    missing = classes - set(by_label.keys())
    if missing:
        print(f"WARNING: classes not found in dataset: {missing}")

    sampled = []
    remaining = []
    for label in labels:
        examples = by_label[label]
        seen = set()
        unique = []
        for ex in examples:
            if ex["text"] not in seen:
                seen.add(ex["text"])
                unique.append(ex)
        rng.shuffle(unique)
        sampled.extend(unique[:shots_per_class])
        remaining.extend(unique[shots_per_class:])

    return sampled, remaining, classes


# ──Entity Output Schema ────────────────────────────────────────


class Entity(BaseModel):
    text: str = Field(description="The matched text span")
    start: int = Field(description="Start character offset")
    end: int = Field(description="End character offset")
    type: str = Field(description="Entity label")


class NEROutput(BaseModel):
    entities: List[Entity]


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
    print("Loading GERMANLER dataset...")
    train_all, test_all, entity_names = load_germanler()
    print(f"  Train: {len(train_all)}, Test: {len(test_all)}, Classes: {len(entity_names)}")

    # 2. Few-shot sample (optionally limited to N classes)

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    train_sample, train_remaining, selected_classes = sample_few_shot(
        train_all,
        shots_per_class=args.shots,
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
    filtered = []
    for ex in test_all:
        kept_labels = [label for label in ex.labels if label["type"] in selected_classes]
        if kept_labels:
            filtered.append(NERSample(text=ex.text, split=ex.split, labels=kept_labels))
    test_data = filtered
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
        name="German LER Named Entity Recognition",
        description=f"Recognize named entities in German legal text. Entities to look for: {', '.join(active_labels)}.",
        input_schema={"text": "str"},
        output_schema=NEROutput,
        type=TaskType.NER,
        text_field="text",
    )

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
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
            "benchmark": "germanler",
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
        dataset_name="german_ler",
        storage_path=storage_dir,
        allowed_formats=[RuleFormat.REGEX],
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
            {"label": ex["entities"]},
        )
    t_add = time.time() - t0
    print(f"  Done ({t_add:.1f}s)")

    # 5. Build eval Dataset from unused training data (for refinement)
    eval_dataset = Dataset(name="germanler_eval", task=task)
    for ex in train_remaining:
        eval_dataset.examples.append(
            Example(
                id=str(uuid.uuid4())[:8],
                input={"text": ex["text"]},
                expected_output={"entities": ex["entities"]},
                source="benchmark",
            )
        )

    # 6. Learn rules (synthesis only, no refinement yet)
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

    # 7. Refine against eval set (unused training data), test set stays held out
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

    # 8. Print rule summary
    print(f"\n{'─' * 70}")
    print("RULES LEARNED:")
    print(f"{'─' * 70}")
    for r in sorted(rules, key=lambda r: -r.priority):
        content_preview = r.content.replace("\n", " ")[:100]
        print(f"  [p={r.priority}] {r.name}: {content_preview}")
    print(f"{'─' * 70}")

    # Build held-out test Dataset for final evaluation
    test_dataset = Dataset(name="germanler_test", task=task)
    for ex in test_data:
        test_dataset.examples.append(
            Example(
                id=str(uuid.uuid4())[:8],
                input={"text": ex.text},
                expected_output={"entities": ex.labels},
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
    run_benchmark(None)


def main():
    parser = argparse.ArgumentParser(
        description="GermanLER Named Entity Recognition Benchmark for RuleChef"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=50,
        help="Examples per intent class for training (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="LLM model for rule synthesis (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
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
        help="Limit to N random classes (default: all 7)",
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
        default=10,
        help="Max rules to generate per synthesis (default: 100)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
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
        default=100,
        help="Limit test set size for quick runs (default: full 3080)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results_germanler.json",
        help="Save results to JSON file (default: benchmarks/results_germanler.json)",
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
