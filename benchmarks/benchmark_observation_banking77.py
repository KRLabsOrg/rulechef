#!/usr/bin/env python3
"""Observation-Mode Banking77 Benchmark for RuleChef

Simulates a real-world scenario: RuleChef observes an LLM classifying banking
queries, learns rules from the LLM's own predictions (silver labels), and
progressively replaces LLM calls with rules.

The story: "After 200 LLM calls ($0.50), rules handle 50% of future queries
with the same accuracy as the LLM -- cutting your ongoing costs in half."

Flow:
  1. chef.start_observing(client, auto_learn=False) -- wrap the LLM client
  2. Stream queries through the wrapped client -- RuleChef silently captures
  3. At each checkpoint, call chef.learn_rules() -- discovers task + synthesizes
  4. Evaluate: rules vs gold on held-out test, LLM predictions vs gold baseline
  5. chef.stop_observing()

Usage:
    # Quick test (5 classes, small checkpoints)
    python benchmarks/benchmark_observation_banking77.py \\
        --checkpoints 10,25 --num-classes 5 --max-iterations 1 \\
        --base-url https://api.groq.com/openai/v1 \\
        --model moonshotai/kimi-k2-instruct-0905

    # Full run with cache
    python benchmarks/benchmark_observation_banking77.py \\
        --cache benchmarks/cache_banking77.jsonl \\
        --base-url https://api.groq.com/openai/v1 \\
        --model moonshotai/kimi-k2-instruct-0905

Requirements:
    pip install datasets rulechef openai
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# -- Dataset loading ----------------------------------------------------------


def load_banking77():
    """Load BANKING77 from HuggingFace and return (all_examples, label_names)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    ds = load_dataset("legacy-datasets/banking77")
    label_names = ds["train"].features["label"].names

    def to_records(split):
        return [{"text": row["text"], "label": label_names[row["label"]]} for row in split]

    train = to_records(ds["train"])
    test = to_records(ds["test"])
    return train, test, label_names


def split_data(train_data, test_data, label_names, num_classes, test_split, seed):
    """Split data into observation stream, test set, and selected classes.

    Returns (observation_data, test_data, selected_classes, active_labels).
    """
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for ex in train_data:
        by_label[ex["label"]].append(ex)

    labels = sorted(by_label.keys())
    if num_classes and num_classes < len(labels):
        rng.shuffle(labels)
        labels = sorted(labels[:num_classes])

    selected_classes = set(labels)

    # Pool all data for selected classes
    pool = [ex for ex in train_data if ex["label"] in selected_classes]
    test_filtered = [ex for ex in test_data if ex["label"] in selected_classes]

    # Use the HF test set as our held-out test set
    rng.shuffle(test_filtered)

    # The training set becomes our observation stream
    rng.shuffle(pool)

    return pool, test_filtered, selected_classes, sorted(selected_classes)


# -- Cache I/O ---------------------------------------------------------------


def load_cache(cache_path):
    """Load cached LLM predictions from JSONL."""
    entries = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_cache_entry(cache_file, entry):
    """Append a single cache entry to JSONL file."""
    cache_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    cache_file.flush()


# -- LLM classification ------------------------------------------------------


def build_classification_messages(text, active_labels):
    """Build messages for LLM intent classification."""
    system_prompt = (
        "You are a banking customer query classifier. "
        "Classify the customer's query into exactly one of the following intent labels.\n\n"
        f"Intent labels:\n{chr(10).join(active_labels)}\n\n"
        "Respond with ONLY the intent label, nothing else."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def parse_llm_label(response_text, active_labels):
    """Parse LLM response to extract the predicted label."""
    text = response_text.strip().strip('"').strip("'").strip()
    # Exact match
    if text in active_labels:
        return text
    # Case-insensitive match
    lower_map = {lab.lower(): lab for lab in active_labels}
    if text.lower() in lower_map:
        return lower_map[text.lower()]
    # Substring match (LLM might add extra text)
    for label in active_labels:
        if label.lower() in text.lower():
            return label
    return text  # Return as-is, will be wrong


# -- Metrics ------------------------------------------------------------------


def compute_classification_metrics(predictions, gold_labels, active_labels):
    """Compute classification metrics: accuracy, per-class precision/recall/F1."""
    assert len(predictions) == len(gold_labels)
    total = len(predictions)
    correct = sum(1 for p, g in zip(predictions, gold_labels) if p == g)

    per_class = {}
    for label in active_labels:
        tp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, gold_labels) if p != label and g == label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Micro-averaged
    total_tp = sum(c["tp"] for c in per_class.values())
    total_fp = sum(c["fp"] for c in per_class.values())
    total_fn = sum(c["fn"] for c in per_class.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Macro F1
    class_f1s = [c["f1"] for c in per_class.values() if c["tp"] + c["fn"] > 0]
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def evaluate_rules_on_test(chef, test_data, active_labels):
    """Evaluate current rules on held-out test set.

    Returns dict with coverage, precision/recall/F1 vs gold, precision vs LLM, etc.
    """
    predictions = []
    covered_count = 0

    for ex in test_data:
        result = chef.extract({"text": ex["text"]})
        pred_label = result.get("label", "").strip()
        predictions.append(pred_label)
        if pred_label:
            covered_count += 1

    coverage = covered_count / len(test_data) if test_data else 0.0

    # Metrics vs gold (only for covered examples)
    gold_labels = [ex["label"] for ex in test_data]
    covered_preds = []
    covered_golds = []
    for pred, gold in zip(predictions, gold_labels):
        if pred:
            covered_preds.append(pred)
            covered_golds.append(gold)

    if covered_preds:
        gold_metrics = compute_classification_metrics(covered_preds, covered_golds, active_labels)
    else:
        gold_metrics = {
            "accuracy": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
        }

    # Overall recall vs gold = coverage * precision (i.e. correct / total)
    correct_vs_gold = sum(1 for p, g in zip(predictions, gold_labels) if p and p == g)
    overall_recall = correct_vs_gold / len(test_data) if test_data else 0.0
    overall_f1 = (
        2
        * gold_metrics["micro_precision"]
        * overall_recall
        / (gold_metrics["micro_precision"] + overall_recall)
        if (gold_metrics["micro_precision"] + overall_recall) > 0
        else 0.0
    )

    return {
        "coverage": coverage,
        "precision_vs_gold": gold_metrics["micro_precision"],
        "recall_vs_gold": overall_recall,
        "f1_vs_gold": overall_f1,
        "macro_f1_vs_gold": gold_metrics["macro_f1"],
        "per_class": gold_metrics.get("per_class", {}),
        "predictions": predictions,
    }


def compute_llm_agreement(rule_predictions, llm_predictions):
    """Compute agreement rate between rules and LLM (on covered examples)."""
    agree = 0
    covered = 0
    for rule_pred, llm_pred in zip(rule_predictions, llm_predictions):
        if rule_pred:
            covered += 1
            if rule_pred == llm_pred:
                agree += 1
    return agree / covered if covered > 0 else 0.0


# -- Benchmark runner ---------------------------------------------------------


def run_benchmark(args):
    import shutil
    import tempfile

    from openai import OpenAI

    from rulechef import RuleChef
    from rulechef.core import RuleFormat

    # 1. Load data
    print("Loading BANKING77 dataset...")
    train_all, test_all, label_names = load_banking77()
    print(f"  Train: {len(train_all)}, Test: {len(test_all)}, Classes: {len(label_names)}")

    # 2. Split data
    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",")]
    max_observations = max(checkpoints)

    obs_data, test_data, selected_classes, active_labels = split_data(
        train_all, test_all, label_names, args.num_classes, args.test_split, args.seed
    )

    # Limit observation stream to max checkpoint
    if len(obs_data) > max_observations:
        obs_data = obs_data[:max_observations]

    num_classes = len(selected_classes)
    print(f"  Observation stream: {len(obs_data)} examples")
    print(f"  Test set: {len(test_data)} examples (held out)")
    print(f"  Classes: {num_classes}")

    # 3. Setup OpenAI client
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url,
    )

    format_map = {
        "code": [RuleFormat.CODE],
        "regex": [RuleFormat.REGEX],
        "both": [RuleFormat.REGEX, RuleFormat.CODE],
    }
    allowed_formats = format_map.get(args.format, [RuleFormat.REGEX])

    storage_dir = tempfile.mkdtemp(prefix="rulechef_obs_banking77_")

    coordinator = None
    if args.agentic:
        from rulechef.coordinator import AgenticCoordinator

        coordinator = AgenticCoordinator(client, model=args.model, prune_after_learn=args.prune)

    # 4. Load or build LLM prediction cache
    cache_entries = []
    cache_file = None
    use_cache = args.cache and Path(args.cache).exists()

    if use_cache:
        print(f"\nLoading LLM prediction cache from {args.cache}...")
        cache_entries = load_cache(args.cache)
        print(f"  Loaded {len(cache_entries)} cached predictions")
    elif args.cache:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        cache_file = Path(args.cache).open("w")  # noqa: SIM115
        print(f"\nWill save LLM predictions to {args.cache}")

    # 5. Compute LLM baseline on test set (from cache or live)
    print(f"\n{'=' * 60}")
    print("OBSERVATION BENCHMARK: Banking77")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model} | Format: {args.format} | Test: {len(test_data)} examples")

    # -- LLM Baseline --
    # We need LLM predictions for the test set to compute the baseline.
    # For the observation stream, LLM predictions are collected during observation.
    print("\n  Computing LLM baseline on test set...")
    llm_test_predictions = []
    llm_baseline_cache_path = None
    if args.cache:
        llm_baseline_cache_path = Path(args.cache).with_suffix(".baseline.jsonl")

    if llm_baseline_cache_path and llm_baseline_cache_path.exists():
        # Load cached baseline
        with open(llm_baseline_cache_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                llm_test_predictions.append(entry["llm_label"])
        print(f"  Loaded {len(llm_test_predictions)} cached baseline predictions")
    else:
        # Run LLM on test set
        baseline_cache_file = None
        if llm_baseline_cache_path:
            baseline_cache_file = Path(llm_baseline_cache_path).open("w")  # noqa: SIM115

        for i, ex in enumerate(test_data):
            messages = build_classification_messages(ex["text"], active_labels)
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0,
                max_tokens=50,
            )
            llm_response = response.choices[0].message.content.strip()
            llm_label = parse_llm_label(llm_response, active_labels)
            llm_test_predictions.append(llm_label)

            if baseline_cache_file:
                save_cache_entry(
                    baseline_cache_file,
                    {
                        "text": ex["text"],
                        "gold_label": ex["label"],
                        "llm_response": llm_response,
                        "llm_label": llm_label,
                    },
                )

            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(test_data)}...")

        if baseline_cache_file:
            baseline_cache_file.close()

    # Compute baseline metrics
    test_gold = [ex["label"] for ex in test_data]
    llm_baseline = compute_classification_metrics(llm_test_predictions, test_gold, active_labels)

    print("\n  LLM BASELINE (evaluated on gold):")
    print(
        f"    Accuracy: {llm_baseline['accuracy']:.1%} ({llm_baseline['correct']}/{llm_baseline['total']})"
    )
    print(f"    Micro F1: {llm_baseline['micro_f1']:.1%}")
    print(f"    Macro F1: {llm_baseline['macro_f1']:.1%}")

    # 6. Progressive observation + learning
    print(f"\n  Checkpoints: {', '.join(str(c) for c in checkpoints)}")
    print()

    # Table header
    header = f"  {'Obs.':>6}  {'Rules':>5}  {'Cover.':>6}  {'Prec(g)':>7}  {'Prec(l)':>7}  {'F1(g)':>6}  {'Replace':>7}  {'Learn(s)':>8}  {'ms/q':>6}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    checkpoint_metrics = []
    observation_llm_labels = []  # Track LLM predictions for observation data

    # Create fresh RuleChef for observation mode (no task -- auto-discover)
    chef = RuleChef(
        client=client,
        dataset_name="banking77_obs_bench",
        storage_path=storage_dir,
        allowed_formats=allowed_formats,
        model=args.model,
        use_grex=not args.no_grex,
        max_rules=args.max_rules,
        max_samples=args.max_samples,
        coordinator=coordinator,
    )

    if not use_cache:
        # Live mode: start observing the client
        chef.start_observing(client, auto_learn=False)

    prev_checkpoint = 0
    first_learn = True

    for _cp_idx, checkpoint in enumerate(checkpoints):
        # Feed observations from prev_checkpoint to checkpoint
        new_start = prev_checkpoint
        new_end = min(checkpoint, len(obs_data))

        if use_cache:
            # Replay from cache via add_raw_observation
            for i in range(new_start, new_end):
                if i < len(cache_entries):
                    entry = cache_entries[i]
                    chef.add_raw_observation(
                        messages=entry["messages"],
                        response=entry["llm_response"],
                    )
                    observation_llm_labels.append(entry["llm_label"])
        else:
            # Live mode: call LLM through the observed client
            for i in range(new_start, new_end):
                ex = obs_data[i]
                messages = build_classification_messages(ex["text"], active_labels)
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=50,
                )
                llm_response = response.choices[0].message.content.strip()
                llm_label = parse_llm_label(llm_response, active_labels)
                observation_llm_labels.append(llm_label)

                # Save to cache
                if cache_file:
                    save_cache_entry(
                        cache_file,
                        {
                            "text": ex["text"],
                            "gold_label": ex["label"],
                            "messages": messages,
                            "llm_response": llm_response,
                            "llm_label": llm_label,
                        },
                    )

        prev_checkpoint = new_end

        # Learn rules at this checkpoint
        t0 = time.time()
        try:
            if first_learn:
                result = chef.learn_rules(
                    max_refinement_iterations=args.max_iterations,
                )
                first_learn = False
            else:
                result = chef.learn_rules(
                    incremental_only=True,
                    max_refinement_iterations=args.max_iterations,
                )
        except Exception as e:
            print(f"  Learning failed at checkpoint {checkpoint}: {e}")
            continue
        t_learn = time.time() - t0

        if result is None:
            print(
                f"  {checkpoint:>6}  {'N/A':>5}  {'N/A':>6}  {'N/A':>7}  {'N/A':>7}  {'N/A':>6}  {'N/A':>7}  {t_learn:>7.1f}  {'N/A':>6}"
            )
            continue

        rules, _ = result
        num_rules = len(rules)

        # Evaluate rules on test set
        t0_eval = time.time()
        rule_metrics = evaluate_rules_on_test(chef, test_data, active_labels)
        t_eval = time.time() - t0_eval

        per_query_ms = t_eval / len(test_data) * 1000 if test_data else 0

        # Agreement with LLM
        precision_vs_llm = compute_llm_agreement(rule_metrics["predictions"], llm_test_predictions)

        # Replacement rate = coverage * precision vs LLM
        replacement_rate = rule_metrics["coverage"] * precision_vs_llm

        # Print row
        print(
            f"  {checkpoint:>6}  {num_rules:>5}  {rule_metrics['coverage']:>5.1%}  "
            f"{rule_metrics['precision_vs_gold']:>6.1%}  {precision_vs_llm:>6.1%}  "
            f"{rule_metrics['f1_vs_gold']:>5.1%}  {replacement_rate:>6.1%}  "
            f"{t_learn:>7.1f}  {per_query_ms:>5.2f}"
        )

        # Store metrics
        per_class_list = [
            {"label": label, **metrics}
            for label, metrics in rule_metrics.get("per_class", {}).items()
        ]
        checkpoint_metrics.append(
            {
                "observations": checkpoint,
                "rules": num_rules,
                "coverage": round(rule_metrics["coverage"], 4),
                "precision_vs_gold": round(rule_metrics["precision_vs_gold"], 4),
                "recall_vs_gold": round(rule_metrics["recall_vs_gold"], 4),
                "f1_vs_gold": round(rule_metrics["f1_vs_gold"], 4),
                "precision_vs_llm": round(precision_vs_llm, 4),
                "replacement_rate": round(replacement_rate, 4),
                "learning_time_s": round(t_learn, 1),
                "per_query_ms": round(per_query_ms, 2),
                "per_class": per_class_list,
            }
        )

    print()
    print("  (g) = vs gold labels, (l) = vs LLM predictions (agreement)")
    print("  Replace = coverage x precision vs LLM (safely offloadable %)")

    # 7. Stop observing
    if not use_cache:
        chef.stop_observing()

    # Close cache file
    if cache_file:
        cache_file.close()

    # 8. Build summary
    final = checkpoint_metrics[-1] if checkpoint_metrics else {}
    summary = {
        "final_coverage": final.get("coverage", 0),
        "final_precision_vs_gold": final.get("precision_vs_gold", 0),
        "final_precision_vs_llm": final.get("precision_vs_llm", 0),
        "final_replacement_rate": final.get("replacement_rate", 0),
        "llm_accuracy": llm_baseline["accuracy"],
        "total_observations": max(checkpoints),
        "total_learning_time_s": round(sum(m["learning_time_s"] for m in checkpoint_metrics), 1),
    }

    # 9. Save JSON results
    output = {
        "config": {
            "dataset": "banking77",
            "checkpoints": checkpoints,
            "model": args.model,
            "format": args.format,
            "num_classes": num_classes,
            "test_size": len(test_data),
            "max_rules": args.max_rules,
            "max_samples": args.max_samples,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "use_grex": not args.no_grex,
            "agentic": args.agentic,
        },
        "llm_baseline": {
            "accuracy_vs_gold": round(llm_baseline["accuracy"], 4),
            "micro_f1": round(llm_baseline["micro_f1"], 4),
            "macro_f1": round(llm_baseline["macro_f1"], 4),
            "total_predictions": llm_baseline["total"],
            "correct_predictions": llm_baseline["correct"],
        },
        "checkpoint_metrics": checkpoint_metrics,
        "summary": summary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")

    # Cleanup
    shutil.rmtree(storage_dir, ignore_errors=True)


# -- CLI ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Observation-Mode Banking77 Benchmark for RuleChef"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="10,25,50,100,200,500",
        help="Comma-separated observation counts (default: 10,25,50,100,200,500)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moonshotai/kimi-k2-instruct-0905",
        help="LLM model (default: moonshotai/kimi-k2-instruct-0905)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.groq.com/openai/v1",
        help="OpenAI-compatible base URL (default: https://api.groq.com/openai/v1)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="regex",
        choices=["regex", "code", "both"],
        help="Rule format (default: regex)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Refinement iterations per checkpoint (default: 3)",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=100,
        help="Max rules per synthesis (default: 100)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max examples in LLM prompt (default: 200)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction held out for test (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/results_observation_banking77.json",
        help="Output JSON path (default: benchmarks/results/results_observation_banking77.json)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to LLM prediction cache JSONL (if exists, skip LLM calls)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Limit to N random classes for faster testing (default: all 77)",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use AgenticCoordinator",
    )
    parser.add_argument(
        "--no-grex",
        action="store_true",
        help="Disable grex hints",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Enable LLM-powered rule pruning/merging after learning (requires --agentic)",
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
