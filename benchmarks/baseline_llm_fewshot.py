#!/usr/bin/env python3
"""LLM Few-Shot Baseline for Banking77

Classifies Banking77 test examples via direct in-context learning (no rule synthesis).
Uses the same data splits, classes, and test set as benchmark_banking77.py for
direct comparison.

Usage:
    # OpenAI
    export OPENAI_API_KEY=sk-...
    python benchmarks/baseline_llm_fewshot.py --model gpt-4o --shots 5

    # Groq
    python benchmarks/baseline_llm_fewshot.py \
        --model llama-3.3-70b-versatile \
        --base-url https://api.groq.com/openai/v1

Requirements:
    pip install datasets openai
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from benchmark_banking77 import load_banking77, sample_few_shot


def build_fewshot_prompt(train_examples, classes):
    """Build a system + few-shot prompt for intent classification."""
    class_list = ", ".join(sorted(classes))
    system = (
        f"You are an intent classifier for banking customer queries. "
        f"Classify each query into exactly one of these intents: {class_list}\n"
        f"Respond with ONLY the intent label, nothing else."
    )

    fewshot_lines = []
    for ex in train_examples:
        fewshot_lines.append(f'Query: "{ex["text"]}"\nIntent: {ex["label"]}')

    fewshot = "\n\n".join(fewshot_lines)
    return system, fewshot


def classify_batch(client, model, system, fewshot, test_examples, classes):
    """Classify test examples one at a time via LLM."""
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, ex in enumerate(test_examples):
        user_msg = f'{fewshot}\n\nQuery: "{ex["text"]}"\nIntent:'

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=50,
                temperature=0,
            )
            prediction = response.choices[0].message.content.strip()
            latency = time.time() - t0

            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens
        except Exception as e:
            print(f"  ERROR on example {i}: {e}")
            prediction = ""
            latency = time.time() - t0

        # Normalize: strip quotes, lowercase match
        prediction_clean = prediction.strip().strip("'\"").strip()
        # Try exact match first, then case-insensitive
        matched = None
        for c in classes:
            if prediction_clean == c:
                matched = c
                break
        if not matched:
            for c in classes:
                if prediction_clean.lower() == c.lower():
                    matched = c
                    break

        results.append(
            {
                "text": ex["text"],
                "expected": ex["label"],
                "predicted": matched or prediction_clean,
                "raw_response": prediction,
                "latency_ms": latency * 1000,
                "matched": matched is not None,
            }
        )

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(test_examples)}] classified...")

    return results, total_prompt_tokens, total_completion_tokens


def compute_metrics(results, classes):
    """Compute accuracy, per-class P/R/F1, micro/macro F1."""
    tp_per_class = defaultdict(int)
    fp_per_class = defaultdict(int)
    fn_per_class = defaultdict(int)

    correct = 0
    for r in results:
        expected = r["expected"]
        predicted = r["predicted"]
        if predicted == expected:
            correct += 1
            tp_per_class[expected] += 1
        else:
            fn_per_class[expected] += 1
            if predicted in classes:
                fp_per_class[predicted] += 1

    accuracy = correct / len(results) if results else 0

    # Micro
    total_tp = sum(tp_per_class.values())
    total_fp = sum(fp_per_class.values())
    total_fn = sum(fn_per_class.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    )

    # Per-class
    per_class = []
    f1_sum = 0
    for c in sorted(classes):
        tp = tp_per_class[c]
        fp = fp_per_class[c]
        fn = fn_per_class[c]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class.append(
            {
                "label": c,
                "precision": p,
                "recall": r,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
        f1_sum += f1

    macro_f1 = f1_sum / len(classes) if classes else 0

    return {
        "accuracy": accuracy,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Few-Shot Baseline for Banking77")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="benchmarks/results_llm_fewshot.json"
    )
    args = parser.parse_args()

    from openai import OpenAI

    # Load data (same as rule benchmark)
    print("Loading BANKING77 dataset...")
    train_all, test_all, label_names = load_banking77()

    classes_list = (
        [c.strip() for c in args.classes.split(",")] if args.classes else None
    )
    train_sample, _, selected_classes = sample_few_shot(
        train_all,
        args.shots,
        seed=args.seed,
        num_classes=args.num_classes,
        classes=classes_list,
    )

    test_data = [ex for ex in test_all if ex["label"] in selected_classes]
    if args.test_limit:
        import random

        rng = random.Random(args.seed)
        test_data = list(test_data)
        rng.shuffle(test_data)
        test_data = test_data[: args.test_limit]

    print(
        f"  {args.shots}-shot x {len(selected_classes)} classes = {len(train_sample)} examples"
    )
    print(f"  Test: {len(test_data)} examples")
    print(f"  Model: {args.model}")

    # Build prompt
    system, fewshot = build_fewshot_prompt(train_sample, selected_classes)

    # Classify
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=args.base_url)
    print(f"\nClassifying {len(test_data)} test examples...")
    t0 = time.time()
    results, prompt_tokens, completion_tokens = classify_batch(
        client,
        args.model,
        system,
        fewshot,
        test_data,
        selected_classes,
    )
    total_time = time.time() - t0

    # Metrics
    metrics = compute_metrics(results, selected_classes)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0

    # Print results
    print(f"\n{'=' * 70}")
    print(f"LLM FEW-SHOT BASELINE RESULTS ({args.model})")
    print(f"{'=' * 70}")
    print(f"  Accuracy:          {metrics['accuracy']:.1%}")
    print(f"  Micro Precision:   {metrics['micro_precision']:.1%}")
    print(f"  Micro Recall:      {metrics['micro_recall']:.1%}")
    print(f"  Micro F1:          {metrics['micro_f1']:.1%}")
    print(f"  Macro F1:          {metrics['macro_f1']:.1%}")
    print(f"  Avg latency:       {avg_latency:.0f}ms")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"  Prompt tokens:     {prompt_tokens:,}")
    print(f"  Completion tokens: {completion_tokens:,}")
    print()

    for cm in metrics["per_class"]:
        print(
            f"  {cm['label']:40s} F1={cm['f1']:.0%} P={cm['precision']:.0%} R={cm['recall']:.0%}"
        )
    print(f"{'=' * 70}")

    # Save
    output = {
        "config": {
            "shots": args.shots,
            "model": args.model,
            "seed": args.seed,
            "test_size": len(test_data),
            "method": "llm_fewshot",
        },
        "results": {
            "accuracy": metrics["accuracy"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "micro_f1": metrics["micro_f1"],
            "macro_f1": metrics["macro_f1"],
            "avg_latency_ms": round(avg_latency, 1),
            "total_time_s": round(total_time, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "per_class": metrics["per_class"],
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
