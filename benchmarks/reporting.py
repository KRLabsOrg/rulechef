import time

from benchmarks.data import BenchmarkRun
from rulechef.evaluation import evaluate_dataset


def eval_metrics(result) -> dict:
    return {
        "exact_match": result.exact_match,
        "micro_f1": result.micro_f1,
        "micro_precision": result.micro_precision,
        "micro_recall": result.micro_recall,
        "macro_f1": result.macro_f1,
        "per_class": [
            {
                "label": cm.label,
                "f1": cm.f1,
                "precision": cm.precision,
                "recall": cm.recall,
            }
            for cm in (result.per_class or [])
        ],
    }


def make_oniteration_callback(iteration_metrics: list):
    """
    Returns a callback function suitable for RuleChef's iteration_callback.
    Logs per-iteration metrics into the provided iteration_metrics list.
    """

    def on_iteration(iteration, iter_rules, eval_result):
        iteration_metrics.append(
            {
                "iteration": iteration,
                "num_rules": len(iter_rules),
                **eval_metrics(eval_result),
            }
        )

    return on_iteration


def evaluate_test(test_dataset, rules, chef, mode="text", iou_threshold=1):
    t0 = time.time()
    eval_results = evaluate_dataset(
        rules,
        test_dataset,
        chef.learner._apply_rules,
        mode=mode,
        iou_threshold=iou_threshold,
    )
    t_eval = time.time() - t0

    return eval_results, t_eval


def print_results(run: BenchmarkRun):
    args = run.args
    # 9. Print results
    print(f"\n{'=' * 70}")
    print(f"German {args.dataset_name} Results")
    print(f"{'=' * 70}")
    print("Configuration:")
    print(f"Shots per class:          {args.shots}")
    print(f"Training examples:        {len(run.train_data)} ")
    print(f"Test examples:            {len(run.test_data)}")
    print(f"Model:                    {args.model}")
    print(f"Max rules:                {args.max_rules}")
    print(f"Max samples in prompt:    {args.max_samples}")
    print(f"Refinement iterations:    {args.max_iterations}")
    print(f"Seed:                     {args.seed}")
    print()

    print("Results:")
    print(f"Accuracy (exact match):   {run.eval_results.exact_match:.1%}")

    print(f"Micro Precision:          {run.eval_results.micro_precision:.1%}")
    print(f"Micro Recall:             {run.eval_results.micro_recall:.1%}")
    print(f"Micro F1:                 {run.eval_results.micro_f1:.1%}")
    print(f"Macro F1:                 {run.eval_results.macro_f1:.1%}")
    print()
    print("Timing:")
    print(f"Learning:                 {run.t_learn:.1f}s")
    print(f"Evaluation:               {run.t_eval:.1f}s")
    print(f"Per-query:                {run.t_eval / len(run.test_data) * 1000:.2f}ms")
    print()
    print(f"Rules:                      {len(run.rules)} total")
    print(f"{'=' * 70}")


def print_rule_summary(rules):
    print(f"\n{'─' * 70}")
    print("RULES LEARNED:")
    print(f"{'─' * 70}")
    for r in sorted(rules, key=lambda r: -r.priority):
        content_preview = r.content.replace("\n", " ")[:100]
        print(f"  [p={r.priority}] {r.name}: {content_preview}")
    print(f"{'─' * 70}")


def print_per_class_breakdown(eval_results):
    if not eval_results.per_class:
        return
    sorted_classes = sorted(eval_results.per_class, key=lambda c: c.f1, reverse=True)
    print("\n  Top 10 classes by F1:")
    for cm in sorted_classes[:10]:
        print(
            f"    {cm.label:50s} F1={cm.f1:.0%} P={cm.precision:.0%} R={cm.recall:.0%} "
            f"(TP={cm.tp} FP={cm.fp})"
        )
    print("\n  Bottom 10 classes by F1:")
    for cm in sorted_classes[-10:]:
        print(
            f"    {cm.label:50s} F1={cm.f1:.0%} P={cm.precision:.0%} R={cm.recall:.0%} "
            f"(TP={cm.tp} FP={cm.fp})"
        )
