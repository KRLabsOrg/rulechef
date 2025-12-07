"""NER span extraction evaluation metrics"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Span:
    text: str
    start: int
    end: int


def span_iou(span1: Dict, span2: Dict) -> float:
    """Calculate Intersection over Union for two spans"""
    s1_start = span1.get("start", 0)
    s1_end = span1.get("end", 0)
    s2_start = span2.get("start", 0)
    s2_end = span2.get("end", 0)

    # Calculate intersection
    inter_start = max(s1_start, s2_start)
    inter_end = min(s1_end, s2_end)
    intersection = max(0, inter_end - inter_start)

    # Calculate union
    union = (s1_end - s1_start) + (s2_end - s2_start) - intersection

    if union == 0:
        return 0.0

    return intersection / union


def boundary_distance(pred_span: Dict, gold_span: Dict) -> int:
    """Calculate average boundary error distance"""
    start_error = abs(pred_span.get("start", 0) - gold_span.get("start", 0))
    end_error = abs(pred_span.get("end", 0) - gold_span.get("end", 0))
    return (start_error + end_error) // 2


def find_best_match(
    pred_span: Dict, gold_spans: List[Dict], iou_threshold: float = 0.5
) -> Tuple[int, float]:
    """
    Find best matching gold span for a predicted span.
    Returns (index, iou) or (-1, 0.0) if no match above threshold.
    """
    best_idx = -1
    best_iou = 0.0

    for idx, gold_span in enumerate(gold_spans):
        iou = span_iou(pred_span, gold_span)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx

    if best_iou >= iou_threshold:
        return best_idx, best_iou
    return -1, 0.0


def evaluate_spans(
    predictions: List[Dict],
    gold_standard: List[Dict],
    exact_match_only: bool = False,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Evaluate predicted spans against gold standard.

    Args:
        predictions: List of predicted spans [{"text": str, "start": int, "end": int}, ...]
        gold_standard: List of gold spans with same format
        exact_match_only: If True, only exact matches (same text) count
        iou_threshold: IoU threshold for partial match (default 0.5)

    Returns:
        Dictionary with metrics:
        - exact_matches: Count of exact boundary matches
        - partial_matches: Count of IoU-based matches
        - false_positives: Predicted but not in gold
        - false_negatives: In gold but not predicted
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1: 2 * (precision * recall) / (precision + recall)
        - boundary_errors: List of (pred, gold, distance) for error analysis
    """

    if not gold_standard:
        return {
            "exact_matches": 0,
            "partial_matches": 0,
            "false_positives": len(predictions),
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy_exact": 0.0,
            "accuracy_partial": 0.0,
            "boundary_errors": [],
        }

    matched_gold = set()
    exact_matches = 0
    partial_matches = 0
    boundary_errors = []

    for pred in predictions:
        found_exact = False
        found_partial = False

        for gold_idx, gold in enumerate(gold_standard):
            if gold_idx in matched_gold:
                continue

            # Check for exact match (same text and position)
            if (
                pred.get("text") == gold.get("text")
                and pred.get("start") == gold.get("start")
                and pred.get("end") == gold.get("end")
            ):
                exact_matches += 1
                matched_gold.add(gold_idx)
                found_exact = True
                found_partial = True
                break

        # If no exact match, check for partial match
        if not found_exact and not exact_match_only:
            best_idx, iou = find_best_match(pred, gold_standard, iou_threshold)
            if best_idx != -1 and best_idx not in matched_gold:
                partial_matches += 1
                matched_gold.add(best_idx)
                found_partial = True
                distance = boundary_distance(pred, gold_standard[best_idx])
                boundary_errors.append(
                    {
                        "predicted": pred,
                        "gold": gold_standard[best_idx],
                        "distance": distance,
                        "iou": iou,
                    }
                )

    true_positives = exact_matches + partial_matches
    false_positives = len(predictions) - true_positives
    false_negatives = len(gold_standard) - len(matched_gold)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_exact": exact_matches / len(gold_standard) if gold_standard else 0.0,
        "accuracy_partial": true_positives / len(gold_standard)
        if gold_standard
        else 0.0,
        "boundary_errors": boundary_errors,
    }


def print_eval_report(metrics: Dict, dataset_name: str = "Dataset"):
    """Pretty-print evaluation metrics"""
    print(f"\n{'=' * 70}")
    print(f"EVALUATION REPORT: {dataset_name}")
    print(f"{'=' * 70}\n")

    print("SPAN EXTRACTION METRICS")
    print("-" * 70)
    print(f"Exact Match Accuracy:    {metrics['accuracy_exact']:.1%}")
    print(f"Partial Match Accuracy:  {metrics['accuracy_partial']:.1%} (IoU > 0.5)")
    print()

    print("DETAILED METRICS")
    print("-" * 70)
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 Score:  {metrics['f1']:.1%}")
    print()

    print("CONFUSION")
    print("-" * 70)
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print()

    if metrics.get("boundary_errors"):
        print("BOUNDARY ERRORS (Sample)")
        print("-" * 70)
        for error in metrics["boundary_errors"][:5]:
            pred = error["predicted"]
            gold = error["gold"]
            dist = error["distance"]
            print(f"Predicted: '{pred['text']}' [{pred['start']}:{pred['end']}]")
            print(f"Gold:      '{gold['text']}' [{gold['start']}:{gold['end']}]")
            print(f"Boundary offset: ±{dist} chars, IoU: {error['iou']:.2f}")
            print()

    print(f"{'=' * 70}\n")


# =============================================================================
# Typed NER Evaluation
# =============================================================================


def evaluate_typed_spans(
    predictions: List[Dict],
    gold_standard: List[Dict],
    type_field: str = "type",
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Evaluate predicted spans with entity types against gold standard.
    A match requires both boundary overlap (IoU) AND matching type.

    Args:
        predictions: List of predicted entities [{"text": str, "start": int, "end": int, "type": str}, ...]
        gold_standard: List of gold entities with same format
        type_field: Name of the type field (default "type")
        iou_threshold: IoU threshold for boundary match (default 0.5)

    Returns:
        Dictionary with:
        - micro: {precision, recall, f1} - overall metrics
        - macro_f1: Average F1 across types
        - per_type: {type: {precision, recall, f1, tp, fp, fn}} - per-type metrics
        - counts: {tp, fp, fn} - overall counts
    """
    from collections import defaultdict

    if not gold_standard and not predictions:
        return {
            "micro": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "macro_f1": 1.0,
            "per_type": {},
            "counts": {"tp": 0, "fp": 0, "fn": 0},
        }

    if not gold_standard:
        return {
            "micro": {"precision": 0.0, "recall": 1.0, "f1": 0.0},
            "macro_f1": 0.0,
            "per_type": {},
            "counts": {"tp": 0, "fp": len(predictions), "fn": 0},
        }

    if not predictions:
        # Collect gold types for per-type stats
        per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for g in gold_standard:
            gtype = g.get(type_field, "O")
            per_type[gtype]["fn"] += 1

        per_type_metrics = {}
        for t, counts in per_type.items():
            per_type_metrics[t] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, **counts}

        return {
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "macro_f1": 0.0,
            "per_type": per_type_metrics,
            "counts": {"tp": 0, "fp": 0, "fn": len(gold_standard)},
        }

    # Group gold by type for faster lookup
    gold_by_type = defaultdict(list)
    for i, g in enumerate(gold_standard):
        gtype = g.get(type_field, "O")
        gold_by_type[gtype].append((i, g))

    # Track matches
    matched_gold = set()
    per_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Match predictions to gold
    for pred in predictions:
        ptype = pred.get(type_field, "O")
        p_start = pred.get("start", 0)
        p_end = pred.get("end", 0)

        # Look for matching gold with same type
        best_match_idx = None
        best_iou = 0.0

        for gold_idx, gold in gold_by_type.get(ptype, []):
            if gold_idx in matched_gold:
                continue

            g_start = gold.get("start", 0)
            g_end = gold.get("end", 0)

            # Calculate IoU
            inter_start = max(p_start, g_start)
            inter_end = min(p_end, g_end)
            intersection = max(0, inter_end - inter_start)
            union = (p_end - p_start) + (g_end - g_start) - intersection
            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_match_idx = gold_idx

        if best_match_idx is not None and best_iou >= iou_threshold:
            # True positive
            matched_gold.add(best_match_idx)
            per_type_counts[ptype]["tp"] += 1
        else:
            # False positive
            per_type_counts[ptype]["fp"] += 1

    # Count false negatives (unmatched gold)
    for gtype, gold_list in gold_by_type.items():
        for gold_idx, gold in gold_list:
            if gold_idx not in matched_gold:
                per_type_counts[gtype]["fn"] += 1

    # Calculate metrics
    def calc_prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    # Micro averages (sum all tp/fp/fn)
    total_tp = sum(c["tp"] for c in per_type_counts.values())
    total_fp = sum(c["fp"] for c in per_type_counts.values())
    total_fn = sum(c["fn"] for c in per_type_counts.values())
    micro_p, micro_r, micro_f1 = calc_prf(total_tp, total_fp, total_fn)

    # Per-type metrics
    per_type_metrics = {}
    for t, counts in per_type_counts.items():
        p, r, f1 = calc_prf(counts["tp"], counts["fp"], counts["fn"])
        per_type_metrics[t] = {"precision": p, "recall": r, "f1": f1, **counts}

    # Macro F1 (average F1 across types)
    type_f1s = [m["f1"] for m in per_type_metrics.values()]
    macro_f1 = sum(type_f1s) / len(type_f1s) if type_f1s else 0.0

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro_f1": macro_f1,
        "per_type": per_type_metrics,
        "counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }


def print_typed_eval_report(metrics: Dict, dataset_name: str = "Dataset"):
    """Pretty-print typed NER evaluation metrics"""
    print(f"\n{'=' * 70}")
    print(f"TYPED NER EVALUATION REPORT: {dataset_name}")
    print(f"{'=' * 70}\n")

    micro = metrics["micro"]
    print("OVERALL METRICS (Micro)")
    print("-" * 70)
    print(f"Precision: {micro['precision']:.1%}")
    print(f"Recall:    {micro['recall']:.1%}")
    print(f"F1 Score:  {micro['f1']:.1%}")
    print(f"\nMacro F1:  {metrics['macro_f1']:.1%}")
    print()

    print("CONFUSION")
    print("-" * 70)
    counts = metrics["counts"]
    print(f"True Positives:  {counts['tp']}")
    print(f"False Positives: {counts['fp']}")
    print(f"False Negatives: {counts['fn']}")
    print()

    if metrics.get("per_type"):
        print("PER-TYPE METRICS")
        print("-" * 70)
        for entity_type, type_metrics in sorted(metrics["per_type"].items()):
            print(f"\n{entity_type}:")
            print(
                f"  Precision: {type_metrics['precision']:.1%}  "
                f"Recall: {type_metrics['recall']:.1%}  "
                f"F1: {type_metrics['f1']:.1%}"
            )
            print(
                f"  TP: {type_metrics['tp']}  FP: {type_metrics['fp']}  FN: {type_metrics['fn']}"
            )

    print(f"\n{'=' * 70}\n")


# =============================================================================
# Key-Value Extraction Evaluation (for Transformation tasks)
# =============================================================================


def evaluate_key_value_extraction(
    predictions: List[Dict],
    gold_standard: List[Dict],
    key_field: str = "key",
    value_field: str = "value",
) -> Dict:
    """
    Evaluate key-value extraction by matching (key, value) pairs exactly.

    A prediction is correct if both key and value match a gold item.
    Ignores extra fields like start/end.

    Args:
        predictions: List of predicted items [{"key": str, "value": str, ...}, ...]
        gold_standard: List of gold items with same format
        key_field: Name of the key field (default "key")
        value_field: Name of the value field (default "value")

    Returns:
        Dictionary with:
        - micro: {precision, recall, f1}
        - per_key: {key: {precision, recall, f1, tp, fp, fn}}
        - counts: {tp, fp, fn}
        - matches: List of matched (pred, gold) pairs
        - errors: List of unmatched predictions and gold items
    """
    from collections import defaultdict

    if not gold_standard and not predictions:
        return {
            "micro": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "per_key": {},
            "counts": {"tp": 0, "fp": 0, "fn": 0},
            "matches": [],
            "errors": [],
        }

    # Build gold lookup: key -> list of values
    gold_by_key = defaultdict(list)
    for i, g in enumerate(gold_standard):
        k = g.get(key_field, "")
        v = g.get(value_field, "")
        gold_by_key[k].append((i, v, g))

    # Track matches
    matched_gold = set()
    per_key_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matches = []
    fp_errors = []

    # Match predictions to gold
    for pred in predictions:
        pk = pred.get(key_field, "")
        pv = pred.get(value_field, "")

        # Look for matching gold with same key and value
        found_match = False
        for gold_idx, gv, gold in gold_by_key.get(pk, []):
            if gold_idx in matched_gold:
                continue

            # Exact value match (or normalized)
            if pv == gv or pv.strip() == gv.strip():
                matched_gold.add(gold_idx)
                per_key_counts[pk]["tp"] += 1
                matches.append({"predicted": pred, "gold": gold})
                found_match = True
                break

        if not found_match:
            per_key_counts[pk]["fp"] += 1
            fp_errors.append({"predicted": pred, "reason": "no matching gold"})

    # Count false negatives (unmatched gold)
    fn_errors = []
    for k, gold_list in gold_by_key.items():
        for gold_idx, gv, gold in gold_list:
            if gold_idx not in matched_gold:
                per_key_counts[k]["fn"] += 1
                fn_errors.append({"gold": gold, "reason": "not predicted"})

    # Calculate metrics
    def calc_prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    # Micro averages
    total_tp = sum(c["tp"] for c in per_key_counts.values())
    total_fp = sum(c["fp"] for c in per_key_counts.values())
    total_fn = sum(c["fn"] for c in per_key_counts.values())
    micro_p, micro_r, micro_f1 = calc_prf(total_tp, total_fp, total_fn)

    # Per-key metrics
    per_key_metrics = {}
    for k, counts in per_key_counts.items():
        p, r, f1 = calc_prf(counts["tp"], counts["fp"], counts["fn"])
        per_key_metrics[k] = {"precision": p, "recall": r, "f1": f1, **counts}

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "per_key": per_key_metrics,
        "counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
        "matches": matches,
        "errors": fp_errors + fn_errors,
    }


def print_key_value_eval_report(metrics: Dict, dataset_name: str = "Dataset"):
    """Pretty-print key-value extraction evaluation metrics"""
    print(f"\n{'=' * 70}")
    print(f"KEY-VALUE EXTRACTION REPORT: {dataset_name}")
    print(f"{'=' * 70}\n")

    micro = metrics["micro"]
    print("OVERALL METRICS")
    print("-" * 70)
    print(f"Precision: {micro['precision']:.1%}")
    print(f"Recall:    {micro['recall']:.1%}")
    print(f"F1 Score:  {micro['f1']:.1%}")
    print()

    counts = metrics["counts"]
    print(f"Correct:   {counts['tp']}")
    print(f"Spurious:  {counts['fp']} (predicted but wrong)")
    print(f"Missing:   {counts['fn']} (gold but not predicted)")
    print()

    if metrics.get("per_key"):
        print("PER-KEY METRICS")
        print("-" * 70)
        for key, key_metrics in sorted(metrics["per_key"].items()):
            status = (
                "✓"
                if key_metrics["f1"] == 1.0
                else "✗"
                if key_metrics["f1"] == 0.0
                else "~"
            )
            print(
                f"{status} {key}: P={key_metrics['precision']:.0%} R={key_metrics['recall']:.0%} F1={key_metrics['f1']:.0%}  "
                f"(TP:{key_metrics['tp']} FP:{key_metrics['fp']} FN:{key_metrics['fn']})"
            )

    print(f"\n{'=' * 70}\n")
