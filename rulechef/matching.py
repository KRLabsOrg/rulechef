"""Output matching - compares rule outputs to expected outputs"""

import json
from typing import Dict, Optional, Callable, Any
from collections import defaultdict
from rulechef.core import Correction, Dataset, Rule, RuleFormat, TaskType

from rulechef.core import TaskType, DEFAULT_OUTPUT_KEYS

# Type alias for matcher functions
OutputMatcher = Callable[[Dict[str, Any], Dict[str, Any]], bool]


def check_overlap(pred, gold, threshold=1):
    pred_start, pred_end, pred_type = pred["start"], pred["end"], pred["type"]
    gold_start, gold_end, gold_type = gold["start"], gold["end"], gold["type"]

    if threshold == 1:
        return pred["text"] == gold["text"] and pred_type == gold_type

    overlap_start = max(pred_start, gold_start)

    overlap_end = min(pred_end, gold_end)

    # no overlap
    if overlap_end <= overlap_start:
        return False

    overlap = overlap_end - overlap_start
    # print(pred["text"], gold["text"],overlap)
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    iou = overlap / union
    # print(iou,"->>>>",threshold,iou >= threshold)

    return iou >= threshold and pred_type == gold_type


def evaluate_rules_ner(all_data, rules, apply_rules_fn, task, threshold):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    labels = set()
    failures = []

    def get_entities(d):
        for key in ["entities", "spans", "ner"]:
            if key in d:
                return d[key]
        return []

    def get_class(e):
        return e.get("type") or e.get("label", "")

    for item in all_data:
        predicted_spans = apply_rules_fn(rules, item.input, task.type, task.text_field)

        gold_spans = get_entities(item.expected_output)
        predicted_spans = (
            get_entities(predicted_spans)
            if isinstance(predicted_spans, dict)
            else predicted_spans
        )

        for g in gold_spans:
            labels.add(get_class(g))
        for p in predicted_spans:
            labels.add(get_class(p))
        print("gold_spans:", gold_spans)
        print("predicted_spans:", predicted_spans)
        print("expected_output keys:", item.expected_output.keys())
        matched_gold = set()
        print(labels)
        for pred in predicted_spans:
            found_match = False
            pred_class = get_class(pred)
            for j, gold in enumerate(gold_spans):
                if j in matched_gold:
                    continue
                if check_overlap(pred, gold, threshold):
                    tp[get_class(gold)] += 1
                    matched_gold.add(j)
                    found_match = True
                    break

            if not found_match:
                fp[pred_class] += 1

                failures.append(
                    {
                        "input": item.input,
                        "expected": [],
                        "got": pred["text"],
                        "is_correction": isinstance(item, Correction),
                    }
                )

        for j, gold in enumerate(gold_spans):
            if j not in matched_gold:
                fn[get_class(gold)] += 1
                failures.append(
                    {
                        "input": item.input,
                        "expected": gold["text"],
                        "got": [],
                        "is_correction": isinstance(item, Correction),
                    }
                )
    per_class = {}
    for label in sorted(labels):
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp[label],
            "fp": fp[label],
            "fn": fn[label],
        }
        print(f"{label} — Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    TP = sum(tp.values())
    FP = sum(fp.values())
    FN = sum(fn.values())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    total = TP + FP + FN
    accuracy = TP / total if total > 0 else 0.0
    print("-----------\nOverall NER Metrics")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return {
        "total": total,
        "correct": TP,
        "accuracy": accuracy,
        "failures": failures,
        "ner_metrics": {
            "overall": {"precision": precision, "recall": recall, "f1": f1},
            "per_class": per_class,
        },
    }


def outputs_match(
    expected: Dict,
    actual: Dict,
    task_type: TaskType = TaskType.EXTRACTION,
    custom_matcher: Optional[OutputMatcher] = None,
    matching_mode: str = "text",
    threshold: float = 0.5,
) -> bool:
    """
    Check if two outputs match.

    Args:
        expected: The expected/ground truth output
        actual: The actual output from rules
        task_type: Type of task (used to select default matcher)
        custom_matcher: Optional custom comparison function.
                       If provided, overrides the default matcher.

    Returns:
        True if outputs match, False otherwise
    """
    # Use custom matcher if provided
    if custom_matcher is not None:
        return custom_matcher(expected, actual)

    # Otherwise use default matcher based on task type
    if task_type == TaskType.EXTRACTION:
        spans1 = expected.get("spans", [])
        spans2 = actual.get("spans", [])

        if len(spans1) != len(spans2):
            return False

        # Check match (order independent)
        def _text(span):
            if isinstance(span, dict):
                return span.get("text", "")
            if hasattr(span, "text"):
                return getattr(span, "text")
            return str(span)

        if matching_mode == "exact":

            def _span_key(span):
                if isinstance(span, dict):
                    return (
                        span.get("text", ""),
                        span.get("start", 0),
                        span.get("end", 0),
                    )
                if hasattr(span, "text"):
                    return (
                        getattr(span, "text"),
                        getattr(span, "start", 0),
                        getattr(span, "end", 0),
                    )
                return (str(span), 0, 0)

            keys1 = sorted([_span_key(s) for s in spans1])
            keys2 = sorted([_span_key(s) for s in spans2])
            return keys1 == keys2

        texts1 = sorted([_text(s) for s in spans1])
        texts2 = sorted([_text(s) for s in spans2])

        return texts1 == texts2

    elif task_type == TaskType.NER:
        # NER: Compare entities including type
        # Use canonical key "entities", fallback to legacy keys for compatibility
        canonical_key = DEFAULT_OUTPUT_KEYS[TaskType.NER]  # "entities"

        def get_entities(d):
            if canonical_key in d:
                return d[canonical_key]
            # Fallback for legacy data
            for fallback in ["spans", "ner"]:
                if fallback in d:
                    return d[fallback]
            return []

        if len(entities1) != len(entities2):
            return False

        def entity_key(e):
            if isinstance(e, dict):
                # Support both "type" and "label" for entity type field
                ent_type = e.get("type") or e.get("label", "")
                return (e.get("text", ""), ent_type)
            if hasattr(e, "text"):
                ent_type = getattr(e, "type", None) or getattr(e, "label", "")
                return (getattr(e, "text"), ent_type)
            return (str(e), "")

        set1 = sorted([entity_key(e) for e in entities1])
        set2 = sorted([entity_key(e) for e in entities2])

        return set1 == set2

    elif task_type == TaskType.CLASSIFICATION:
        # Compare labels (case insensitive)
        label1 = str(expected.get("label", "")).lower().strip()
        label2 = str(actual.get("label", "")).lower().strip()
        return label1 == label2

    elif task_type == TaskType.TRANSFORMATION:
        # Transformation: Compare all array keys with relaxed matching
        for key in set(expected.keys()) | set(actual.keys()):
            val1 = expected.get(key)
            val2 = actual.get(key)

            if isinstance(val1, list) and isinstance(val2, list):
                if len(val1) != len(val2):
                    return False
                try:
                    set1 = sorted([json.dumps(v, sort_keys=True) for v in val1])
                    set2 = sorted([json.dumps(v, sort_keys=True) for v in val2])
                    if set1 != set2:
                        return False
                except:
                    if val1 != val2:
                        return False
            elif val1 != val2:
                return False
        return True

    else:
        # Other: Exact match
        return expected == actual
