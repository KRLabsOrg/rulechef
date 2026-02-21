"""Evaluation metrics for RuleChef: entity-level, per-class, per-rule."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rulechef.core import (
    Correction,
    Dataset,
    Rule,
    TaskType,
)

# ============================================================================
# Metric dataclasses
# ============================================================================


@dataclass
class ClassMetrics:
    """Precision / recall / F1 for a single entity type or key.

    Attributes:
        label: The class/entity type name.
        tp: True positive count.
        fp: False positive count.
        fn: False negative count.
    """

    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class EvalResult:
    """Rich evaluation result across a dataset.

    Attributes:
        micro_precision: Entity-level micro-averaged precision.
        micro_recall: Entity-level micro-averaged recall.
        micro_f1: Entity-level micro-averaged F1 score.
        macro_f1: Macro F1 (unweighted average of per-class F1 scores).
        per_class: Per-class precision/recall/F1 breakdown.
        exact_match: Fraction of documents with perfect output (0.0-1.0).
        total_tp: Total true positive count across all classes.
        total_fp: Total false positive count across all classes.
        total_fn: Total false negative count across all classes.
        total_docs: Number of documents evaluated.
        failures: List of failure dicts with keys 'input', 'expected', 'got',
            'is_correction'. Used by the refinement loop to generate patches.
    """

    # Entity-level micro-averaged
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0

    # Macro F1 (average of per-class F1)
    macro_f1: float = 0.0

    # Per-class breakdown
    per_class: list[ClassMetrics] = field(default_factory=list)

    # Document-level exact match (legacy compat)
    exact_match: float = 0.0

    # Raw counts
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_docs: int = 0

    # Failures for refinement prompts
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "micro_precision": round(self.micro_precision, 4),
            "micro_recall": round(self.micro_recall, 4),
            "micro_f1": round(self.micro_f1, 4),
            "macro_f1": round(self.macro_f1, 4),
            "exact_match": round(self.exact_match, 4),
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_fn": self.total_fn,
            "total_docs": self.total_docs,
            "per_class": [c.to_dict() for c in self.per_class],
            "failures": self.failures,
        }


@dataclass
class RuleMetrics:
    """Evaluation of a single rule in isolation.

    Attributes:
        rule_id: Unique identifier of the evaluated rule.
        rule_name: Human-readable name of the rule.
        precision: Precision of this rule alone (TP / (TP + FP)).
        recall: Recall of this rule alone (covered / total expected entities).
        f1: F1 score derived from precision and recall.
        matches: Total number of entities this rule produced.
        true_positives: Entities that matched an expected entity.
        false_positives: Entities that did not match any expected entity.
        covered_expected: How many expected entities this rule correctly finds.
        total_expected: Total expected entities across the full dataset.
        per_class: Per-class breakdown of TP/FP/FN for this rule.
        sample_matches: Up to 10 sample match dicts showing rule behavior.
    """

    rule_id: str
    rule_name: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    matches: int = 0
    true_positives: int = 0
    false_positives: int = 0
    covered_expected: int = 0  # how many expected entities this rule finds
    total_expected: int = 0
    per_class: list[ClassMetrics] = field(default_factory=list)
    sample_matches: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "matches": self.matches,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "covered_expected": self.covered_expected,
            "total_expected": self.total_expected,
            "per_class": [c.to_dict() for c in self.per_class],
            "sample_matches": self.sample_matches[:10],
        }


# ============================================================================
# Core evaluation helpers
# ============================================================================


def _get_entities(output: dict, task_type: TaskType) -> list[dict]:
    """Extract the entity/span list from an output dict."""
    if task_type == TaskType.NER:
        for key in ("entities", "spans", "ner"):
            if key in output:
                return output[key] or []
        return []
    elif task_type == TaskType.EXTRACTION:
        # Check common keys — rules may use custom output_key
        for key in ("spans", "entities", "items", "results"):
            if key in output:
                val = output[key]
                if isinstance(val, list):
                    return val
        # Fallback: first list value in the output
        for v in output.values():
            if isinstance(v, list):
                return v
        return []
    elif task_type == TaskType.TRANSFORMATION:
        # Collect all list values from output
        items = []
        for v in output.values():
            if isinstance(v, list):
                items.extend(v)
        return items
    elif task_type == TaskType.CLASSIFICATION:
        label = str(output.get("label", "")).strip()
        return [{"label": label}] if label else []
    return []


def _entity_key_exact(e: dict, task_type: TaskType) -> tuple:
    """Key for exact span matching (text + type + start + end)."""
    if task_type == TaskType.CLASSIFICATION:
        return (str(e.get("label", "")).lower().strip(),)
    etype = e.get("type") or e.get("label", "")
    return (e.get("text", ""), etype, e.get("start"), e.get("end"))


def _entity_key_text(e: dict, task_type: TaskType) -> tuple:
    """Key for text-only matching (text + type, ignore position)."""
    if task_type == TaskType.CLASSIFICATION:
        return (str(e.get("label", "")).lower().strip(),)
    etype = e.get("type") or e.get("label", "")
    return (e.get("text", ""), etype)


def _entity_type(e: dict) -> str:
    """Get entity type/label."""
    return e.get("type") or e.get("label") or "_NONE"


def _match_entities(
    predicted: list[dict],
    expected: list[dict],
    task_type: TaskType,
    mode: str = "text",
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """
    Match predicted entities to expected entities.

    Returns:
        (matched_pairs, false_positives, false_negatives)
    """
    key_fn = _entity_key_exact if mode == "exact" else _entity_key_text

    matched_pairs = []
    unmatched_pred = []
    used_gold = set()

    for pred in predicted:
        pred_key = key_fn(pred, task_type)
        best_idx = None
        for i, gold in enumerate(expected):
            if i in used_gold:
                continue
            if key_fn(gold, task_type) == pred_key:
                best_idx = i
                break

        if best_idx is not None:
            matched_pairs.append((pred, expected[best_idx]))
            used_gold.add(best_idx)
        else:
            unmatched_pred.append(pred)

    unmatched_gold = [g for i, g in enumerate(expected) if i not in used_gold]
    return matched_pairs, unmatched_pred, unmatched_gold


# ============================================================================
# Dataset-level evaluation
# ============================================================================


def evaluate_dataset(
    rules: list[Rule],
    dataset: Dataset,
    apply_rules_fn,
    mode: str = "text",
) -> EvalResult:
    """Evaluate rules against a full dataset, producing entity-level metrics.

    Args:
        rules: Rules to evaluate.
        dataset: Dataset with examples and corrections.
        apply_rules_fn: Callable(rules, input_data, task_type, text_field) -> output_dict.
        mode: 'text' (match by text+type) or 'exact' (match by text+type+start+end).

    Returns:
        EvalResult with micro/macro metrics, per-class breakdown, exact match
        rate, and a list of failure dicts for refinement.
    """
    task_type = dataset.task.type
    all_data = dataset.get_all_training_data()
    total_docs = len(all_data)

    # Per-class accumulators
    class_counts: dict[str, ClassMetrics] = defaultdict(lambda: ClassMetrics(label=""))
    exact_match_count = 0
    failures = []

    for item in all_data:
        extracted = apply_rules_fn(rules, item.input, task_type, dataset.task.text_field)
        expected_output = item.expected_output

        pred_entities = _get_entities(extracted, task_type)
        gold_entities = _get_entities(expected_output, task_type)

        matched, fp_list, fn_list = _match_entities(pred_entities, gold_entities, task_type, mode)

        # Document-level exact match
        if not fp_list and not fn_list:
            exact_match_count += 1
        else:
            failures.append(
                {
                    "input": item.input,
                    "expected": expected_output,
                    "got": extracted,
                    "is_correction": isinstance(item, Correction),
                }
            )

        # Accumulate per-class TP
        for _pred, gold in matched:
            cls = _entity_type(gold)
            if class_counts[cls].label == "":
                class_counts[cls].label = cls
            class_counts[cls].tp += 1

        # Accumulate per-class FP
        for pred in fp_list:
            cls = _entity_type(pred)
            if class_counts[cls].label == "":
                class_counts[cls].label = cls
            class_counts[cls].fp += 1

        # Accumulate per-class FN
        for gold in fn_list:
            cls = _entity_type(gold)
            if class_counts[cls].label == "":
                class_counts[cls].label = cls
            class_counts[cls].fn += 1

    # Compute aggregates
    per_class = sorted(class_counts.values(), key=lambda c: c.label)
    total_tp = sum(c.tp for c in per_class)
    total_fp = sum(c.fp for c in per_class)
    total_fn = sum(c.fn for c in per_class)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    class_f1s = [c.f1 for c in per_class]
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0

    return EvalResult(
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        per_class=per_class,
        exact_match=exact_match_count / total_docs if total_docs > 0 else 0.0,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_docs=total_docs,
        failures=failures,
    )


# ============================================================================
# Per-rule evaluation
# ============================================================================


def evaluate_rules_individually(
    rules: list[Rule],
    dataset: Dataset,
    apply_rules_fn,
    mode: str = "text",
    max_samples: int = 10,
) -> list[RuleMetrics]:
    """Evaluate each rule in isolation against the dataset.

    For each rule, runs it alone and computes how many expected entities it
    produces correctly (TP), how many spurious entities it produces (FP),
    and how many expected entities it misses (recall denominator).

    Args:
        rules: Rules to evaluate individually.
        dataset: Dataset with examples and corrections.
        apply_rules_fn: Callable(rules, input_data, task_type, text_field) -> output_dict.
        mode: 'text' or 'exact'.
        max_samples: Max sample matches to store per rule.

    Returns:
        List[RuleMetrics], one entry per rule, with per-rule precision/recall/F1,
        match counts, per-class breakdown, and sample matches.
    """
    task_type = dataset.task.type
    all_data = dataset.get_all_training_data()

    # Count total expected entities across all docs (for recall denominator)
    total_expected = 0
    for item in all_data:
        gold_entities = _get_entities(item.expected_output, task_type)
        total_expected += len(gold_entities)

    results = []

    for rule in rules:
        class_counts: dict[str, ClassMetrics] = defaultdict(lambda: ClassMetrics(label=""))
        sample_matches = []
        rule_total_matches = 0
        rule_covered = 0

        for item in all_data:
            extracted = apply_rules_fn([rule], item.input, task_type, dataset.task.text_field)
            expected_output = item.expected_output

            pred_entities = _get_entities(extracted, task_type)
            gold_entities = _get_entities(expected_output, task_type)

            matched, fp_list, fn_list = _match_entities(
                pred_entities, gold_entities, task_type, mode
            )

            rule_total_matches += len(pred_entities)
            rule_covered += len(matched)

            for _pred, gold in matched:
                cls = _entity_type(gold)
                if class_counts[cls].label == "":
                    class_counts[cls].label = cls
                class_counts[cls].tp += 1

            for pred in fp_list:
                cls = _entity_type(pred)
                if class_counts[cls].label == "":
                    class_counts[cls].label = cls
                class_counts[cls].fp += 1

            # Note: we don't count FN per-rule since a single rule isn't
            # expected to find everything. But we track it for completeness.
            for gold in fn_list:
                cls = _entity_type(gold)
                if class_counts[cls].label == "":
                    class_counts[cls].label = cls
                class_counts[cls].fn += 1

            # Collect sample matches
            if pred_entities and len(sample_matches) < max_samples:
                sample_matches.append(
                    {
                        "input": item.input,
                        "rule_output": pred_entities,
                        "expected": gold_entities,
                        "tp": len(matched),
                        "fp": len(fp_list),
                    }
                )

        per_class = sorted(class_counts.values(), key=lambda c: c.label)
        rule_tp = sum(c.tp for c in per_class)
        rule_fp = sum(c.fp for c in per_class)

        precision = rule_tp / (rule_tp + rule_fp) if (rule_tp + rule_fp) > 0 else 0.0
        recall = rule_covered / total_expected if total_expected > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append(
            RuleMetrics(
                rule_id=rule.id,
                rule_name=rule.name,
                precision=precision,
                recall=recall,
                f1=f1,
                matches=rule_total_matches,
                true_positives=rule_tp,
                false_positives=rule_fp,
                covered_expected=rule_covered,
                total_expected=total_expected,
                per_class=per_class,
                sample_matches=sample_matches,
            )
        )

    return results


# ============================================================================
# Pretty printing
# ============================================================================


def print_eval_result(result: EvalResult, name: str = "Dataset") -> None:
    """Pretty-print an EvalResult."""
    print(f"\n{'=' * 70}")
    print(f"EVALUATION: {name}")
    print(f"{'=' * 70}")

    print(f"\n  Documents: {result.total_docs}")
    print(f"  Exact match: {result.exact_match:.1%}")
    print("\n  Entity-level (micro):")
    print(f"    Precision: {result.micro_precision:.1%}")
    print(f"    Recall:    {result.micro_recall:.1%}")
    print(f"    F1:        {result.micro_f1:.1%}")
    print(f"  Macro F1:    {result.macro_f1:.1%}")

    if result.per_class:
        print(f"\n  {'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
        print(f"  {'-' * 60}")
        for c in result.per_class:
            print(
                f"  {c.label:<20} {c.precision:>5.0%} {c.recall:>5.0%} {c.f1:>5.0%}"
                f"  {c.tp:>4} {c.fp:>4} {c.fn:>4}"
            )

    print(f"\n{'=' * 70}\n")


def print_rule_metrics(rule_metrics: list[RuleMetrics]) -> None:
    """Pretty-print per-rule metrics."""
    print(f"\n{'=' * 70}")
    print("PER-RULE METRICS")
    print(f"{'=' * 70}")
    print(f"\n  {'Rule':<30} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'Match':>5}")
    print(f"  {'-' * 70}")

    for rm in sorted(rule_metrics, key=lambda r: r.f1, reverse=True):
        name = rm.rule_name[:28] + ".." if len(rm.rule_name) > 30 else rm.rule_name
        print(
            f"  {name:<30} {rm.precision:>5.0%} {rm.recall:>5.0%} {rm.f1:>5.0%}"
            f"  {rm.true_positives:>4} {rm.false_positives:>4} {rm.matches:>5}"
        )

    print(f"\n{'=' * 70}\n")


# ============================================================================
# Legacy compatibility wrappers
# ============================================================================


def span_iou(span1: dict, span2: dict) -> float:
    """Calculate Intersection over Union for two spans."""
    s1_start = span1.get("start", 0)
    s1_end = span1.get("end", 0)
    s2_start = span2.get("start", 0)
    s2_end = span2.get("end", 0)
    inter_start = max(s1_start, s2_start)
    inter_end = min(s1_end, s2_end)
    intersection = max(0, inter_end - inter_start)
    union = (s1_end - s1_start) + (s2_end - s2_start) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def evaluate_typed_spans(
    predictions: list[dict],
    gold_standard: list[dict],
    type_field: str = "type",
    iou_threshold: float = 0.5,
) -> dict:
    """Legacy typed span evaluation (kept for backward compatibility)."""
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

    gold_by_type = defaultdict(list)
    for i, g in enumerate(gold_standard):
        gtype = g.get(type_field, "O")
        gold_by_type[gtype].append((i, g))

    matched_gold = set()
    per_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred in predictions:
        ptype = pred.get(type_field, "O")
        p_start = pred.get("start", 0)
        p_end = pred.get("end", 0)
        best_match_idx = None
        best_iou = 0.0
        for gold_idx, gold in gold_by_type.get(ptype, []):
            if gold_idx in matched_gold:
                continue
            g_start = gold.get("start", 0)
            g_end = gold.get("end", 0)
            inter_start = max(p_start, g_start)
            inter_end = min(p_end, g_end)
            intersection = max(0, inter_end - inter_start)
            union = (p_end - p_start) + (g_end - g_start) - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_match_idx = gold_idx
        if best_match_idx is not None and best_iou >= iou_threshold:
            matched_gold.add(best_match_idx)
            per_type_counts[ptype]["tp"] += 1
        else:
            per_type_counts[ptype]["fp"] += 1

    for gtype, gold_list in gold_by_type.items():
        for gold_idx, _gold in gold_list:
            if gold_idx not in matched_gold:
                per_type_counts[gtype]["fn"] += 1

    def calc_prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    total_tp = sum(c["tp"] for c in per_type_counts.values())
    total_fp = sum(c["fp"] for c in per_type_counts.values())
    total_fn = sum(c["fn"] for c in per_type_counts.values())
    micro_p, micro_r, micro_f1 = calc_prf(total_tp, total_fp, total_fn)

    per_type_metrics = {}
    for t, counts in per_type_counts.items():
        p, r, f1 = calc_prf(counts["tp"], counts["fp"], counts["fn"])
        per_type_metrics[t] = {"precision": p, "recall": r, "f1": f1, **counts}

    type_f1s = [m["f1"] for m in per_type_metrics.values()]
    macro_f1 = sum(type_f1s) / len(type_f1s) if type_f1s else 0.0

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro_f1": macro_f1,
        "per_type": per_type_metrics,
        "counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }
