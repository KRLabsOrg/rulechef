"""Rule ranking: evaluate rules alone and together, resolve conflicts.

Two rules can match the same input and disagree; execution order decides
which wins. This module measures each rule's standalone precision and its
marginal contribution to the full ensemble, stamps validated stats onto the
rules (which the executor uses as a tie-breaker within the same priority),
and can prune rules that hurt the ensemble.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from rulechef.core import Dataset, Rule
from rulechef.evaluation import evaluate_dataset, evaluate_rules_individually


def wilson_lower_bound(precision: float, support: int, z: float = 1.96) -> float:
    """Wilson score lower bound for a measured precision.

    Raw precision is unreliable at low support: 1/1 correct is "100%
    precision" but tells you almost nothing. The Wilson lower bound
    discounts the estimate by sample size — use it instead of raw
    validated_precision when deciding whether to trust a rule for routing.

    Args:
        precision: Observed precision (0.0-1.0).
        support: Number of predictions the estimate is based on.
        z: Confidence z-score (1.96 = 95% confidence).

    Returns:
        Lower bound on the true precision; 0.0 when support is 0.
    """
    if support <= 0:
        return 0.0
    denominator = 1 + z * z / support
    center = precision + z * z / (2 * support)
    spread = z * math.sqrt((precision * (1 - precision) + z * z / (4 * support)) / support)
    return max(0.0, (center - spread) / denominator)


@dataclass
class RuleRanking:
    """Ranking entry for one rule.

    Attributes:
        rule_id: Unique identifier of the rule.
        rule_name: Human-readable rule name.
        solo_precision: Precision of the rule evaluated in isolation.
        solo_recall: Recall of the rule evaluated in isolation.
        solo_f1: F1 of the rule evaluated in isolation.
        support: Number of predictions the rule made alone (TP + FP).
        marginal_f1: Ensemble micro F1 minus the ensemble micro F1 without
            this rule. Positive means the rule helps the ensemble; negative
            means the ensemble is better off without it. None when the
            ablation pass was skipped.
    """

    rule_id: str
    rule_name: str
    solo_precision: float = 0.0
    solo_recall: float = 0.0
    solo_f1: float = 0.0
    support: int = 0
    marginal_f1: float | None = None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "solo_precision": round(self.solo_precision, 4),
            "solo_recall": round(self.solo_recall, 4),
            "solo_f1": round(self.solo_f1, 4),
            "support": self.support,
            "marginal_f1": round(self.marginal_f1, 4) if self.marginal_f1 is not None else None,
        }


@dataclass
class RankingReport:
    """Result of rank_rules(): ensemble metrics plus per-rule rankings.

    Attributes:
        ensemble_precision: Micro precision of all rules together.
        ensemble_recall: Micro recall of all rules together.
        ensemble_f1: Micro F1 of all rules together.
        rankings: Per-rule rankings, sorted most valuable first
            (by marginal F1 when available, then solo precision).
    """

    ensemble_precision: float = 0.0
    ensemble_recall: float = 0.0
    ensemble_f1: float = 0.0
    rankings: list[RuleRanking] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ensemble_precision": round(self.ensemble_precision, 4),
            "ensemble_recall": round(self.ensemble_recall, 4),
            "ensemble_f1": round(self.ensemble_f1, 4),
            "rankings": [r.to_dict() for r in self.rankings],
        }


def rank_rules(
    rules: list[Rule],
    dataset: Dataset,
    apply_rules_fn: Callable,
    mode: str = "text",
    compute_marginal: bool = True,
    stamp_validated_stats: bool = True,
) -> RankingReport:
    """Evaluate rules alone and as an ensemble, producing a ranking.

    For each rule this computes standalone precision/recall/F1, and
    optionally its marginal contribution: how much the ensemble micro F1
    drops when the rule is removed. The ranking answers "which rule should
    win a conflict" (solo precision) and "which rules earn their place"
    (marginal F1).

    Args:
        rules: Rules to rank.
        dataset: Dataset to evaluate against — use a held-out dev set when
            available so validated stats measure generalization, not memorization.
        apply_rules_fn: Callable(rules, input_data, task_type, text_field) -> output dict.
        mode: Matching mode ('text', 'exact', or 'partial').
        compute_marginal: If True, run the leave-one-out ablation pass.
            Costs one full-dataset evaluation per rule; disable for very
            large rule sets or datasets.
        stamp_validated_stats: If True, write each rule's solo precision and
            support onto rule.validated_precision / rule.validated_support.
            The executor uses validated_precision to order rules within the
            same priority, so the empirically more precise rule wins conflicts.

    Returns:
        RankingReport with ensemble metrics and per-rule rankings sorted
        most valuable first.
    """
    if not rules:
        return RankingReport()

    ensemble_eval = evaluate_dataset(rules, dataset, apply_rules_fn, mode=mode)
    solo_metrics = evaluate_rules_individually(rules, dataset, apply_rules_fn, mode=mode)
    solo_by_id = {m.rule_id: m for m in solo_metrics}

    rankings = []
    for rule in rules:
        solo = solo_by_id.get(rule.id)
        entry = RuleRanking(rule_id=rule.id, rule_name=rule.name)
        if solo:
            entry.solo_precision = solo.precision
            entry.solo_recall = solo.recall
            entry.solo_f1 = solo.f1
            entry.support = solo.true_positives + solo.false_positives

        if compute_marginal and len(rules) > 1:
            without = [r for r in rules if r.id != rule.id]
            without_eval = evaluate_dataset(without, dataset, apply_rules_fn, mode=mode)
            entry.marginal_f1 = ensemble_eval.micro_f1 - without_eval.micro_f1

        if stamp_validated_stats:
            rule.validated_precision = entry.solo_precision
            rule.validated_support = entry.support

        rankings.append(entry)

    rankings.sort(
        key=lambda r: (
            r.marginal_f1 if r.marginal_f1 is not None else 0.0,
            r.solo_precision,
            r.support,
        ),
        reverse=True,
    )

    return RankingReport(
        ensemble_precision=ensemble_eval.micro_precision,
        ensemble_recall=ensemble_eval.micro_recall,
        ensemble_f1=ensemble_eval.micro_f1,
        rankings=rankings,
    )


def prune_harmful_rules(
    rules: list[Rule],
    report: RankingReport,
    min_marginal_f1: float = 0.0,
    min_support: int = 1,
) -> tuple[list[Rule], list[Rule]]:
    """Split rules into (kept, dropped) based on marginal contribution and support.

    A rule is dropped when its marginal F1 is known and below
    min_marginal_f1 — i.e. the ensemble measurably does better without it —
    or when its validated_support is below min_support. A zero-support rule
    never matches anything on the dev split, so removing it leaves ensemble
    F1 unchanged (marginal F1 of exactly 0.0, which is not < min_marginal_f1)
    and it would otherwise survive as dead weight.

    Returns:
        Tuple of (kept_rules, dropped_rules).
    """
    harmful_ids = {
        r.rule_id
        for r in report.rankings
        if r.marginal_f1 is not None and r.marginal_f1 < min_marginal_f1
    }
    zero_support_ids = {r.id for r in rules if r.validated_support < min_support}
    dropped_ids = harmful_ids | zero_support_ids
    kept = [r for r in rules if r.id not in dropped_ids]
    dropped = [r for r in rules if r.id in dropped_ids]
    return kept, dropped


def print_ranking_report(report: RankingReport) -> None:
    """Pretty-print a RankingReport."""
    print(f"\n{'=' * 76}")
    print("RULE RANKING")
    print(f"{'=' * 76}")
    print(
        f"\n  Ensemble: P={report.ensemble_precision:.1%} "
        f"R={report.ensemble_recall:.1%} F1={report.ensemble_f1:.1%}"
    )
    print(f"\n  {'Rule':<32} {'SoloP':>6} {'SoloR':>6} {'SoloF1':>6} {'Supp':>5} {'ΔF1':>7}")
    print(f"  {'-' * 70}")
    for r in report.rankings:
        name = r.rule_name[:30] + ".." if len(r.rule_name) > 32 else r.rule_name
        marginal = f"{r.marginal_f1:+.3f}" if r.marginal_f1 is not None else "    — "
        print(
            f"  {name:<32} {r.solo_precision:>5.0%} {r.solo_recall:>5.0%} "
            f"{r.solo_f1:>5.0%} {r.support:>5} {marginal:>7}"
        )
    print(f"\n{'=' * 76}\n")
