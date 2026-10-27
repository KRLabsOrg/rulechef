"""Search synthesis: candidates -> deterministic measurement -> beam search.

No LLM call anywhere. Per label, candidates are generated from the training
positives, measured alone with the executor's semantics, pooled, optionally
filtered by a judge, and assembled into an ordered ruleset by beam search.
Rules supplied by the caller (for example LLM-written ones, ``seed_rules``)
join the candidate pool and compete on the same terms.
"""

from __future__ import annotations

import dataclasses
import random
import time
import uuid
from collections import Counter
from dataclasses import dataclass

from rulechef.core import Dataset, Rule, RuleFormat, TaskType
from rulechef.evaluation import _get_entities
from rulechef.executor import RuleExecutor
from rulechef.search.beam import beam_search
from rulechef.search.candidates import (
    CandidateKey,
    Positive,
    context_candidates,
    group_candidates,
    lookahead_candidates,
    mutation_candidates,
)
from rulechef.search.judge import Judge, RuleEvidence
from rulechef.search.measure import Candidate, build_pool, dedup, measure, rank_score

_SUPPORTED = (TaskType.NER, TaskType.EXTRACTION)


@dataclass
class SearchConfig:
    """Settings for search synthesis.

    Attributes:
        labels: Labels to learn rules for, in output order (earlier labels win
            position conflicts in the executor). None learns every label in the
            data, most frequent first.
        min_support: A candidate must match at least this many gold spans, and a
            signature group must have this many positives, to be considered.
        beam_width: Beam width of the rule-list search.
        max_rules_per_label: Search depth, the largest ruleset per label.
        pool_size: Candidates per label the search chooses from.
        per_origin: Pool slots reserved per origin group before filling by score.
        max_candidates: Cap on measured candidates per label.
        min_gain: Stop when the best expansion improves the value by less.
        judge_weight: Weight of the judge's confirmed share in precision (0..1).
        judge_iou: Char IoU at which a judge span confirms an unlabelled match.
        judge_samples: Sample matches shown to ``Judge.reject``.
        match_timeout: Per-text regex timeout in seconds during measurement.
    """

    labels: list[str] | None = None
    min_support: int = 2
    beam_width: int = 8
    max_rules_per_label: int = 25
    pool_size: int = 300
    per_origin: int = 8
    max_candidates: int = 4000
    min_gain: float = 0.001
    judge_weight: float = 0.5
    judge_iou: float = 0.5
    judge_samples: int = 8
    match_timeout: float = 1.0


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


class SearchSynthesizer:
    """LLM-free rule synthesis for span extraction and NER."""

    def __init__(self, config: SearchConfig | None = None, judge: Judge | None = None):
        self.config = config or SearchConfig()
        self.judge = judge or Judge()
        self.report: dict[str, dict] = {}

    def synthesize(
        self,
        dataset: Dataset,
        unlabeled_texts: list[str] | None = None,
        seed_rules: list[Rule] | None = None,
    ) -> list[Rule]:
        """Learn an ordered regex ruleset from the dataset's examples and corrections.

        Args:
            dataset: Training data (NER or EXTRACTION task).
            unlabeled_texts: Optional unlabelled pool. The judge reads it; it is
                also part of candidate dedup, so rules that only differ there
                both stay available.
            seed_rules: Optional rules to add to the candidate pool (hybrid mode).
                Only REGEX rules are used.

        Returns:
            Rules in execution order, with strictly decreasing priorities.
        """
        task = dataset.task
        if task.type not in _SUPPORTED:
            raise ValueError(
                f"search synthesis supports NER and EXTRACTION tasks, not {task.type.value}"
            )
        ner = task.type == TaskType.NER
        executor = RuleExecutor()
        items = dataset.get_all_training_data()
        texts = [executor._select_text(item.input, task.text_field) for item in items]
        docs = [_get_entities(item.expected_output or {}, task.type) for item in items]
        unlabeled = list(unlabeled_texts or [])

        labels: list[str | None]
        if not ner:
            labels = [None]
        elif self.config.labels is not None:
            labels = list(self.config.labels)
        else:
            counts = Counter(e.get("type") for ents in docs for e in ents if e.get("type"))
            labels = sorted(counts, key=lambda t: (-counts[t], t))

        judge_spans = None
        if unlabeled:
            names = [label for label in labels if label is not None] or ["span"]
            judge_spans = self.judge.predict_spans(unlabeled, names)

        self.report = {}
        selected: list[tuple[Candidate, str | None]] = []
        for label in labels:
            t0 = time.time()
            golds = [
                Counter(e["text"].strip() for e in ents if not ner or e.get("type") == label)
                for ents in docs
            ]
            gold_total = sum(sum(g.values()) for g in golds)
            if gold_total == 0:
                continue
            seeds = [
                r
                for r in seed_rules or []
                if r.format == RuleFormat.REGEX
                and (not ner or (r.output_template or {}).get("type") == label)
            ]
            candidates = self._candidates(label, texts, docs, golds, unlabeled, seeds)
            pool = build_pool(
                candidates, self.config.min_support, self.config.per_origin, self.config.pool_size
            )
            n_pool = len(pool)
            pool = [c for c in pool if not self.judge.reject(self._evidence(c, label, unlabeled))]

            confirm = None
            if judge_spans is not None:
                key = label if label is not None else "span"
                spans = [
                    [(s["start"], s["end"]) for s in doc if s.get("label") == key]
                    for doc in judge_spans
                ]
                iou_min = self.config.judge_iou

                def confirm(i, p, spans=spans, iou_min=iou_min):
                    return int(any(_iou((p[0], p[1]), q) >= iou_min for q in spans[i]))

            result = beam_search(
                pool,
                golds,
                gold_total,
                confirm=confirm,
                judge_weight=self.config.judge_weight,
                beam_width=self.config.beam_width,
                max_rules=self.config.max_rules_per_label,
                min_gain=self.config.min_gain,
            )
            by_id = {c.id: c for c in pool}
            selected.extend((by_id[cid], label) for cid in result.rules)
            self.report[str(label)] = {
                "candidates": len(candidates),
                "pool": n_pool,
                "pool_after_judge": len(pool),
                "rules": len(result.rules),
                "value": result.value,
                "train_precision": result.tp / result.n if result.n else 0.0,
                "train_recall": result.tp / gold_total,
                "unlabeled_matches": result.unlabeled_n,
                "unlabeled_confirmed": result.unlabeled_confirmed,
                "seconds": round(time.time() - t0, 1),
            }

        total = len(selected)
        return [
            self._to_rule(c, label, ner, priority=total - k)
            for k, (c, label) in enumerate(selected)
        ]

    # ------------------------------------------------------------------ internals

    def _candidates(
        self,
        label: str | None,
        texts: list[str],
        docs: list[list[dict]],
        golds: list[Counter],
        unlabeled: list[str],
        seeds: list[Rule],
    ) -> list[Candidate]:
        cfg = self.config
        positives = [
            Positive(e["text"].strip(), i, e["start"], e["end"])
            for i, ents in enumerate(docs)
            for e in ents
            if (label is None or e.get("type") == label) and e["text"].strip()
        ]
        measured = measure(
            group_candidates(positives, cfg.min_support),
            texts,
            unlabeled,
            golds,
            cfg.match_timeout,
        )
        ranked = sorted(measured.items(), key=lambda kv: -rank_score(kv[1]))
        cores = [(rx, m["origin"]) for (rx, tpl), m in ranked[:40] if tpl == "$0"]
        extra: dict[CandidateKey, str] = {}
        extra.update(context_candidates(texts, positives, cores[:20]))
        extra.update(mutation_candidates(cores))
        extra.update(
            lookahead_candidates(
                [(k, m["origin"], m["train"]) for k, m in ranked[:200]], texts, golds
            )
        )
        extra = {k: v for k, v in extra.items() if k not in measured}
        measured.update(measure(extra, texts, unlabeled, golds, cfg.match_timeout))

        seed_by_key: dict[CandidateKey, Rule] = {}
        for rule in seeds:
            key = (rule.content, (rule.output_template or {}).get("text", "$0"))
            if key not in measured and key not in seed_by_key:
                seed_by_key[key] = rule
        seed_origin = {k: "seed:" + (r.name or r.id) for k, r in seed_by_key.items()}
        measured.update(measure(seed_origin, texts, unlabeled, golds, cfg.match_timeout))

        kept = dedup(measured, cfg.max_candidates)
        return [
            Candidate(
                id=f"c{i}",
                pattern=rx,
                template=tpl,
                origin=m["origin"],
                tp=m["tp"],
                fp=m["fp"],
                train=m["train"],
                unlabeled=m["unlabeled"],
                seed=seed_by_key.get((rx, tpl)),
            )
            for i, ((rx, tpl), m) in enumerate(sorted(kept.items()))
        ]

    def _evidence(self, c: Candidate, label: str | None, unlabeled: list[str]) -> RuleEvidence:
        matches = [(i, p) for i, row in enumerate(c.unlabeled) for p in row]
        rng = random.Random(42)
        sample = rng.sample(matches, min(self.config.judge_samples, len(matches)))
        shown = []
        for i, (a, b, _t) in sample:
            text = unlabeled[i]
            left = text[max(0, a - 25) : a].replace("\n", " ")
            right = text[b : b + 25].replace("\n", " ")
            shown.append(left + "[[" + text[a:b] + "]]" + right)
        return RuleEvidence(label=label, pattern=c.pattern, n_matches=len(matches), samples=shown)

    def _to_rule(self, c: Candidate, label: str | None, ner: bool, priority: int) -> Rule:
        if c.seed is not None:
            return dataclasses.replace(c.seed, id=str(uuid.uuid4())[:8], priority=priority)
        template = {"text": c.template, "start": "$start", "end": "$end"}
        if ner:
            template["type"] = label
        name = f"{(label or 'span').lower()}_{c.id}"
        return Rule(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=f"search ({c.origin}); train TP {c.tp} / FP {c.fp}",
            format=RuleFormat.REGEX,
            content=c.pattern,
            priority=priority,
            output_template=template,
            output_key="entities" if ner else "spans",
        )
