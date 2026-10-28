"""Deterministic measurement of candidate rules and pool selection.

Each candidate runs alone over the training texts (and the unlabelled pool,
if any) with the executor's template semantics. Train matches are scored
against gold with whitespace-insensitive text matching, as a multiset per
document, the same way the extraction benchmarks score rulesets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import regex

from rulechef.core import Rule
from rulechef.executor import substitute_template
from rulechef.search.candidates import CandidateKey, Row


@dataclass
class Candidate:
    """A measured candidate rule for one label."""

    id: str
    pattern: str
    template: str  # "$0" for the whole match, "$k" for a capture group
    origin: str
    tp: int
    fp: int
    train: list[Row]
    unlabeled: list[Row]
    seed: Rule | None = field(default=None, repr=False)  # set for rules supplied by the caller

    @property
    def score(self) -> float:
        """Precision-weighted hits, TP * precision."""
        return self.tp**2 / (self.tp + self.fp)


def run_pattern(
    pattern: str, template: str, texts: list[str], timeout: float = 1.0
) -> list[Row] | None:
    """Matches of one rule per text, or None if it fails to compile or times out."""
    try:
        compiled = regex.compile(pattern)
    except Exception:
        return None
    tpl = {"text": template, "start": "$start", "end": "$end"}
    rows = []
    for text in texts:
        try:
            matches = list(compiled.finditer(text, timeout=timeout))
        except Exception:
            return None
        row = []
        for m in matches:
            if m.end() == m.start():
                continue
            out = substitute_template(tpl, m.group(), m.start(), m.end(), m.groups())
            row.append((out["start"], out["end"], out["text"]))
        rows.append(row)
    return rows


def count_hits(rows: list[Row], golds: list[Counter]) -> tuple[int, int]:
    """(TP, FP) of a rule alone, multiset text match per document."""
    tp = fp = 0
    for row, gold in zip(rows, golds):
        if not row:
            continue
        for text, n in Counter(x[2].strip() for x in row).items():
            hit = min(n, gold.get(text, 0))
            tp += hit
            fp += n - hit
    return tp, fp


def measure(
    candidates: dict[CandidateKey, str],
    texts: list[str],
    unlabeled: list[str],
    golds: list[Counter],
    timeout: float = 1.0,
) -> dict[CandidateKey, dict]:
    """Run every candidate; keep those with at least one true positive."""
    out = {}
    for (pattern, template), origin in candidates.items():
        rows = run_pattern(pattern, template, texts, timeout)
        if rows is None:
            continue
        tp, fp = count_hits(rows, golds)
        if tp == 0:
            continue
        u_rows = run_pattern(pattern, template, unlabeled, timeout)
        if u_rows is None:
            continue
        out[(pattern, template)] = {
            "origin": origin,
            "train": rows,
            "unlabeled": u_rows,
            "tp": tp,
            "fp": fp,
        }
    return out


def rank_score(m: dict) -> float:
    return m["tp"] ** 2 / (m["tp"] + m["fp"])


def dedup(measured: dict[CandidateKey, dict], cap: int) -> dict[CandidateKey, dict]:
    """Keep one candidate per identical (train, unlabelled) match set, the shortest.

    Matching on train alone would merge the alternatives a judge has to tell
    apart (observed months vs a wider month class), so the unlabelled matches
    are part of the key. At most ``cap`` candidates survive, by rank score.
    """
    seen: dict[tuple, CandidateKey] = {}
    for key, m in sorted(measured.items(), key=lambda kv: (len(kv[0][0]), kv[0])):
        match_set = (
            tuple(tuple(r) for r in m["train"]),
            tuple(tuple(r) for r in m["unlabeled"]),
        )
        if match_set not in seen:
            seen[match_set] = key
    kept = {k: measured[k] for k in seen.values()}
    if len(kept) > cap:
        best = sorted(kept, key=lambda k: -rank_score(kept[k]))[:cap]
        kept = {k: kept[k] for k in best}
    return kept


def build_pool(
    candidates: list[Candidate], min_support: int, per_origin: int = 8, size: int = 300
) -> list[Candidate]:
    """Search pool: the best ``per_origin`` of each origin group, then fill by score.

    Ranking only by score would fill the pool with near-identical variants
    of the dominant pattern and push out the rare-pattern rules.
    """

    def order(c: Candidate) -> tuple:
        return (-c.score, len(c.pattern), c.id)

    eligible = [c for c in candidates if c.tp >= min_support]
    by_origin: dict[str, list[Candidate]] = defaultdict(list)
    for c in eligible:
        by_origin[c.origin].append(c)
    pool = []
    for group in by_origin.values():
        pool.extend(sorted(group, key=order)[:per_origin])
    pool.sort(key=order)
    pool = pool[:size]
    ids = {c.id for c in pool}
    for c in sorted(eligible, key=order):
        if len(pool) >= size:
            break
        if c.id not in ids:
            pool.append(c)
            ids.add(c.id)
    return pool
