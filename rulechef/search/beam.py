"""Beam search over ordered rule lists for one label.

The state is an ordered list of pool candidates. Appending a candidate adds
only the matches the executor would keep: a match is dropped if an already
kept match of the label overlaps it with IoU > 0.7 (the executor's
same-type dedup, which also covers exact-position duplicates). The value of
a state is the F1 of its kept matches on the training texts; with a judge,
precision is mixed with the share of its unlabelled matches the judge
confirms.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from rulechef.search.measure import Candidate

_DEDUP_IOU = 0.7


class _Split:
    """Unique predicted items of all pool candidates on one text split."""

    def __init__(
        self,
        pool: list[Candidate],
        field: str,
        golds: list[Counter] | None = None,
        confirm: Callable[[int, tuple[int, int, str]], int] | None = None,
    ):
        items: list[tuple[int, int, int, str]] = []
        index: dict[tuple[int, int, int, str], int] = {}
        self.cand: dict[str, list[int]] = {}
        for c in pool:
            ids = set()
            for i, row in enumerate(getattr(c, field)):
                for a, b, t in row:
                    item = (i, a, b, t)
                    if item not in index:
                        index[item] = len(items)
                        items.append(item)
                    ids.add(index[item])
            self.cand[c.id] = sorted(ids)
        self.n = len(items)

        # Conflict lists: items in the same document with IoU > 0.7 (incl. itself)
        by_doc: dict[int, list[int]] = defaultdict(list)
        for j, item in enumerate(items):
            by_doc[item[0]].append(j)
        self.nbr: list[list[int]] = [[] for _ in range(self.n)]
        for js in by_doc.values():
            js = sorted(js, key=lambda j: items[j][1])
            for p, j in enumerate(js):
                a, b = items[j][1], items[j][2]
                self.nbr[j].append(j)
                for q in js[p + 1 :]:
                    a2, b2 = items[q][1], items[q][2]
                    if a2 >= b:  # sorted by start: no later item overlaps j
                        break
                    inter = max(0, min(b, b2) - max(a, a2))
                    union = (b - a) + (b2 - a2) - inter
                    if inter / max(union, 1) > _DEDUP_IOU:
                        self.nbr[j].append(q)
                        self.nbr[q].append(j)

        if golds is not None:
            keys: dict[tuple[int, str], int] = {}
            key_ids = []
            for i, _a, _b, t in items:
                kk = (i, t.strip())
                if kk not in keys:
                    keys[kk] = len(keys)
                key_ids.append(keys[kk])
            self.key = key_ids
            self.gold = [golds[i].get(t, 0) for (i, t) in keys]
            self.nkeys = len(keys)
        if confirm is not None:
            self.conf = [confirm(i, (a, b, t)) for (i, a, b, t) in items]


@dataclass
class _State:
    blocked: set[int]
    counts: dict[int, int]  # kept predictions per gold key
    tp: int = 0
    n: int = 0
    conf: int = 0


def _delta(sp: _Split, st: _State, cid: str, scored: bool, confirmed: bool) -> tuple:
    new = [j for j in sp.cand[cid] if j not in st.blocked]
    dtp = dconf = 0
    if new and scored:
        for k, cnt in Counter(sp.key[j] for j in new).items():
            dtp += min(cnt, max(0, sp.gold[k] - st.counts.get(k, 0)))
    if new and confirmed:
        dconf = sum(sp.conf[j] for j in new)
    return dtp, len(new), dconf, new


def _apply(sp: _Split, st: _State, d: tuple, scored: bool) -> _State:
    dtp, dn, dconf, new = d
    blocked = set(st.blocked)
    for j in new:
        blocked.update(sp.nbr[j])
    counts = dict(st.counts)
    if scored:
        for j in new:
            counts[sp.key[j]] = counts.get(sp.key[j], 0) + 1
    return _State(blocked, counts, st.tp + dtp, st.n + dn, st.conf + dconf)


@dataclass
class BeamResult:
    """Selected candidate ids in rule order, with their train and judge totals."""

    rules: tuple[str, ...]
    value: float
    tp: int
    n: int
    unlabeled_n: int
    unlabeled_confirmed: int


def beam_search(
    pool: list[Candidate],
    golds: list[Counter],
    gold_total: int,
    confirm: Callable[[int, tuple[int, int, str]], int] | None = None,
    judge_weight: float = 0.5,
    beam_width: int = 8,
    max_rules: int = 25,
    min_gain: float = 0.001,
) -> BeamResult:
    """Pick an ordered rule list from ``pool`` that maximises train F1.

    With ``confirm`` (a judge's verdict on one unlabelled match), precision
    becomes ``(1 - w) * P_train + w * P_judge``, where ``P_judge`` is the
    confirmed share of the state's unlabelled matches. Ties go to the
    shorter total pattern length, then to candidate ids. Search stops when
    the best expansion gains less than ``min_gain``.
    """
    judged = confirm is not None
    train = _Split(pool, "train", golds=golds)
    unl = _Split(pool, "unlabeled", confirm=confirm) if judged else None
    length = {c.id: len(c.pattern) for c in pool}

    def value(tp: int, n: int, conf: int, un: int) -> float:
        p_train = tp / n if n else 0.0
        recall = tp / gold_total if gold_total else 0.0
        if judged:
            p_judge = conf / un if un else p_train
            precision = (1 - judge_weight) * p_train + judge_weight * p_judge
        else:
            precision = p_train
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    empty = _State(set(), {})
    beams: list[tuple] = [((), empty, _State(set(), {}) if judged else None, 0.0, 0)]
    best = beams[0]
    for _ in range(max_rules):
        expansions: dict[tuple, tuple] = {}
        for bi, (rules, st, su, _v, total_len) in enumerate(beams):
            used = set(rules)
            for c in pool:
                if c.id in used:
                    continue
                ds = _delta(train, st, c.id, True, False)
                if ds[1] == 0:
                    continue
                du = _delta(unl, su, c.id, False, True) if judged else (0, 0, 0, [])
                v2 = value(
                    st.tp + ds[0],
                    st.n + ds[1],
                    (su.conf if su else 0) + du[2],
                    (su.n if su else 0) + du[1],
                )
                key = tuple(sorted(rules + (c.id,)))
                len2 = total_len + length[c.id]
                if key not in expansions or (v2, -len2) > (
                    expansions[key][3],
                    -expansions[key][4],
                ):
                    expansions[key] = (rules + (c.id,), bi, (ds, du), v2, len2)
        if not expansions:
            break
        top = sorted(expansions.values(), key=lambda e: (-e[3], e[4], e[0]))[:beam_width]
        if top[0][3] <= best[3] + min_gain:
            break
        next_beams = []
        for rules2, bi, (ds, du), v2, len2 in top:
            st0, su0 = beams[bi][1], beams[bi][2]
            st2 = _apply(train, st0, ds, True)
            su2 = _apply(unl, su0, du, False) if judged else None
            next_beams.append((rules2, st2, su2, v2, len2))
        beams = next_beams
        best = beams[0]
    rules, st, su, v, _ = best
    return BeamResult(
        rules=rules,
        value=v,
        tp=st.tp,
        n=st.n,
        unlabeled_n=su.n if su else 0,
        unlabeled_confirmed=su.conf if su else 0,
    )
