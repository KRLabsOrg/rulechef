"""Tests for rulechef.search — LLM-free synthesis. No model and no API calls."""

from collections import Counter

import pytest

from rulechef.core import Dataset, Example, Rule, RuleFormat, Task, TaskType
from rulechef.executor import RuleExecutor
from rulechef.search import Judge, RuleEvidence, SearchConfig, SearchSynthesizer, SpanModelJudge
from rulechef.search.beam import beam_search
from rulechef.search.candidates import (
    Positive,
    atomise,
    build_group_regex,
    context_candidates,
    group_candidates,
    signature,
)
from rulechef.search.measure import Candidate, build_pool, count_hits, run_pattern

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MONTHS = ["January", "March", "May", "June", "October", "December"]


def _ner_task():
    return Task(
        name="ids",
        description="dates and application numbers",
        input_schema={"text": "str"},
        output_schema={"entities": "List[Entity]"},
        type=TaskType.NER,
        text_field="text",
    )


def _doc(i):
    day, month, year = 1 + i % 27, MONTHS[i % len(MONTHS)], 1990 + i % 9
    code = f"{10000 + 37 * i}/{90 + i % 9}"
    date = f"{day} {month} {year}"
    text = f"The applicant lodged application no. {code} on {date}, page {i + 3}."
    ents = [
        {
            "text": code,
            "start": text.index(code),
            "end": text.index(code) + len(code),
            "type": "CODE",
        },
        {
            "text": date,
            "start": text.index(date),
            "end": text.index(date) + len(date),
            "type": "DATE",
        },
    ]
    return text, ents


def _dataset(n=12):
    ds = Dataset(name="t", task=_ner_task())
    for i in range(n):
        text, ents = _doc(i)
        ds.examples.append(
            Example(
                id=f"e{i}", input={"text": text}, expected_output={"entities": ents}, source="t"
            )
        )
    return ds


def _two_shape_dataset():
    ds = Dataset(name="t2", task=_ner_task())
    codes = ["12345/96", "AB-1234", "23456/97", "CD-5678", "34567/98", "EF-9012"]
    for i, code in enumerate(codes):
        text = f"See no. {code} on file."
        start = text.index(code)
        ent = {"text": code, "start": start, "end": start + len(code), "type": "CODE"}
        ds.examples.append(
            Example(
                id=f"t{i}", input={"text": text}, expected_output={"entities": [ent]}, source="t"
            )
        )
    return ds


_SEED = Rule(
    id="s1",
    name="llm_code",
    description="written elsewhere",
    format=RuleFormat.REGEX,
    content=r"no\. (\S+) on",
    output_template={"text": "$1", "start": "$start", "end": "$end", "type": "CODE"},
    output_key="entities",
)


def _extract(rules, text):
    out = RuleExecutor().apply_rules(rules, {"text": text}, TaskType.NER, "text")
    return sorted((e["type"], e["text"]) for e in out.get("entities", []))


def _cand(cid, pattern, train, tp, fp, unlabeled=None, origin=None):
    return Candidate(
        id=cid,
        pattern=pattern,
        template="$0",
        origin=origin or cid,
        tp=tp,
        fp=fp,
        train=train,
        unlabeled=unlabeled or [[] for _ in train],
    )


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


class TestCandidates:
    def test_atomise_kinds(self):
        atoms = atomise("12 March 1996")
        assert [k for k, _ in atoms] == ["N", "S", "WC", "S", "N"]
        assert signature(atomise("12345/96")) == ("N", "P/", "N")

    def test_group_regex_levels(self):
        members = [atomise("12 March 1996"), atomise("3 June 2001")]
        lit = build_group_regex(members, ("range", "lit", "ws"), {}, True)
        assert lit == r"\b\d{1,2}\s+(?:March|June)\s+\d{4}\b"
        cls = build_group_regex(members, ("gen", "class", "sp"), {}, False)
        assert cls == r"\d+(?:[.,]\d+)* [A-Z][a-z]+ \d+(?:[.,]\d+)*"

    def test_min_support_filters_groups(self):
        pos = [
            Positive("12345/96", 0, 0, 8),
            Positive("23456/97", 1, 0, 8),
            Positive("ab-1", 2, 0, 4),
        ]
        out = group_candidates(pos, min_support=2)
        assert (r"\b\d{5}/\d{2}\b", "$0") in out
        assert not any("ab" in rx for rx, _ in out)

    def test_context_candidate_uses_capture_group(self):
        texts = ["see no. 123", "see no. 456"]
        pos = [Positive("123", 0, 8, 11), Positive("456", 1, 8, 11)]
        out = context_candidates(texts, pos, [(r"\b\d{3}\b", "g0")])
        (rx, tpl), origin = next(iter(out.items()))
        assert tpl == "$1" and origin == "cg0"
        assert run_pattern(rx, tpl, ["filed under no. 789"])[0][0][2] == "789"


# ---------------------------------------------------------------------------
# Measurement and pool
# ---------------------------------------------------------------------------


class TestMeasure:
    def test_capture_template_offsets_follow_executor(self):
        rows = run_pattern(r"no\.\s*(\d+)", "$1", ["x no. 42 y"])
        assert rows == [[(6, 8, "42")]]

    def test_bad_pattern_is_skipped(self):
        assert run_pattern("(unclosed", "$0", ["text"]) is None

    def test_count_hits_is_a_multiset(self):
        rows = [[(0, 2, "12"), (5, 7, "12"), (9, 11, "34")]]
        assert count_hits(rows, [Counter({"12": 1, "34": 1})]) == (2, 1)

    def test_pool_reserves_slots_per_origin(self):
        big = [_cand(f"a{i}", "x" * (i + 1), [[]], tp=100, fp=0, origin="g0") for i in range(10)]
        rare = [_cand("r", "y", [[]], tp=3, fp=0, origin="g1")]
        pool = build_pool(big + rare, min_support=2, per_origin=2, size=3)
        assert "r" in {c.id for c in pool}


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


class TestBeam:
    def test_prefers_precise_rule(self):
        golds = [Counter({"2001": 1}), Counter({"1999": 1})]
        precise = _cand("p", r"\d{4}", [[(0, 4, "2001")], [(0, 4, "1999")]], 2, 0)
        broad = _cand(
            "b",
            r"\d+",
            [[(0, 4, "2001"), (5, 6, "7")], [(0, 4, "1999"), (5, 6, "8")]],
            2,
            2,
        )
        res = beam_search([precise, broad], golds, gold_total=2)
        assert res.rules == ("p",)
        assert res.value == pytest.approx(1.0)

    def test_overlapping_matches_are_not_counted_twice(self):
        golds = [Counter({"12 May 2001": 1})]
        full = _cand("f", "full", [[(0, 11, "12 May 2001")]], 1, 0)
        part = _cand("q", "part", [[(0, 10, "12 May 200")]], 0, 1)
        res = beam_search([full, part], golds, gold_total=1)
        assert res.rules == ("f",)  # the part is IoU-deduped, so adding it gains nothing

    def test_judge_confirmation_penalises_unconfirmed_rule(self):
        # Both rules are perfect on train; only one is confirmed on unlabelled text.
        golds = [Counter({"A1": 1, "B1": 1})]
        good = _cand("g", "A", [[(0, 2, "A1")]], 1, 0, unlabeled=[[(0, 2, "A2")]])
        bad = _cand("x", "B", [[(3, 5, "B1")]], 1, 0, unlabeled=[[(3, 5, "B2")]])

        def confirm(i, p):
            return int(p[0] == 0)

        no_judge = beam_search([good, bad], golds, gold_total=2)
        judged = beam_search([good, bad], golds, gold_total=2, confirm=confirm, judge_weight=1.0)
        assert set(no_judge.rules) == {"g", "x"}
        assert judged.rules == ("g",)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class TestSynthesizer:
    def test_learns_rules_that_generalise(self):
        rules = SearchSynthesizer().synthesize(_dataset())
        assert rules and all(r.format == RuleFormat.REGEX for r in rules)
        assert [r.priority for r in rules] == sorted((r.priority for r in rules), reverse=True)
        text = "application no. 55555/93 was lodged on 9 March 1994."
        assert _extract(rules, text) == [("CODE", "55555/93"), ("DATE", "9 March 1994")]

    def test_deterministic(self):
        a = [r.content for r in SearchSynthesizer().synthesize(_dataset())]
        b = [r.content for r in SearchSynthesizer().synthesize(_dataset())]
        assert a == b

    def test_label_subset_and_order(self):
        rules = SearchSynthesizer(SearchConfig(labels=["DATE"])).synthesize(_dataset())
        assert {r.output_template["type"] for r in rules} == {"DATE"}

    def test_rejects_unsupported_task(self):
        task = Task("c", "c", {"text": "str"}, {"label": "str"}, type=TaskType.CLASSIFICATION)
        with pytest.raises(ValueError):
            SearchSynthesizer().synthesize(Dataset(name="c", task=task))

    def test_seed_rules_join_the_pool(self):
        # Two code shapes: one generated rule covers one shape, the seed covers both.
        cfg = SearchConfig(labels=["CODE"], max_rules_per_label=1)
        rules = SearchSynthesizer(cfg).synthesize(_two_shape_dataset(), seed_rules=[_SEED])
        assert [r.content for r in rules] == [_SEED.content]
        assert rules[0].name == "llm_code" and rules[0].id != "s1"
        assert rules[0].output_template["text"] == "$1"

    def test_reject_hook_sees_evidence_and_filters(self):
        seen = []

        class NoYearsJudge(Judge):
            def reject(self, evidence: RuleEvidence) -> bool:
                seen.append(evidence)
                return "\\d{4}" in evidence.pattern

        unlabeled = ["Judgment of 4 May 1999 in case no. 30000/95."]
        cfg = SearchConfig(labels=["DATE"])
        rules = SearchSynthesizer(cfg, NoYearsJudge()).synthesize(_dataset(), unlabeled)
        assert all("\\d{4}" not in r.content for r in rules)
        assert seen and all("[[" in s for e in seen for s in e.samples)

    def test_span_model_judge_maps_label_names(self):
        calls = []

        def fake_model(text, names):
            calls.append(names)
            i = text.find("1999")
            return [
                {"start": i, "end": i + 4, "label": "date"},
                {"start": 0, "end": 1, "label": "x"},
            ]

        judge = SpanModelJudge(fake_model, label_names={"DATE": "date"})
        out = judge.predict_spans(["in 1999"], ["DATE", "CODE"])
        assert calls == [["date", "CODE"]]
        assert out == [[{"start": 3, "end": 7, "label": "DATE"}]]

    def test_default_judge_is_noop(self):
        judge = Judge()
        assert judge.predict_spans(["x"], ["A"]) is None
        assert judge.reject(RuleEvidence("A", "x", 0, [])) is False
