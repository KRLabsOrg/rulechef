import pytest

from rulechef.core import Rule, RuleFormat, TaskType
from rulechef.executor import RuleExecutor


def _rule(rule_id, pattern, type, priority=0):
    return Rule(
        id=rule_id,
        name=rule_id,
        description="",
        format=RuleFormat.REGEX,
        content=pattern,
        priority=priority,
        output_template={"text": "$0", "start": "$start", "end": "$end", "type": type},
        output_key="entities",
    )


def _texts(rules, **kwargs):
    out = RuleExecutor(**kwargs).apply_rules(rules, {"text": TEXT}, TaskType.NER, "text")
    return sorted(e["text"] for e in out.get("entities", []))


ORG = _rule("org", r"Max Müller Rechtsanwalt GmbH", type="ORG", priority=1)
ORG_SHORT = _rule("org_short", r"Max Müller", type="ORG", priority=9)
PER = _rule("per", r"Max Müller", type="PER", priority=9)
TEXT = "Kanzlei Max Müller Rechtsanwalt GmbH"


@pytest.mark.parametrize(
    "rules, kwargs, expected",
    [
        # default: partial overlap (IoU < 0.7) -> both kept
        ([ORG, ORG_SHORT], {}, ["Max Müller", "Max Müller Rechtsanwalt GmbH"]),
        # longest wins, even against higher priority
        ([ORG, ORG_SHORT], {"overlap_strategy": "longest"}, ["Max Müller Rechtsanwalt GmbH"]),
        # priority: higher-ranked rule wins
        ([ORG, ORG_SHORT], {"overlap_strategy": "priority"}, ["Max Müller"]),
        # different types don't conflict by default
        (
            [ORG, PER],
            {"overlap_strategy": "longest"},
            ["Max Müller", "Max Müller Rechtsanwalt GmbH"],
        ),
        # unless cross_type_overlaps=True
        (
            [ORG, PER],
            {"overlap_strategy": "longest", "cross_type_overlaps": True},
            ["Max Müller Rechtsanwalt GmbH"],
        ),
    ],
)
def test_overlap_resolution(rules, kwargs, expected):
    assert _texts(rules, **kwargs) == expected


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        RuleExecutor(overlap_strategy="shortest")
