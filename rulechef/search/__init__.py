"""LLM-free rule synthesis: candidate generation, exact measurement, beam search.

Usage::

    from rulechef import RuleChef
    from rulechef.search import SearchConfig, SpanModelJudge

    chef = RuleChef(task, synth="search")  # no LLM client needed
    chef.learn_rules(holdout_fraction=0.2)

    # optional: a span model confirms rule matches on unlabelled text
    judge = SpanModelJudge(my_span_fn, label_names={"DATETIME": "date or time"})
    chef = RuleChef(task, synth="search", judge=judge)
    chef.learn_rules(unlabeled_texts=pool)
"""

from rulechef.search.judge import Judge, RuleEvidence, SpanModelJudge
from rulechef.search.synthesizer import SearchConfig, SearchSynthesizer

__all__ = ["Judge", "RuleEvidence", "SearchConfig", "SearchSynthesizer", "SpanModelJudge"]
