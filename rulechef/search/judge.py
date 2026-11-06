"""Judge protocol for search synthesis.

A judge looks at unlabelled text and at candidate rules; it never sees the
evaluation data. Two optional hooks:

- ``predict_spans``: span predictions on the unlabelled pool. The search
  counts a candidate's unlabelled match as confirmed when a predicted span
  of the same label overlaps it, and mixes the confirmed share into
  precision. Any span extractor fits (a GLiNER-style model, an encoder with
  an NER head, an LLM if you want one).
- ``reject``: a veto on one pool candidate, given sample matches in context
  (for example a classifier that calls a rule "too broad").

The base class answers neither, so ``Judge()`` is the no-op default and the
search then runs on labelled data alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class RuleEvidence:
    """What a judge sees about one candidate rule."""

    label: str | None
    pattern: str
    n_matches: int  # matches on the unlabelled pool
    samples: list[str]  # sample matches as "left context[[match]]right context"


class Judge:
    """No-op judge. Subclass and override either hook."""

    def predict_spans(self, texts: list[str], labels: list[str]) -> list[list[dict]] | None:
        """Per text, predicted spans as dicts with ``start``, ``end`` and ``label``.

        Return None to give no opinion (the default).
        """
        return None

    def reject(self, evidence: RuleEvidence) -> bool:
        """True drops the candidate from the search pool. Default keeps everything."""
        return False


class SpanModelJudge(Judge):
    """Wrap any span model as a judge.

    Args:
        predict: ``predict(text, label_names) -> [{"start", "end", "label"}, ...]``.
            ``label`` must be one of the given names.
        label_names: Optional mapping from task labels to the names the model
            is asked for (``{"DATETIME": "date or time"}``). Labels without an
            entry are passed through unchanged.
    """

    def __init__(
        self,
        predict: Callable[[str, list[str]], list[dict]],
        label_names: dict[str, str] | None = None,
    ):
        self._predict = predict
        self.label_names = label_names or {}

    def predict_spans(self, texts: list[str], labels: list[str]) -> list[list[dict]]:
        names = [self.label_names.get(label, label) for label in labels]
        to_label = dict(zip(names, labels))
        out = []
        for text in texts:
            spans = []
            for s in self._predict(text, names) or []:
                label = to_label.get(s.get("label"))
                if label is not None:
                    spans.append({"start": s["start"], "end": s["end"], "label": label})
            out.append(spans)
        return out
