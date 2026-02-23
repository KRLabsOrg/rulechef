"""Output matching - compares rule outputs to expected outputs"""

import json
from collections.abc import Callable
from typing import Any

from rulechef.core import DEFAULT_OUTPUT_KEYS, TaskType

# Type alias for matcher functions
OutputMatcher = Callable[[dict[str, Any], dict[str, Any]], bool]


def outputs_match(
    expected: dict,
    actual: dict,
    task_type: TaskType = TaskType.EXTRACTION,
    custom_matcher: OutputMatcher | None = None,
    matching_mode: str = "text",
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
                return span.text
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
                        span.text,
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

        entities1 = get_entities(expected)
        entities2 = get_entities(actual)

        if len(entities1) != len(entities2):
            return False

        def entity_key(e):
            if isinstance(e, dict):
                # Support both "type" and "label" for entity type field
                ent_type = e.get("type") or e.get("label", "")
                return (e.get("text", ""), ent_type)
            if hasattr(e, "text"):
                ent_type = getattr(e, "type", None) or getattr(e, "label", "")
                return (e.text, ent_type)
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
                except Exception:
                    if val1 != val2:
                        return False
            elif val1 != val2:
                return False
        return True

    else:
        # Other: Exact match
        return expected == actual
