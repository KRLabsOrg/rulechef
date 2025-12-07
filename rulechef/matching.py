"""Output matching - compares rule outputs to expected outputs"""

import json
from typing import Dict, Optional, Callable, Any

from rulechef.core import TaskType

# Type alias for matcher functions
OutputMatcher = Callable[[Dict[str, Any], Dict[str, Any]], bool]


def outputs_match(
    expected: Dict,
    actual: Dict,
    task_type: TaskType = TaskType.EXTRACTION,
    custom_matcher: Optional[OutputMatcher] = None,
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

        # Check text match (order independent)
        texts1 = sorted([s["text"] if isinstance(s, dict) else s.text for s in spans1])
        texts2 = sorted([s["text"] if isinstance(s, dict) else s.text for s in spans2])

        return texts1 == texts2

    elif task_type == TaskType.NER:
        # NER: Compare entities including type
        for key in ["entities", "spans", "ner"]:
            if key in expected or key in actual:
                entities1 = expected.get(key, [])
                entities2 = actual.get(key, [])
                break
        else:
            return expected == actual

        if len(entities1) != len(entities2):
            return False

        def entity_key(e):
            if isinstance(e, dict):
                return (e.get("text", ""), e.get("type", ""))
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
                except:
                    if val1 != val2:
                        return False
            elif val1 != val2:
                return False
        return True

    else:
        # Other: Exact match
        return expected == actual
