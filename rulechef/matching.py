"""Output matching - compares rule outputs to expected outputs"""

import json
from typing import Dict

from rulechef.core import TaskType


def outputs_match(
    output1: Dict, output2: Dict, task_type: TaskType = TaskType.EXTRACTION
) -> bool:
    """Check if two outputs match based on task type"""

    if task_type == TaskType.EXTRACTION:
        spans1 = output1.get("spans", [])
        spans2 = output2.get("spans", [])

        if len(spans1) != len(spans2):
            return False

        # Check text match (order independent)
        texts1 = sorted([s["text"] if isinstance(s, dict) else s.text for s in spans1])
        texts2 = sorted([s["text"] if isinstance(s, dict) else s.text for s in spans2])

        return texts1 == texts2

    elif task_type == TaskType.NER:
        # NER: Compare entities including type
        for key in ["entities", "spans", "ner"]:
            if key in output1 or key in output2:
                entities1 = output1.get(key, [])
                entities2 = output2.get(key, [])
                break
        else:
            return output1 == output2

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
        label1 = str(output1.get("label", "")).lower().strip()
        label2 = str(output2.get("label", "")).lower().strip()
        return label1 == label2

    elif task_type == TaskType.TRANSFORMATION:
        # Transformation: Compare all array keys with relaxed matching
        for key in set(output1.keys()) | set(output2.keys()):
            val1 = output1.get(key)
            val2 = output2.get(key)

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
        return output1 == output2
