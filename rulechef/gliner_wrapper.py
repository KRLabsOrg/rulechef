"""GLiNER / GLiNER2 observer for learning rules from entity predictions.

Monkey-patches GLiNER's predict_entities() or GLiNER2's extract_entities() /
classify_text() / extract_json() methods and feeds results into the buffer.

No LLM calls needed — GLiNER output is already structured.
"""

from __future__ import annotations

from typing import Any

from rulechef.buffer import ExampleBuffer
from rulechef.core import Task

# GLiNER2 method → task kind mapping
_GLINER2_METHODS = {
    "extract_entities": "ner",
    "classify_text": "classification",
    "extract_json": "transformation",
}


class GlinerObserver:
    """Observes GLiNER/GLiNER2 prediction calls and feeds results to buffer."""

    def __init__(
        self,
        buffer: ExampleBuffer,
        task: Task | None = None,
        method: str | None = None,
    ):
        self._buffer = buffer
        self.task = task
        self._original_fn: Any = None
        self._method_name: str | None = None
        self._task_kind: str | None = None  # "ner", "classification", "transformation"
        self._client: Any = None
        self._skip = False
        self._observed_count = 0
        self._observed_labels: set[str] = set()
        self._requested_method = method

    def attach(self, gliner_model: Any) -> None:
        """Monkey-patch the model's prediction method.

        GLiNER (auto-detected):  patches predict_entities → NER
        GLiNER2 (method param):  patches the specified method → NER/CLASSIFICATION/TRANSFORMATION
        """
        if self._requested_method:
            if not hasattr(gliner_model, self._requested_method):
                raise TypeError(f"Model has no method '{self._requested_method}'")
            self._method_name = self._requested_method
            self._task_kind = _GLINER2_METHODS.get(self._requested_method, "ner")
        elif hasattr(gliner_model, "predict_entities"):
            self._method_name = "predict_entities"
            self._task_kind = "ner"
        elif hasattr(gliner_model, "extract_entities"):
            self._method_name = "extract_entities"
            self._task_kind = "ner"
        else:
            raise TypeError(
                "Model has neither predict_entities() nor extract_entities(). "
                "Expected a GLiNER or GLiNER2 model."
            )

        self._original_fn = getattr(gliner_model, self._method_name)
        self._client = gliner_model
        observer = self

        def observed_fn(*args, **kwargs):
            result = observer._original_fn(*args, **kwargs)
            if not observer._skip:
                observer._capture(args, kwargs, result)
            return result

        setattr(gliner_model, self._method_name, observed_fn)

    def detach(self) -> None:
        """Restore the original method on the model."""
        if self._client and self._original_fn and self._method_name:
            setattr(self._client, self._method_name, self._original_fn)
            self._client = None

    def _capture(self, args: tuple, kwargs: dict, result: Any) -> None:
        """Route to the appropriate normalizer based on task kind."""
        # First positional arg is always text
        text = args[0] if args else kwargs.get("text", "")
        # Second positional arg is labels/schema/entity_types/tasks/structures
        if len(args) > 1:
            schema_arg = args[1]
        else:
            schema_arg = (
                kwargs.get("labels")
                or kwargs.get("entity_types")
                or kwargs.get("tasks")
                or kwargs.get("structures")
                or kwargs.get("schema")
                or []
            )

        input_data = {"text": text}

        if self._task_kind == "ner":
            output_data = self._normalize_ner(schema_arg, result)
        elif self._task_kind == "classification":
            output_data = self._normalize_classification(result)
        elif self._task_kind == "transformation":
            output_data = self._normalize_transformation(result)
        else:
            return

        self._buffer.add_llm_observation(
            input_data,
            output_data,
            metadata={
                "source": "gliner",
                "method": self._method_name,
                "task_kind": self._task_kind,
            },
        )
        self._observed_count += 1

    def _normalize_ner(self, labels: Any, result: Any) -> dict:
        """Normalize NER output from GLiNER or GLiNER2."""
        if isinstance(labels, list):
            self._observed_labels.update(labels)
        elif isinstance(labels, dict):
            self._observed_labels.update(labels.keys())

        if isinstance(result, list):
            # GLiNER: flat list [{"text", "label", "start", "end", "score"}]
            entities = [
                {
                    "text": e["text"],
                    "start": e["start"],
                    "end": e["end"],
                    "type": e["label"],
                }
                for e in result
            ]
        else:
            # GLiNER2: {"entities": {"type": [{"text", ...}]}} or {"entities": {"type": ["text"]}}
            entities = []
            for entity_type, items in result.get("entities", {}).items():
                for e in items:
                    ent: dict[str, Any] = {
                        "text": e["text"] if isinstance(e, dict) else e,
                        "type": entity_type,
                    }
                    if isinstance(e, dict):
                        if "start" in e:
                            ent["start"] = e["start"]
                        if "end" in e:
                            ent["end"] = e["end"]
                    entities.append(ent)

        return {"entities": entities}

    def _normalize_classification(self, result: dict) -> dict:
        """Normalize classification output from GLiNER2."""
        for value in result.values():
            if isinstance(value, str):
                label = value
            elif isinstance(value, dict) and "label" in value:
                label = value["label"]
            elif isinstance(value, list):
                label = value[0] if value else ""
            else:
                label = str(value)
            self._observed_labels.add(label)
            return {"label": label}
        return {"label": ""}

    def _normalize_transformation(self, result: dict) -> dict:
        """Normalize structured extraction output from GLiNER2.

        GLiNER2 returns e.g. {"structure_name": [{"field": "value", ...}]}.
        Pass through as-is — already structured.
        """
        return result

    def build_task(self, name: str | None = None) -> Task:
        """Auto-create a Task from observed data."""
        from rulechef.core import TaskType

        if not self._observed_count:
            raise RuntimeError("No observations yet — cannot build task.")

        if self._task_kind == "ner":
            return Task(
                name=name or "gliner_ner",
                description=f"NER ({', '.join(sorted(self._observed_labels))})",
                input_schema={"text": "str"},
                output_schema={
                    "entities": [{"text": "str", "start": "int", "end": "int", "type": "str"}]
                },
                type=TaskType.NER,
                text_field="text",
            )
        elif self._task_kind == "classification":
            return Task(
                name=name or "gliner_classify",
                description=f"Classification ({', '.join(sorted(self._observed_labels))})",
                input_schema={"text": "str"},
                output_schema={"label": "str"},
                type=TaskType.CLASSIFICATION,
                text_field="text",
            )
        else:  # transformation
            return Task(
                name=name or "gliner_extract",
                description="Structured extraction from GLiNER2",
                input_schema={"text": "str"},
                output_schema={},
                type=TaskType.TRANSFORMATION,
                text_field="text",
            )

    def get_stats(self) -> dict:
        """Return observer statistics."""
        return {
            "observed": self._observed_count,
            "task_kind": self._task_kind,
            "method": self._method_name,
            "labels_seen": sorted(self._observed_labels),
        }
