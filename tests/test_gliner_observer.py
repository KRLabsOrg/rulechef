"""Tests for GLiNER/GLiNER2 observer integration."""

from unittest.mock import MagicMock

import pytest

from rulechef.buffer import ExampleBuffer
from rulechef.core import Task, TaskType
from rulechef.engine import RuleChef
from rulechef.gliner_wrapper import GlinerObserver

# =========================================================================
# Helpers
# =========================================================================


def _make_gliner_model():
    """Mock GLiNER model with predict_entities."""
    model = MagicMock()
    model.predict_entities = MagicMock(
        return_value=[
            {"text": "Apple", "label": "company", "start": 0, "end": 5, "score": 0.95},
            {"text": "Steve Jobs", "label": "person", "start": 22, "end": 32, "score": 0.92},
        ]
    )
    return model


def _make_gliner2_model():
    """Mock GLiNER2 model with extract_entities, classify_text, extract_json."""
    model = MagicMock()

    model.extract_entities = MagicMock(
        return_value={
            "entities": {
                "company": [{"text": "Apple", "start": 0, "end": 5}],
                "person": [{"text": "Steve Jobs", "start": 22, "end": 32}],
            }
        }
    )

    model.classify_text = MagicMock(return_value={"sentiment": "positive"})

    model.extract_json = MagicMock(
        return_value={"companies": [{"name": "Apple", "founded": "1976"}]}
    )

    # Remove predict_entities so it's detected as GLiNER2
    del model.predict_entities

    return model


def _make_ner_task():
    return Task(
        name="test_ner",
        description="Extract entities",
        input_schema={"text": "str"},
        output_schema={"entities": [{"text": "str", "start": "int", "end": "int", "type": "str"}]},
        type=TaskType.NER,
        text_field="text",
    )


# =========================================================================
# GlinerObserver — attach / detach
# =========================================================================


class TestGlinerObserverAttach:
    def test_attach_gliner(self):
        """GLiNER model auto-detected via predict_entities."""
        model = _make_gliner_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        assert obs._method_name == "predict_entities"
        assert obs._task_kind == "ner"

    def test_attach_gliner2_default(self):
        """GLiNER2 model without method defaults to extract_entities."""
        model = _make_gliner2_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        assert obs._method_name == "extract_entities"
        assert obs._task_kind == "ner"

    def test_attach_gliner2_classify(self):
        """GLiNER2 with method='classify_text' → classification."""
        model = _make_gliner2_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="classify_text")
        obs.attach(model)

        assert obs._method_name == "classify_text"
        assert obs._task_kind == "classification"

    def test_attach_gliner2_extract_json(self):
        """GLiNER2 with method='extract_json' → transformation."""
        model = _make_gliner2_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="extract_json")
        obs.attach(model)

        assert obs._method_name == "extract_json"
        assert obs._task_kind == "transformation"

    def test_attach_invalid_model(self):
        """Model without predict_entities or extract_entities raises TypeError."""
        model = MagicMock(spec=[])
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)

        with pytest.raises(TypeError, match="neither predict_entities.*nor extract_entities"):
            obs.attach(model)

    def test_attach_invalid_method(self):
        """Requested method not on model raises TypeError."""
        model = MagicMock(spec=["extract_entities", "classify_text"])
        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="nonexistent_method")

        with pytest.raises(TypeError, match="no method 'nonexistent_method'"):
            obs.attach(model)

    def test_detach_restores_original(self):
        """Detach restores the original method."""
        model = _make_gliner_model()
        original_fn = model.predict_entities
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        # Method should be patched now
        assert model.predict_entities is not original_fn

        obs.detach()
        assert model.predict_entities is original_fn


# =========================================================================
# GlinerObserver — NER capture (GLiNER)
# =========================================================================


class TestGlinerNERCapture:
    def test_capture_gliner_entities(self):
        """GLiNER predict_entities results are captured to buffer."""
        model = _make_gliner_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        result = model.predict_entities("Apple was founded by Steve Jobs.", ["company", "person"])

        # Result should be passed through unchanged
        assert len(result) == 2
        assert result[0]["text"] == "Apple"

        # Buffer should have the normalized observation
        assert buf.get_stats()["new_examples"] == 1
        example = buf.get_all_examples()[0]
        assert example.input == {"text": "Apple was founded by Steve Jobs."}
        entities = example.output["entities"]
        assert len(entities) == 2
        assert entities[0] == {"text": "Apple", "start": 0, "end": 5, "type": "company"}
        assert entities[1] == {"text": "Steve Jobs", "start": 22, "end": 32, "type": "person"}

    def test_labels_tracked(self):
        """Observed labels are tracked."""
        model = _make_gliner_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        model.predict_entities("test", ["company", "person"])

        stats = obs.get_stats()
        assert set(stats["labels_seen"]) == {"company", "person"}

    def test_skip_flag(self):
        """When _skip is True, observations are not captured."""
        model = _make_gliner_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        obs._skip = True
        model.predict_entities("test", ["company", "person"])

        assert buf.get_stats()["new_examples"] == 0
        assert obs._observed_count == 0

    def test_multiple_calls(self):
        """Multiple calls accumulate in buffer."""
        model = _make_gliner_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        for i in range(5):
            model.predict_entities(f"text {i}", ["company"])

        assert buf.get_stats()["new_examples"] == 5
        assert obs._observed_count == 5

    def test_kwargs_call(self):
        """Calling with keyword arguments works."""
        model = MagicMock()
        model.predict_entities = MagicMock(
            return_value=[
                {"text": "NYC", "label": "location", "start": 5, "end": 8, "score": 0.9},
            ]
        )
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        model.predict_entities(text="I love NYC", labels=["location"])

        example = buf.get_all_examples()[0]
        assert example.input == {"text": "I love NYC"}


# =========================================================================
# GlinerObserver — NER capture (GLiNER2)
# =========================================================================


class TestGliner2NERCapture:
    def test_capture_gliner2_entities(self):
        """GLiNER2 extract_entities results are normalized to flat entity list."""
        model = _make_gliner2_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="extract_entities")
        obs.attach(model)

        result = model.extract_entities("Apple was founded by Steve Jobs.", ["company", "person"])

        # Result should be passed through unchanged
        assert "entities" in result

        # Buffer should have normalized entities
        example = buf.get_all_examples()[0]
        entities = example.output["entities"]
        assert len(entities) == 2
        # Check both entities are present (order may vary by dict iteration)
        types = {e["type"] for e in entities}
        assert types == {"company", "person"}

    def test_gliner2_string_entities(self):
        """GLiNER2 entities as plain strings (no dict) are handled."""
        model = MagicMock()
        model.extract_entities = MagicMock(
            return_value={
                "entities": {
                    "company": ["Apple", "Google"],
                }
            }
        )
        del model.predict_entities

        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="extract_entities")
        obs.attach(model)

        model.extract_entities("Apple and Google", ["company"])

        entities = buf.get_all_examples()[0].output["entities"]
        assert len(entities) == 2
        assert entities[0]["text"] == "Apple"
        assert entities[0]["type"] == "company"


# =========================================================================
# GlinerObserver — Classification capture (GLiNER2)
# =========================================================================


class TestGliner2ClassificationCapture:
    def test_capture_classification(self):
        """GLiNER2 classify_text results are captured."""
        model = _make_gliner2_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="classify_text")
        obs.attach(model)

        result = model.classify_text("Great product!", {"sentiment": ["positive", "negative"]})

        assert result == {"sentiment": "positive"}

        example = buf.get_all_examples()[0]
        assert example.output == {"label": "positive"}
        assert example.metadata["task_kind"] == "classification"

    def test_classification_with_confidence(self):
        """Classification result with confidence dict."""
        model = MagicMock()
        model.classify_text = MagicMock(
            return_value={"category": {"label": "tech", "confidence": 0.95}}
        )
        del model.predict_entities

        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="classify_text")
        obs.attach(model)

        model.classify_text("New iPhone released", {"category": ["tech", "sports"]})

        assert buf.get_all_examples()[0].output == {"label": "tech"}

    def test_classification_list_value(self):
        """Classification result with list value takes first."""
        model = MagicMock()
        model.classify_text = MagicMock(return_value={"topic": ["science", "technology"]})
        del model.predict_entities

        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="classify_text")
        obs.attach(model)

        model.classify_text("Quantum computing breakthrough", {"topic": ["science", "technology"]})

        assert buf.get_all_examples()[0].output == {"label": "science"}

    def test_classification_labels_tracked(self):
        """Classification labels are tracked."""
        model = _make_gliner2_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="classify_text")
        obs.attach(model)

        model.classify_text("test", {"sentiment": ["pos", "neg"]})

        assert "positive" in obs._observed_labels


# =========================================================================
# GlinerObserver — Transformation capture (GLiNER2)
# =========================================================================


class TestGliner2TransformationCapture:
    def test_capture_transformation(self):
        """GLiNER2 extract_json results are passed through."""
        model = _make_gliner2_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="extract_json")
        obs.attach(model)

        result = model.extract_json(
            "Apple was founded in 1976",
            {"companies": [{"name": "str", "founded": "str"}]},
        )

        assert result == {"companies": [{"name": "Apple", "founded": "1976"}]}

        example = buf.get_all_examples()[0]
        assert example.output == {"companies": [{"name": "Apple", "founded": "1976"}]}
        assert example.metadata["task_kind"] == "transformation"


# =========================================================================
# GlinerObserver — build_task
# =========================================================================


class TestGlinerBuildTask:
    def test_build_ner_task(self):
        """build_task creates NER task with observed labels."""
        model = _make_gliner_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        model.predict_entities("test", ["company", "person"])

        task = obs.build_task()
        assert task.type == TaskType.NER
        assert task.name == "gliner_ner"
        assert "company" in task.description
        assert "person" in task.description
        assert task.text_field == "text"

    def test_build_classification_task(self):
        """build_task creates CLASSIFICATION task."""
        model = _make_gliner2_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="classify_text")
        obs.attach(model)

        model.classify_text("test", {"sentiment": ["pos", "neg"]})

        task = obs.build_task()
        assert task.type == TaskType.CLASSIFICATION
        assert task.name == "gliner_classify"

    def test_build_transformation_task(self):
        """build_task creates TRANSFORMATION task."""
        model = _make_gliner2_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf, method="extract_json")
        obs.attach(model)

        model.extract_json("test", {"items": [{"name": "str"}]})

        task = obs.build_task()
        assert task.type == TaskType.TRANSFORMATION
        assert task.name == "gliner_extract"

    def test_build_task_custom_name(self):
        """build_task accepts custom name."""
        model = _make_gliner_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)
        model.predict_entities("test", ["company"])

        task = obs.build_task(name="my_ner_task")
        assert task.name == "my_ner_task"

    def test_build_task_no_observations_raises(self):
        """build_task raises when no observations yet."""
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs._task_kind = "ner"

        with pytest.raises(RuntimeError, match="No observations yet"):
            obs.build_task()


# =========================================================================
# GlinerObserver — get_stats
# =========================================================================


class TestGlinerStats:
    def test_stats(self):
        model = _make_gliner_model()
        buf = ExampleBuffer()
        obs = GlinerObserver(buf)
        obs.attach(model)

        model.predict_entities("text1", ["company"])
        model.predict_entities("text2", ["person"])

        stats = obs.get_stats()
        assert stats["observed"] == 2
        assert stats["task_kind"] == "ner"
        assert stats["method"] == "predict_entities"
        assert set(stats["labels_seen"]) == {"company", "person"}


# =========================================================================
# RuleChef integration — start_observing_gliner / stop_observing_gliner
# =========================================================================


class TestRuleChefGlinerIntegration:
    def test_start_observing_gliner(self):
        """start_observing_gliner patches model and creates observer."""
        model = _make_gliner_model()
        chef = RuleChef(client=MagicMock())

        returned = chef.start_observing_gliner(model, auto_learn=False)

        assert returned is model
        assert chef._observations._gliner_observer is not None
        assert chef._observations._gliner_observer._task_kind == "ner"

    def test_stop_observing_gliner(self):
        """stop_observing_gliner detaches and clears observer."""
        model = _make_gliner_model()
        chef = RuleChef(client=MagicMock())
        chef.start_observing_gliner(model, auto_learn=False)

        chef.stop_observing_gliner()

        assert chef._observations._gliner_observer is None

    def test_stop_observing_also_detaches_gliner(self):
        """stop_observing() detaches GLiNER observer too."""
        model = _make_gliner_model()
        chef = RuleChef(client=MagicMock())
        chef.start_observing_gliner(model, auto_learn=False)

        chef.stop_observing()

        assert chef._observations._gliner_observer is None

    def test_gliner_observations_reach_buffer(self):
        """Observations from GLiNER reach the buffer via RuleChef."""
        model = _make_gliner_model()
        chef = RuleChef(client=MagicMock())
        chef.start_observing_gliner(model, auto_learn=False)

        model.predict_entities("Apple founded by Jobs", ["company", "person"])

        stats = chef.get_buffer_stats()
        assert stats["new_examples"] == 1
        assert "gliner_observer" in stats

    def test_gliner_with_task(self):
        """GLiNER observation with pre-defined task."""
        model = _make_gliner_model()
        task = _make_ner_task()
        chef = RuleChef(task=task, client=MagicMock())
        chef.start_observing_gliner(model, auto_learn=False)

        model.predict_entities("Apple", ["company"])

        assert chef.buffer.get_stats()["new_examples"] == 1

    def test_gliner2_classification_integration(self):
        """GLiNER2 classification via start_observing_gliner."""
        model = _make_gliner2_model()
        chef = RuleChef(client=MagicMock())
        chef.start_observing_gliner(model, method="classify_text", auto_learn=False)

        model.classify_text("Great!", {"sentiment": ["pos", "neg"]})

        example = chef.buffer.get_all_examples()[0]
        assert example.output == {"label": "positive"}
