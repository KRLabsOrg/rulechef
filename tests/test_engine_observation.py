"""Tests for RuleChef's framework-agnostic observation API."""

import json
from unittest.mock import MagicMock

from rulechef.engine import RuleChef
from rulechef.core import Task, TaskType
from rulechef.openai_wrapper import RawObservation


def _make_task():
    return Task(
        name="test_classify",
        description="Classify text",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )


def _make_response(content):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.parsed = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


# =========================================================================
# add_observation()
# =========================================================================


class TestAddObservation:
    def test_adds_to_buffer(self):
        chef = RuleChef(task=_make_task(), client=MagicMock())
        chef.add_observation({"text": "hello"}, {"label": "greeting"})

        stats = chef.buffer.get_stats()
        assert stats["llm_observations"] == 1
        assert stats["new_examples"] == 1

    def test_no_task_required(self):
        """add_observation works without a task defined."""
        chef = RuleChef(client=MagicMock())
        assert chef.task is None

        chef.add_observation({"text": "hello"}, {"label": "greeting"})

        stats = chef.buffer.get_stats()
        assert stats["llm_observations"] == 1

    def test_metadata_passed_through(self):
        chef = RuleChef(client=MagicMock())
        chef.add_observation(
            {"text": "hello"},
            {"label": "greeting"},
            metadata={"model": "claude-3", "provider": "anthropic"},
        )

        examples = chef.buffer.get_all_examples()
        assert examples[0].metadata["model"] == "claude-3"
        assert examples[0].metadata["provider"] == "anthropic"

    def test_multiple_observations(self):
        chef = RuleChef(client=MagicMock())
        for i in range(10):
            chef.add_observation({"text": f"text {i}"}, {"label": f"label_{i}"})

        assert chef.buffer.get_stats()["new_examples"] == 10


# =========================================================================
# add_raw_observation()
# =========================================================================


class TestAddRawObservation:
    def test_stored_in_pending(self):
        chef = RuleChef(client=MagicMock())
        chef.add_raw_observation(
            messages=[{"role": "user", "content": "classify this"}],
            response="positive",
        )

        assert len(chef._pending_raw_observations) == 1
        obs = chef._pending_raw_observations[0]
        assert isinstance(obs, RawObservation)
        assert obs.messages == [{"role": "user", "content": "classify this"}]
        assert obs.response_content == "positive"

    def test_no_task_required(self):
        chef = RuleChef(client=MagicMock())
        assert chef.task is None

        chef.add_raw_observation(
            messages=[{"role": "user", "content": "test"}],
            response="response",
        )
        assert len(chef._pending_raw_observations) == 1

    def test_metadata_stored(self):
        chef = RuleChef(client=MagicMock())
        chef.add_raw_observation(
            messages=[{"role": "user", "content": "test"}],
            response="response",
            metadata={"model": "llama-3"},
        )
        assert chef._pending_raw_observations[0].metadata == {"model": "llama-3"}

    def test_messages_copied(self):
        """Messages list should be copied, not stored by reference."""
        chef = RuleChef(client=MagicMock())
        msgs = [{"role": "user", "content": "test"}]
        chef.add_raw_observation(messages=msgs, response="ok")
        msgs.append({"role": "assistant", "content": "extra"})

        assert len(chef._pending_raw_observations[0].messages) == 1

    def test_creates_observer_lazily_at_learn(self):
        """Observer is NOT created by add_raw_observation, only at learn time."""
        chef = RuleChef(client=MagicMock())
        chef.add_raw_observation(
            messages=[{"role": "user", "content": "test"}],
            response="response",
        )
        assert chef._observer is None  # not yet created

    def test_shows_in_buffer_stats(self):
        chef = RuleChef(client=MagicMock())
        chef.add_raw_observation(
            messages=[{"role": "user", "content": "test"}],
            response="response",
        )
        stats = chef.get_buffer_stats()
        assert stats["pending_raw_observations"] == 1


# =========================================================================
# Integration: add_raw_observation + learn_rules merge
# =========================================================================


class TestRawObservationMerge:
    def test_merged_into_observer_at_learn(self):
        """Pending raw observations get merged into observer at learn_rules() time."""
        discovery_json = json.dumps(
            {
                "name": "classify",
                "description": "Classify text",
                "type": "classification",
                "input_schema": {"text": "str"},
                "output_schema": {"label": "str"},
                "text_field": "text",
            }
        )
        mapping_json = json.dumps(
            [
                {
                    "relevant": True,
                    "input": {"text": "hello"},
                    "output": {"label": "greeting"},
                },
            ]
        )

        # Mock LLM: first call = discovery, subsequent = mapping, then synthesis
        call_count = [0]

        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_response(discovery_json)
            elif call_count[0] == 2:
                return _make_response(mapping_json)
            else:
                # Synthesis — return a rule
                return _make_response(
                    json.dumps(
                        [
                            {
                                "name": "greeting_rule",
                                "description": "Match greetings",
                                "format": "regex",
                                "content": "hello|hi",
                                "priority": 5,
                                "output_template": {"label": "greeting"},
                            }
                        ]
                    )
                )

        client = MagicMock()
        client.chat.completions.create = MagicMock(side_effect=mock_create)

        chef = RuleChef(client=client, model="test-model")
        chef.add_raw_observation(
            messages=[{"role": "user", "content": "hello"}],
            response="greeting",
        )

        assert chef._observer is None  # not yet
        assert len(chef._pending_raw_observations) == 1

        # learn_rules should merge pending → observer → discover → map → learn
        # This will call the LLM multiple times; we just verify the merge happened
        try:
            chef.learn_rules()
        except Exception:
            pass  # may fail at synthesis with mock, that's OK

        # Pending should be cleared
        assert len(chef._pending_raw_observations) == 0
        # Observer should have been created
        assert chef._observer is not None
        # Raw observation should be in the observer
        assert len(chef._observer._raw_observations) == 1

    def test_discover_task_merges_pending(self):
        """discover_task() also merges pending raw observations."""
        discovery_json = json.dumps(
            {
                "name": "sentiment",
                "description": "Classify sentiment",
                "type": "classification",
                "input_schema": {"text": "str"},
                "output_schema": {"label": "str"},
                "text_field": "text",
            }
        )

        client = MagicMock()
        client.chat.completions.create = MagicMock(
            return_value=_make_response(discovery_json)
        )

        chef = RuleChef(client=client, model="test-model")
        for i in range(5):
            chef.add_raw_observation(
                messages=[{"role": "user", "content": f"text {i}"}],
                response=f"label_{i}",
            )

        task = chef.discover_task()
        assert task.name == "sentiment"
        assert task.type == TaskType.CLASSIFICATION
        assert len(chef._pending_raw_observations) == 0
        assert chef._observer is not None

    def test_mixed_raw_and_monkey_patch(self):
        """Raw observations from both add_raw_observation and start_observing coexist."""
        client = MagicMock()
        client.chat.completions.create = MagicMock(
            return_value=_make_response("some response")
        )

        chef = RuleChef(client=client, model="test-model")

        # Add via add_raw_observation
        chef.add_raw_observation(
            messages=[{"role": "user", "content": "manual"}],
            response="manual_response",
        )

        # Start observing (monkey-patch)
        chef.start_observing(client, auto_learn=False)

        # Make a call through the monkey-patched client
        client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "observed"}],
        )

        # Observer has 1 from monkey-patch
        assert chef._observer.get_stats()["observed"] == 1
        # Pending has 1 from manual
        assert len(chef._pending_raw_observations) == 1

        chef.stop_observing()
