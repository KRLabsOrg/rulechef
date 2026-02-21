"""Tests for rulechef.openai_wrapper — OpenAI observation/discovery/mapping."""

import json
from unittest.mock import MagicMock

import pytest

from rulechef.buffer import ExampleBuffer
from rulechef.core import Task, TaskType
from rulechef.openai_wrapper import (
    OpenAIObserver,
    RawObservation,
    _extract_response_content,
)

# =========================================================================
# Helpers
# =========================================================================


def _make_response(content=None, tool_calls=None, parsed=None):
    """Create a mock OpenAI response object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    if parsed is not None:
        msg.parsed = parsed
    else:
        msg.parsed = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_task():
    """Create a simple NER task for testing."""
    return Task(
        name="test_ner",
        description="Extract person names",
        input_schema={"text": "str"},
        output_schema={"entities": "List[dict]"},
        type=TaskType.NER,
        text_field="text",
    )


def _make_client(original_create=None):
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.chat.completions.create = original_create or MagicMock(
        return_value=_make_response(content="Hello!")
    )
    return client


# =========================================================================
# _extract_response_content
# =========================================================================


class TestExtractResponseContent:
    def test_plain_text(self):
        resp = _make_response(content="Hello world")
        assert _extract_response_content(resp) == "Hello world"

    def test_empty_string_preserved(self):
        """Empty string is valid content, not None."""
        resp = _make_response(content="")
        assert _extract_response_content(resp) == ""

    def test_none_content_no_tools(self):
        resp = _make_response(content=None)
        assert _extract_response_content(resp) is None

    def test_tool_calls(self):
        tc = MagicMock()
        tc.function.name = "extract"
        tc.function.arguments = '{"text": "John"}'
        resp = _make_response(content=None, tool_calls=[tc])
        result = _extract_response_content(resp)
        parsed = json.loads(result)
        assert parsed[0]["function"] == "extract"
        assert parsed[0]["arguments"] == '{"text": "John"}'

    def test_parsed_pydantic(self):
        parsed_obj = MagicMock()
        parsed_obj.model_dump_json.return_value = '{"label": "positive"}'
        resp = _make_response(content=None, parsed=parsed_obj)
        result = _extract_response_content(resp)
        assert result == '{"label": "positive"}'

    def test_malformed_response(self):
        """No choices at all → None."""
        resp = MagicMock()
        resp.choices = []
        assert _extract_response_content(resp) is None

    def test_no_choices_attribute(self):
        resp = MagicMock(spec=[])
        assert _extract_response_content(resp) is None


# =========================================================================
# OpenAIObserver — attach / detach
# =========================================================================


class TestObserverAttachDetach:
    def test_attach_replaces_create(self):
        buffer = ExampleBuffer()
        original = MagicMock(return_value=_make_response(content="Hi"))
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
        )
        client = _make_client()
        old_create = client.chat.completions.create
        observer.attach(client)
        assert client.chat.completions.create is not old_create

    def test_detach_restores_create(self):
        buffer = ExampleBuffer()
        original = MagicMock(return_value=_make_response(content="Hi"))
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
        )
        client = _make_client()
        observer.attach(client)
        observer.detach()
        assert client.chat.completions.create is original


# =========================================================================
# OpenAIObserver — raw capture
# =========================================================================


class TestObserverCapture:
    def test_captures_raw_observation(self):
        buffer = ExampleBuffer()
        response = _make_response(content="The answer is 42")
        original = MagicMock(return_value=response)
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
        )
        client = _make_client(original)
        observer.attach(client)

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 6*7?"}],
        )

        stats = observer.get_stats()
        assert stats["observed"] == 1
        assert stats["pending"] == 1
        assert stats["mapped"] == 0

    def test_skip_flag_prevents_capture(self):
        buffer = ExampleBuffer()
        response = _make_response(content="internal")
        original = MagicMock(return_value=response)
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
        )
        client = _make_client(original)
        observer.attach(client)

        observer._skip = True
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "internal call"}],
        )
        observer._skip = False

        assert observer.get_stats()["observed"] == 0

    def test_streaming_captured_after_completion(self):
        """Streaming calls are wrapped and captured after the stream completes."""
        buffer = ExampleBuffer()

        # Simulate stream chunks
        def make_chunk(content):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = content
            chunk.choices[0].delta.tool_calls = None
            return chunk

        chunks = [make_chunk("Hello"), make_chunk(" world"), make_chunk("!")]
        stream = iter(chunks)

        original = MagicMock(return_value=stream)
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
        )
        client = _make_client(original)
        observer.attach(client)

        # Make a streaming call
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "say hello"}],
            stream=True,
        )

        # Consume the stream (user iterates as normal)
        collected = list(result)
        assert len(collected) == 3

        # After stream completes, observation should be captured
        stats = observer.get_stats()
        assert stats["observed"] == 1
        assert stats["streaming_captured"] == 1

        # Verify captured content is the reassembled text
        obs = observer._raw_observations[0]
        assert obs.response_content == "Hello world!"
        assert obs.metadata["streamed"] is True

    def test_streaming_with_context_manager(self):
        """Stream wrapper works as a context manager."""
        buffer = ExampleBuffer()

        def make_chunk(content):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = content
            chunk.choices[0].delta.tool_calls = None
            return chunk

        chunks = [make_chunk("Hi")]
        stream = MagicMock()
        stream.__iter__ = MagicMock(return_value=iter(chunks))
        stream.__next__ = MagicMock(side_effect=iter(chunks).__next__)
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)

        original = MagicMock(return_value=stream)
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
        )
        client = _make_client(original)
        observer.attach(client)

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
        )

        with result as s:
            for _ in s:
                pass

        assert observer.get_stats()["observed"] == 1

    def test_none_content_skipped(self):
        """If response has no extractable content, skip it."""
        buffer = ExampleBuffer()
        response = _make_response(content=None)
        original = MagicMock(return_value=response)
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
        )
        client = _make_client(original)
        observer.attach(client)

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
        )

        stats = observer.get_stats()
        assert stats["observed"] == 0
        assert stats["skipped"] == 1


# =========================================================================
# OpenAIObserver — custom extractors
# =========================================================================


class TestObserverCustomExtractors:
    def test_custom_extractors_add_to_buffer(self):
        buffer = ExampleBuffer()
        response = _make_response(content="John lives in NYC")

        def extract_input(kwargs):
            msgs = kwargs.get("messages", [])
            return {"text": msgs[-1]["content"]} if msgs else None

        def extract_output(resp):
            return {"entities": [{"text": "John", "type": "PERSON"}]}

        original = MagicMock(return_value=response)
        observer = OpenAIObserver(
            buffer=buffer,
            task=_make_task(),
            original_create=original,
            extract_input=extract_input,
            extract_output=extract_output,
        )
        client = _make_client(original)
        observer.attach(client)

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "John lives in NYC"}],
        )

        stats = buffer.get_stats()
        assert stats["llm_observations"] == 1
        # Raw observation store should be empty (custom mode skips it)
        assert observer.get_stats()["observed"] == 0


# =========================================================================
# OpenAIObserver — discover_task
# =========================================================================


class TestDiscoverTask:
    def test_too_few_observations_raises(self):
        buffer = ExampleBuffer()
        original = MagicMock()
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
            min_observations_for_discovery=5,
        )
        # Only 2 observations
        observer._raw_observations = [
            RawObservation(
                messages=[{"role": "user", "content": "hello"}],
                response_content="world",
            )
            for _ in range(2)
        ]

        with pytest.raises(ValueError, match="Need at least 5"):
            observer.discover_task(MagicMock(), "gpt-4o-mini")

    def test_discovers_task_from_response(self):
        buffer = ExampleBuffer()
        discovery_response = _make_response(
            content=json.dumps(
                {
                    "name": "sentiment_analysis",
                    "description": "Classify sentiment of text",
                    "type": "classification",
                    "input_schema": {"text": "str"},
                    "output_schema": {"label": "str"},
                    "text_field": "text",
                }
            )
        )
        original = MagicMock()
        llm_client = _make_client(MagicMock(return_value=discovery_response))
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
            min_observations_for_discovery=3,
        )
        # Add enough observations
        observer._raw_observations = [
            RawObservation(
                messages=[{"role": "user", "content": f"text {i}"}],
                response_content=f"response {i}",
            )
            for i in range(5)
        ]

        task = observer.discover_task(llm_client, "gpt-4o-mini")
        assert task.name == "sentiment_analysis"
        assert task.type == TaskType.CLASSIFICATION
        assert task.input_schema == {"text": "str"}
        assert task.text_field == "text"

    def test_skip_flag_during_discovery(self):
        """discover_task sets _skip=True during its own LLM call."""
        buffer = ExampleBuffer()
        skip_values = []

        def capturing_create(**kwargs):
            skip_values.append(observer._skip)
            return _make_response(
                content=json.dumps(
                    {
                        "name": "test",
                        "description": "test",
                        "type": "extraction",
                        "input_schema": {"text": "str"},
                        "output_schema": {"spans": "list"},
                        "text_field": "text",
                    }
                )
            )

        original = MagicMock()
        llm_client = _make_client(MagicMock(side_effect=capturing_create))
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=original,
            min_observations_for_discovery=1,
        )
        observer._raw_observations = [
            RawObservation(
                messages=[{"role": "user", "content": "test"}],
                response_content="test",
            )
        ]

        observer.discover_task(llm_client, "gpt-4o-mini")
        assert skip_values[0] is True
        # After call, skip should be restored
        assert observer._skip is False


# =========================================================================
# OpenAIObserver — map_pending
# =========================================================================


class TestMapPending:
    def test_maps_batch_and_advances_cursor(self):
        buffer = ExampleBuffer()
        task = _make_task()

        mapping_response = _make_response(
            content=json.dumps(
                [
                    {
                        "relevant": True,
                        "input": {"text": "John went home"},
                        "output": {"entities": [{"text": "John", "type": "PERSON"}]},
                    },
                    {"relevant": False, "input": None, "output": None},
                ]
            )
        )
        original = MagicMock()
        llm_client = _make_client(MagicMock(return_value=mapping_response))
        observer = OpenAIObserver(
            buffer=buffer,
            task=task,
            original_create=original,
        )
        observer._raw_observations = [
            RawObservation(
                messages=[{"role": "user", "content": "John went home"}],
                response_content='{"entities": [{"text": "John"}]}',
            ),
            RawObservation(
                messages=[{"role": "user", "content": "What time is it?"}],
                response_content="It's 3pm",
            ),
        ]

        added = observer.map_pending(task, llm_client, "gpt-4o-mini")
        assert added == 1  # Only first was relevant

        stats = observer.get_stats()
        assert stats["mapped"] == 2  # Cursor advanced past both
        assert stats["pending"] == 0

        # Buffer should have the mapped example
        buf_stats = buffer.get_stats()
        assert buf_stats["llm_observations"] == 1

    def test_no_pending_returns_zero(self):
        buffer = ExampleBuffer()
        task = _make_task()
        original = MagicMock()
        observer = OpenAIObserver(
            buffer=buffer,
            task=task,
            original_create=original,
        )
        assert observer.map_pending(task, MagicMock(), "gpt-4o-mini") == 0

    def test_handles_irrelevant_observations(self):
        buffer = ExampleBuffer()
        task = _make_task()

        mapping_response = _make_response(
            content=json.dumps(
                [
                    {"relevant": False, "input": None, "output": None},
                    {"relevant": False, "input": None, "output": None},
                ]
            )
        )
        original = MagicMock()
        llm_client = _make_client(MagicMock(return_value=mapping_response))
        observer = OpenAIObserver(
            buffer=buffer,
            task=task,
            original_create=original,
        )
        observer._raw_observations = [
            RawObservation(
                messages=[{"role": "user", "content": f"irrelevant {i}"}],
                response_content=f"irrelevant {i}",
            )
            for i in range(2)
        ]

        added = observer.map_pending(task, llm_client, "gpt-4o-mini")
        assert added == 0
        assert observer.get_stats()["skipped"] == 2

    def test_batch_failure_increments_failed(self, capsys):
        buffer = ExampleBuffer()
        task = _make_task()

        original = MagicMock()
        llm_client = _make_client(MagicMock(side_effect=Exception("API error")))
        observer = OpenAIObserver(
            buffer=buffer,
            task=task,
            original_create=original,
        )
        observer._raw_observations = [
            RawObservation(
                messages=[{"role": "user", "content": "test"}],
                response_content="test",
            )
        ]

        added = observer.map_pending(task, llm_client, "gpt-4o-mini")
        assert added == 0
        assert observer.get_stats()["failed"] == 1


# =========================================================================
# OpenAIObserver — stats
# =========================================================================


class TestObserverStats:
    def test_initial_stats(self):
        buffer = ExampleBuffer()
        observer = OpenAIObserver(
            buffer=buffer,
            task=None,
            original_create=MagicMock(),
        )
        stats = observer.get_stats()
        assert stats == {
            "observed": 0,
            "mapped": 0,
            "pending": 0,
            "skipped": 0,
            "failed": 0,
            "streaming_captured": 0,
        }


# =========================================================================
# JSON parsing helper
# =========================================================================


class TestParseJson:
    def test_plain_json(self):
        result = OpenAIObserver._parse_json('{"key": "value"}', "test")
        assert result == {"key": "value"}

    def test_markdown_json_block(self):
        raw = '```json\n{"key": "value"}\n```'
        result = OpenAIObserver._parse_json(raw, "test")
        assert result == {"key": "value"}

    def test_generic_code_block(self):
        raw = "```\n[1, 2, 3]\n```"
        result = OpenAIObserver._parse_json(raw, "test")
        assert result == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            OpenAIObserver._parse_json("not json at all", "test")
