"""Tests for TrainingDataLogger."""

import json

from rulechef.training_logger import TrainingDataLogger


def test_basic_logging(tmp_path):
    """Log a single entry and verify JSONL output."""
    path = tmp_path / "train.jsonl"
    logger = TrainingDataLogger(str(path))

    logger.log(
        "rule_synthesis",
        [{"role": "user", "content": "Generate rules"}],
        '{"rules": []}',
        {"task_name": "test", "dataset_size": 10},
    )

    assert logger.count == 1
    assert logger.stats == {"rule_synthesis": 1}

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["call_type"] == "rule_synthesis"
    assert len(entry["messages"]) == 2
    assert entry["messages"][0]["role"] == "user"
    assert entry["messages"][1]["role"] == "assistant"
    assert entry["messages"][1]["content"] == '{"rules": []}'
    assert entry["metadata"]["task_name"] == "test"
    assert entry["metadata"]["dataset_size"] == 10
    assert "timestamp" in entry


def test_multiple_call_types(tmp_path):
    """Log different call types and verify stats."""
    path = tmp_path / "train.jsonl"
    logger = TrainingDataLogger(str(path))

    logger.log("rule_synthesis", [{"role": "user", "content": "a"}], "b")
    logger.log("rule_patch", [{"role": "user", "content": "c"}], "d")
    logger.log("rule_patch", [{"role": "user", "content": "e"}], "f")
    logger.log("guide_refinement", [{"role": "user", "content": "g"}], "h")

    assert logger.count == 4
    assert logger.stats == {
        "rule_synthesis": 1,
        "rule_patch": 2,
        "guide_refinement": 1,
    }

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 4


def test_run_metadata(tmp_path):
    """Run metadata is merged into every entry."""
    path = tmp_path / "train.jsonl"
    logger = TrainingDataLogger(str(path), run_metadata={"model": "kimi-k2", "run_id": "run_001"})

    logger.log("rule_synthesis", [{"role": "user", "content": "a"}], "b", {"task_name": "NER"})

    entry = json.loads(path.read_text().strip())
    assert entry["metadata"]["model"] == "kimi-k2"
    assert entry["metadata"]["run_id"] == "run_001"
    assert entry["metadata"]["task_name"] == "NER"


def test_append_mode(tmp_path):
    """Multiple logger instances append to the same file."""
    path = tmp_path / "train.jsonl"

    logger1 = TrainingDataLogger(str(path))
    logger1.log("rule_synthesis", [{"role": "user", "content": "a"}], "b")

    logger2 = TrainingDataLogger(str(path))
    logger2.log("rule_patch", [{"role": "user", "content": "c"}], "d")

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["call_type"] == "rule_synthesis"
    assert json.loads(lines[1])["call_type"] == "rule_patch"


def test_creates_parent_directories(tmp_path):
    """Logger creates parent directories if they don't exist."""
    path = tmp_path / "nested" / "deep" / "train.jsonl"
    logger = TrainingDataLogger(str(path))
    logger.log("test", [{"role": "user", "content": "a"}], "b")
    assert path.exists()


def test_repr(tmp_path):
    """Repr includes path and count."""
    path = tmp_path / "train.jsonl"
    logger = TrainingDataLogger(str(path))
    assert "entries=0" in repr(logger)

    logger.log("test", [{"role": "user", "content": "a"}], "b")
    assert "entries=1" in repr(logger)


def test_none_metadata(tmp_path):
    """Metadata defaults to empty dict when None."""
    path = tmp_path / "train.jsonl"
    logger = TrainingDataLogger(str(path))
    logger.log("test", [{"role": "user", "content": "a"}], "b", metadata=None)

    entry = json.loads(path.read_text().strip())
    assert entry["metadata"] == {}


def test_messages_format(tmp_path):
    """System + user messages are preserved, assistant response appended."""
    path = tmp_path / "train.jsonl"
    logger = TrainingDataLogger(str(path))

    messages = [
        {"role": "system", "content": "You are a rule generator."},
        {"role": "user", "content": "Generate rules for NER."},
    ]
    logger.log("rule_synthesis", messages, "Here are the rules...")

    entry = json.loads(path.read_text().strip())
    assert len(entry["messages"]) == 3
    assert entry["messages"][0]["role"] == "system"
    assert entry["messages"][1]["role"] == "user"
    assert entry["messages"][2]["role"] == "assistant"
    assert entry["messages"][2]["content"] == "Here are the rules..."

    # Original messages list should not be mutated
    assert len(messages) == 2


def test_import_from_package():
    """TrainingDataLogger is importable from the rulechef package."""
    from rulechef import TrainingDataLogger as TDL

    assert TDL is TrainingDataLogger


def test_observer_logs_task_discovery(tmp_path):
    """OpenAIObserver logs task_discovery calls when training_logger is set."""
    from unittest.mock import MagicMock

    from rulechef.openai_wrapper import OpenAIObserver

    path = tmp_path / "train.jsonl"
    logger = TrainingDataLogger(str(path))

    # Mock LLM client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(
        {
            "name": "Medical NER",
            "description": "Extract entities",
            "type": "ner",
            "input_schema": {"text": "str"},
            "output_schema": {"entities": "List"},
            "text_field": "text",
        }
    )
    mock_client.chat.completions.create.return_value = mock_response

    observer = OpenAIObserver(
        buffer=MagicMock(),
        task=None,
        original_create=mock_client.chat.completions.create,
        min_observations_for_discovery=1,
        training_logger=logger,
    )

    # Add a raw observation so discovery has data
    from rulechef.openai_wrapper import RawObservation

    observer._raw_observations.append(
        RawObservation(
            messages=[{"role": "user", "content": "Extract from: Aspirin 500mg"}],
            response_content='{"entities": []}',
        )
    )

    observer.discover_task(mock_client, "test-model")

    assert logger.count == 1
    assert logger.stats == {"task_discovery": 1}
    entry = json.loads(path.read_text().strip())
    assert entry["call_type"] == "task_discovery"
    assert entry["metadata"]["discovered_task_name"] == "Medical NER"
    assert entry["metadata"]["num_observations"] == 1


def test_observer_logs_observation_mapping(tmp_path):
    """OpenAIObserver logs observation_mapping calls when training_logger is set."""
    from unittest.mock import MagicMock

    from rulechef.core import Task, TaskType
    from rulechef.openai_wrapper import OpenAIObserver, RawObservation

    path = tmp_path / "train.jsonl"
    logger = TrainingDataLogger(str(path))

    task = Task(
        name="Test NER",
        description="Test",
        input_schema={"text": "str"},
        output_schema={"entities": "List"},
        type=TaskType.NER,
        text_field="text",
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(
        [
            {"relevant": True, "input": {"text": "Aspirin 500mg"}, "output": {"entities": []}},
            {"relevant": False, "input": None, "output": None},
        ]
    )
    mock_client.chat.completions.create.return_value = mock_response

    observer = OpenAIObserver(
        buffer=MagicMock(),
        task=task,
        original_create=mock_client.chat.completions.create,
        training_logger=logger,
    )

    batch = [
        RawObservation(messages=[{"role": "user", "content": "a"}], response_content="b"),
        RawObservation(messages=[{"role": "user", "content": "c"}], response_content="d"),
    ]

    observer._map_batch(task, batch, mock_client, "test-model")

    assert logger.count == 1
    assert logger.stats == {"observation_mapping": 1}
    entry = json.loads(path.read_text().strip())
    assert entry["call_type"] == "observation_mapping"
    assert entry["metadata"]["batch_size"] == 2
    assert entry["metadata"]["relevant_count"] == 1
    assert entry["metadata"]["task_name"] == "Test NER"
