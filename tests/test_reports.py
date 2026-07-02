"""Tests for the rulechef-report and rulechef-savings generators."""

import json

from rulechef.report import generate, load_jsonl
from rulechef.savings import main as savings_main


def test_report_generate(tmp_path):
    rules = [
        {
            "name": "year",
            "format": "regex",
            "content": r"\b\d{4}\b",
            "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "DATETIME"},
            "output_key": "entities",
        }
    ]
    gold = [
        {"text": "filed in 2006", "entities": [{"text": "2006", "start": 9, "end": 13, "type": "DATETIME"}]},
        {"text": "code 1234 here", "entities": []},  # 1234 matches -> FP
    ]
    out = tmp_path / "r.html"
    doc = generate(rules, gold, str(out))
    assert out.exists()
    assert "year" in doc and "<mark>" in doc
    assert "1 TP" in doc and "1 FP" in doc


def test_load_jsonl(tmp_path):
    f = tmp_path / "g.jsonl"
    f.write_text('{"text": "a", "entities": []}\n\n{"text": "b", "entities": []}\n')
    assert len(load_jsonl(str(f))) == 2


def test_savings_cli(tmp_path, monkeypatch, capsys):
    rules = {"rules": [{"name": "rate", "format": "regex", "content": r"(?i)exchange rate", "output_template": {"label": "exchange_rate"}, "output_key": "label"}]}
    rf = tmp_path / "rules.json"
    rf.write_text(json.dumps(rules))
    tf = tmp_path / "traffic.jsonl"
    tf.write_text(
        '{"text": "what is the exchange rate", "llm_label": "exchange_rate"}\n'
        '{"text": "card is missing", "llm_label": "card_arrival"}\n'
    )
    out = tmp_path / "s.html"
    monkeypatch.setattr(
        "sys.argv",
        ["rulechef-savings", "--rules", str(rf), "--traffic", str(tf), "--out", str(out)],
    )
    savings_main()
    html_doc = out.read_text()
    assert "KR" in html_doc and "replaceable" in html_doc
    assert "50%" in html_doc  # 1 of 2 calls answered


def test_export_traffic_classification(tmp_path):
    """Test export_traffic writes classification observations as savings-report JSONL."""
    from rulechef import RuleChef
    from rulechef.core import Task, TaskType

    task = Task(
        name="test_classify",
        description="Classify text",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )
    chef = RuleChef(task=task)
    chef.add_observation({"text": "hello"}, {"label": "greeting"})
    chef.add_observation({"text": "bye"}, {"label": "farewell"})

    out = tmp_path / "traffic.jsonl"
    n = chef.export_traffic(str(out))

    assert n == 2
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"text": "hello", "llm_label": "greeting"}
    assert json.loads(lines[1]) == {"text": "bye", "llm_label": "farewell"}


def test_export_traffic_no_task_detects_format(tmp_path):
    """export_traffic auto-detects classification format when no task is set."""
    from rulechef import RuleChef

    chef = RuleChef()
    chef.add_observation({"text": "hello"}, {"label": "greeting"})

    out = tmp_path / "traffic.jsonl"
    n = chef.export_traffic(str(out))

    assert n == 1
    record = json.loads(out.read_text().strip())
    assert record == {"text": "hello", "llm_label": "greeting"}


def test_export_traffic_roundtrip_savings(tmp_path, monkeypatch):
    """Export observations, then replay through rulechef-savings CLI."""
    from rulechef import RuleChef
    from rulechef.core import Task, TaskType

    task = Task(
        name="banking",
        description="Classify banking intents",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )
    chef = RuleChef(task=task)
    chef.add_observation({"text": "what is the exchange rate"}, {"label": "exchange_rate"})
    chef.add_observation({"text": "card is missing"}, {"label": "card_arrival"})
    chef.add_observation({"text": "transfer money please"}, {"label": "transfer"})

    traffic_path = tmp_path / "traffic.jsonl"
    chef.export_traffic(str(traffic_path))

    rules = {"rules": [{"name": "rate", "format": "regex", "content": r"(?i)exchange rate", "output_template": {"label": "exchange_rate"}, "output_key": "label"}]}
    rf = tmp_path / "rules.json"
    rf.write_text(json.dumps(rules))

    out = tmp_path / "s.html"
    monkeypatch.setattr(
        "sys.argv",
        ["rulechef-savings", "--rules", str(rf), "--traffic", str(traffic_path), "--out", str(out)],
    )
    savings_main()
    html_doc = out.read_text()
    assert "KR" in html_doc and "replaceable" in html_doc


def test_export_traffic_skips_human_examples(tmp_path):
    """export_traffic only exports LLM observations, not human examples."""
    from rulechef import RuleChef
    from rulechef.core import Task, TaskType

    task = Task(
        name="test_classify",
        description="Classify text",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )
    chef = RuleChef(task=task)
    chef.add_observation({"text": "a"}, {"label": "A"})            # LLM
    chef.add_example({"text": "b"}, {"label": "B"})                # human

    out = tmp_path / "traffic.jsonl"
    n = chef.export_traffic(str(out))
    assert n == 1
    assert json.loads(out.read_text().strip()) == {"text": "a", "llm_label": "A"}
