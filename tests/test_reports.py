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


def test_savings_cli_ner_traffic(tmp_path, monkeypatch):
    rules = {
        "rules": [
            {
                "name": "year",
                "format": "regex",
                "content": r"\b\d{4}\b",
                "output_template": {
                    "text": "$0",
                    "start": "$start",
                    "end": "$end",
                    "type": "DATETIME",
                },
                "output_key": "entities",
            }
        ]
    }
    rf = tmp_path / "rules.json"
    rf.write_text(json.dumps(rules))
    tf = tmp_path / "traffic.jsonl"
    tf.write_text(
        json.dumps(
            {
                "text": "filed in 2006",
                "llm_entities": [{"text": "2006", "start": 9, "end": 13, "type": "DATETIME"}],
            }
        )
        + "\n"
        + json.dumps({"text": "code 1234 here", "llm_entities": []})  # rule fires -> FP
        + "\n"
    )
    out = tmp_path / "s.html"
    monkeypatch.setattr(
        "sys.argv",
        ["rulechef-savings", "--rules", str(rf), "--traffic", str(tf), "--out", str(out)],
    )
    savings_main()
    html_doc = out.read_text()
    assert "KR" in html_doc and "replaceable" in html_doc
    assert "100%" in html_doc  # both rows answered (rule fired on each)
    assert "67%" in html_doc  # micro-F1: TP=1, FP=1, FN=0 -> P=.5 R=1 F1=.667
