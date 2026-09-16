import json
import re
import signal
import time
from pathlib import Path

from ner_datasets import load_ner_dataset_from_conll
from rulechef.core import Rule, RuleFormat


class TimeoutError_(Exception):
    pass


def _handler(signum, frame):
    raise TimeoutError_()


saved = json.loads(
    Path("reports/germanler/Qwen_Qwen3.5-35B-A3B/ORG/2026-06-11/results_germanler.json").read_text()
)
rules = [
    Rule(
        id=r["id"],
        name=r["name"],
        description=r["description"],
        format=RuleFormat(r["format"]),
        content=r["content"],
        output_template=r.get("output_template"),
        output_key=r.get("output_key"),
    )
    for r in saved["rules"]
    if RuleFormat(r["format"]) == RuleFormat.REGEX
]
print(f"Checking {len(rules)} regex rules")

test_data_raw = load_ner_dataset_from_conll("/share/nverdha/data/ler/validation.conllu")
sentences = [sent.text for s in test_data_raw.samples for sent in s.sentences]
print(f"Against {len(sentences)} sentences")

signal.signal(signal.SIGALRM, _handler)

for rule in rules:
    pattern = re.compile(rule.content)
    for i, text in enumerate(sentences):
        signal.setitimer(signal.ITIMER_REAL, 1.0)  # 1s timeout per call
        t0 = time.time()
        try:
            list(pattern.finditer(text))
        except TimeoutError_:
            print(f"SLOW: rule={rule.id} ({rule.name}) sentence_idx={i}")
            print(f"  text: {text[:200]!r}")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        dt = time.time() - t0
        if dt > 0.1:
            print(f"  rule={rule.id} sentence_idx={i} took {dt:.2f}s")
