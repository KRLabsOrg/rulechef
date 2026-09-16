import json
import time
from pathlib import Path
from types import SimpleNamespace

from benchmarks.chef_setup import patch_regex_timeout
from benchmarks.schemas import NEROutput
from ner_datasets import load_ner_dataset_from_conll
from ner_datasets.conversion import make_dataset
from rulechef.core import Rule, RuleFormat, Task, TaskType
from rulechef.evaluation import evaluate_dataset, evaluate_rules_individually
from rulechef.executor import RuleExecutor

t0 = time.time()
saved = json.loads(
    Path("reports/germanler/Qwen_Qwen3.5-35B-A3B/ORG/2026-06-11/results_germanler.json").read_text()
)
# saved = json.loads(
#   Path(
#      "reports/findok/Qwen_Qwen3.5-35B-A3B/organisation/2026-05-21/results_ris_transfer.json"
# ).read_text()
# )

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
]
print(f"Loaded {len(rules)} rules in {time.time() - t0:.2f}s")

t0 = time.time()
test_data_raw = load_ner_dataset_from_conll("/share/nverdha/data/ler/validation.conllu")
# test_data_raw = load_ner_dataset_from_conll("/share/nverdha/data/ris/corrected/ris_dev_corrected_patched_v2.conllu")

print(f"Loaded conll in {time.time() - t0:.2f}s, {len(test_data_raw.samples)} docs")

t0 = time.time()
test_data = [
    {"text": sent.text, "entities": sent.labels, "sent_id": sent.sent_id, "doc_id": s.doc_id}
    for s in test_data_raw.samples
    for sent in s.sentences
]
print(f"Built test_data in {time.time() - t0:.2f}s, {len(test_data)} sentences")

task = Task(
    name="German Legal Named Entity Recognition",
    description="Recognize named entities in German legal text. Entities to look for: ORG.",
    input_schema={"text": "str"},
    output_schema=NEROutput,
    type=TaskType.NER,
    text_field="text",
)
executor = RuleExecutor()
patch_regex_timeout(executor)
apply_rules_fn = executor.apply_rules

t0 = time.time()
test_dataset = make_dataset("germanler_eval", test_data, task)
print(f"make_dataset in {time.time() - t0:.2f}s")

t0 = time.time()
result = evaluate_dataset(rules, test_dataset, apply_rules_fn, mode="text")
print(f"evaluate_dataset (all {len(rules)} rules, full set) in {time.time() - t0:.2f}s")

# time a single rule individually evaluated on full dataset
t0 = time.time()
_ = evaluate_rules_individually(
    [rules[0]], test_dataset, apply_rules_fn, mode="text", max_samples=100, iou_threshold=0.5
)
print(f"evaluate_rules_individually for 1 rule on full set in {time.time() - t0:.2f}s")
print(f"=> extrapolated for {len(rules)} rules serially: {(time.time() - t0) * len(rules):.2f}s")
