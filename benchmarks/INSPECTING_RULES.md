# Inspecting learned rules

## Load a saved ruleset into RuleChef

`RuleChef.load_rules` accepts any of: the per-stage checkpoint
(`*.ckpt_rulechef.json`), the three-way comparison result
(`results_extract_*.json`), a dataset JSON, or a bare list of rule dicts.

```python
from rulechef import RuleChef
from rulechef.core import Task, TaskType

task = Task(
    name="tab", description="extract entities",
    input_schema={"text": "str"},
    output_schema={"entities": "List[{text,start,end,type}]"},
    type=TaskType.NER, text_field="text",
)
chef = RuleChef(task=task)            # client optional; not needed just to run rules

chef.load_rules("benchmarks/results/results_extract_tab.ckpt_rulechef.json")
# chef.load_rules("benchmarks/results/results_extract_tab.json")   # also works

# Run the rules (deterministic, no LLM):
print(chef.extract({"text": "filed under no. 36244/06 in 2006"}))
```

### Per-rule TP / FP / F1

These need gold examples in the dataset to score against:

```python
for ex in gold:                      # gold: list of {"text": ..., "entities": [...]}
    chef.add_example({"text": ex["text"]}, {"entities": ex["entities"]})

for m in chef.get_rule_metrics(verbose=True):
    print(m.rule_name, m.precision, m.recall, m.f1, m.true_positives, m.false_positives)

chef.rank_rules()    # solo + leave-one-out marginal F1
```

## Browsable HTML report

`rule_report.py` runs each rule over labeled data, classifies every match as a
true/false positive, and writes a self-contained HTML file (rules sorted by
precision; click a rule to see its TP/FP with the matched span highlighted in
context).

```bash
# On the TAB gold data (built-in loader), full 600-doc test split:
python benchmarks/rule_report.py \
  --rules benchmarks/results/results_extract_tab.ckpt_rulechef.json \
  --dataset tab --test 600 --out rules_tab.html
# → open rules_tab.html in a browser

# On any other labeled data — a JSONL file, one object per line:
#   {"text": "...", "entities": [{"text": "Smith", "start": 0, "end": 5, "type": "PERSON"}]}
python benchmarks/rule_report.py \
  --rules path/to/your_rules.json \
  --data path/to/gold.jsonl --out report.html
```

The generated `*.html` is git-ignored, so producing one won't dirty the repo.

> Note: the report recomputes precision live with same-type span-overlap
> matching, which is slightly more lenient than the benchmark's exact scorer.

## Savings report (branded)

Replay a ruleset over observed LLM traffic and get a print-ready, KR Labs-branded
report of how many calls the rules could take over:

```bash
rulechef-savings \
  --rules benchmarks/results/results_banking77_new.json \
  --traffic benchmarks/results/sample_traffic_banking77.jsonl \
  --cost-per-call 0.002 --out savings.html
```

Traffic JSONL: one call per line, `{"text": ..., "llm_label": ..., "gold_label": (optional)}`.
Print to PDF from the browser, or:
`chrome --headless --print-to-pdf=savings.pdf savings.html`

### Producing the traffic file from observation mode

If you're already using `chef.start_observing(...)` / `chef.add_observation(...)`
to capture real LLM calls, you don't need to hand-write the traffic JSONL —
`export_traffic` writes it directly from the observation buffer:

```python
chef = RuleChef(client=client)
wrapped = chef.start_observing(client)
# ... traffic flows through `wrapped` as usual ...

chef.export_traffic("traffic.jsonl")
```

```bash
rulechef-savings --rules rules.json --traffic traffic.jsonl --out savings.html
```

Only LLM-sourced observations are exported (human-labeled examples aren't
traffic). Classification output (`{"label": ...}`) is written as `llm_label`;
NER/extraction output (`{"entities": [...]}` or `{"spans": [...]}`) is written
as `llm_entities` — note `rulechef-savings` itself currently only scores the
classification (`llm_label`) shape.
