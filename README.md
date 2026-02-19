# RuleChef

<p align="center">
  <img src="https://github.com/KRLabsOrg/rulechef/blob/main/assets/mascot.png?raw=true" alt="RuleChef" width="350"/>
</p>

<p align="center">
  <strong>Learn rule-based models from examples using LLM-powered synthesis.</strong><br>
  Replace expensive LLM calls with fast, deterministic, inspectable rules.
</p>

<p align="center">
  <a href="https://github.com/KRLabsOrg/rulechef/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://pypi.org/project/rulechef/">
    <img src="https://img.shields.io/pypi/v/rulechef.svg" alt="PyPI">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  </a>
</p>

---

## What is RuleChef?

RuleChef learns regex, Python code, and spaCy patterns from labeled examples using LLM-powered synthesis. You provide examples, RuleChef generates rules, and those rules run locally without any LLM at inference time.

**Why rules instead of LLMs?**
- **Cost**: Rules cost nothing to run. No API calls, no tokens.
- **Latency**: Sub-millisecond per query vs hundreds of ms for LLM calls.
- **Determinism**: Same input always produces the same output.
- **Inspectability**: You can read, edit, and debug every rule.
- **No drift**: Rules don't change unless you change them.

## Installation

```bash
pip install rulechef
```

**Extras:**
```bash
pip install rulechef[grex]     # Regex pattern suggestions from examples
pip install rulechef[spacy]    # spaCy token/dependency matcher patterns
pip install rulechef[agentic]  # LLM-powered coordinator for adaptive learning
pip install rulechef[all]      # Everything
```

## Quick Start

### Extraction

Extract answer spans from text:

```python
from openai import OpenAI
from rulechef import RuleChef, Task, TaskType

client = OpenAI()
task = Task(
    name="Q&A Extraction",
    description="Extract answer spans from context",
    input_schema={"question": "str", "context": "str"},
    output_schema={"spans": "List[Span]"},
    type=TaskType.EXTRACTION,
)

chef = RuleChef(task, client)

chef.add_example(
    {"question": "When?", "context": "Built in 1991"},
    {"spans": [{"text": "1991", "start": 9, "end": 13}]}
)
chef.add_example(
    {"question": "When?", "context": "Released in 2005"},
    {"spans": [{"text": "2005", "start": 12, "end": 16}]}
)

chef.learn_rules()

result = chef.extract({"question": "When?", "context": "Founded in 1997"})
print(result)  # {"spans": [{"text": "1997", ...}]}
```

### Named Entity Recognition (NER)

```python
from pydantic import BaseModel
from typing import List, Literal

class Entity(BaseModel):
    text: str
    start: int
    end: int
    type: Literal["DRUG", "DOSAGE", "CONDITION"]

class NEROutput(BaseModel):
    entities: List[Entity]

task = Task(
    name="Medical NER",
    description="Extract drugs, dosages, and conditions",
    input_schema={"text": "str"},
    output_schema=NEROutput,
    type=TaskType.NER,
)

chef = RuleChef(task, client)
chef.add_example(
    {"text": "Take Aspirin 500mg for headache"},
    {"entities": [
        {"text": "Aspirin", "start": 5, "end": 12, "type": "DRUG"},
        {"text": "500mg", "start": 13, "end": 18, "type": "DOSAGE"},
        {"text": "headache", "start": 23, "end": 31, "type": "CONDITION"},
    ]}
)
chef.learn_rules()
```

### Classification

```python
task = Task(
    name="Intent Classification",
    description="Classify banking customer queries",
    input_schema={"text": "str"},
    output_schema={"label": "str"},
    type=TaskType.CLASSIFICATION,
    text_field="text",
)

chef = RuleChef(task, client)
chef.add_example({"text": "what is the exchange rate?"}, {"label": "exchange_rate"})
chef.add_example({"text": "I want to know the rates"}, {"label": "exchange_rate"})
chef.add_example({"text": "my card hasn't arrived"}, {"label": "card_arrival"})

chef.learn_rules()
result = chef.extract({"text": "current exchange rate please"})
print(result)  # {"label": "exchange_rate"}
```

### Transformation

```python
task = Task(
    name="Invoice Parser",
    description="Extract company and amount from invoices",
    input_schema={"text": "str"},
    output_schema={"company": "str", "amount": "str"},
    type=TaskType.TRANSFORMATION,
)

chef = RuleChef(task, client)
chef.add_example(
    {"text": "Invoice from Acme Corp for $1,500.00"},
    {"company": "Acme Corp", "amount": "$1,500.00"}
)
chef.learn_rules()
```

## Core Concepts

### Task Types

| Type | Output | Use Case |
|------|--------|----------|
| `EXTRACTION` | `{"spans": [...]}` | Find text spans (untyped) |
| `NER` | `{"entities": [...]}` | Find typed entities with labels |
| `CLASSIFICATION` | `{"label": "..."}` | Classify text into categories |
| `TRANSFORMATION` | Custom dict | Extract structured fields |

### Rule Formats

| Format | Best For | Example |
|--------|----------|---------|
| `RuleFormat.REGEX` | Keyword patterns, structured text | `\b\d{4}\b` |
| `RuleFormat.CODE` | Complex logic, multi-field extraction | `def extract(input_data): ...` |
| `RuleFormat.SPACY` | Linguistic patterns, POS/dependency | `[{"POS": "PROPN", "OP": "+"}]` |

```python
from rulechef import RuleFormat

# Only generate regex rules (fastest, most portable)
chef = RuleChef(task, client, allowed_formats=[RuleFormat.REGEX])

# Only code rules (most flexible)
chef = RuleChef(task, client, allowed_formats=[RuleFormat.CODE])
```

### Buffer-First Architecture

Examples go to a buffer first, then get committed to the dataset during `learn_rules()`. This enables batch learning and coordinator-driven decisions:

```python
chef.add_example(input1, output1)   # Goes to buffer
chef.add_example(input2, output2)   # Goes to buffer
chef.add_correction(input3, wrong_output, correct_output)  # High-priority signal

chef.learn_rules()  # Buffer -> Dataset -> Synthesis -> Refinement
```

### Corrections & Feedback

Corrections are the highest-value training signal -- they show exactly where the current rules fail:

```python
result = chef.extract({"text": "some input"})
# Result was wrong! Correct it:
chef.add_correction(
    {"text": "some input"},
    model_output=result,
    expected_output={"label": "correct_label"},
    feedback="The rule matched too broadly"
)

# Task-level guidance
chef.add_feedback("Drug names always follow 'take' or 'prescribe'")

# Rule-level feedback
chef.add_feedback("This rule is too broad", level="rule", target_id="rule_id")

chef.learn_rules()  # Re-learns with corrections prioritized
```

## Evaluation

RuleChef includes built-in evaluation with entity-level precision, recall, and F1:

```python
# Dataset-level evaluation
eval_result = chef.evaluate()
# Prints: Exact match, micro/macro P/R/F1, per-class breakdown

# Per-rule evaluation (find dead or harmful rules)
metrics = chef.get_rule_metrics()
# Shows: per-rule TP/FP/FN, sample matches, identifies dead rules

# Delete a bad rule
chef.delete_rule("rule_id")
```

## Advanced Features

### Synthesis Strategy

For multi-class tasks, RuleChef can synthesize rules one class at a time for better coverage:

```python
# Auto-detect (default): per-class if >1 class, bulk otherwise
chef = RuleChef(task, client, synthesis_strategy="auto")

# Force per-class synthesis
chef = RuleChef(task, client, synthesis_strategy="per_class")

# Force single-prompt bulk synthesis
chef = RuleChef(task, client, synthesis_strategy="bulk")
```

### Agentic Coordinator

The `AgenticCoordinator` uses LLM calls to guide the refinement loop, focusing on weak classes:

```python
from rulechef import RuleChef, AgenticCoordinator

coordinator = AgenticCoordinator(client, model="gpt-4o-mini")
chef = RuleChef(task, client, coordinator=coordinator)

chef.learn_rules(max_refinement_iterations=10)
# Coordinator analyzes per-class metrics each iteration,
# tells the synthesis prompt which classes to focus on,
# and stops early when performance plateaus.
```

### Rule Pruning

With `prune_after_learn=True`, the agentic coordinator audits rules after learning -- merging redundant rules and removing pure noise. A safety net reverts if F1 drops:

```python
coordinator = AgenticCoordinator(client, prune_after_learn=True)
chef = RuleChef(task, client, coordinator=coordinator)

chef.learn_rules()
# After synthesis+refinement:
# 1. LLM analyzes rules + per-rule metrics
# 2. Merges similar patterns (e.g. two regexes → one)
# 3. Removes precision=0 rules (pure false positives)
# 4. Re-evaluates — reverts if F1 drops
```

In the CLI: `learn --agentic --prune`.

### Incremental Patching

After the initial learn, you can patch existing rules without full re-synthesis:

```python
chef.learn_rules()           # Initial synthesis
chef.add_correction(...)     # Add corrections
chef.learn_rules(incremental_only=True)  # Patch, don't re-synthesize
```

### Observation Mode

Collect training data from any LLM -- no task definition needed:

```python
# Works with any LLM provider (Anthropic, Groq, local models, etc.)
chef = RuleChef(client=client, model="gpt-4o-mini")  # No task needed
chef.add_observation({"text": "what's the exchange rate?"}, {"label": "exchange_rate"})
chef.learn_rules()  # Auto-discovers the task schema
```

For raw LLM interactions where you don't know the schema:

```python
chef.add_raw_observation(
    messages=[{"role": "user", "content": "classify: what's the rate?"}],
    response="exchange_rate",
)
chef.learn_rules()  # Discovers task + maps observations + learns rules
```

For OpenAI-compatible clients, auto-capture with monkey-patching:

```python
wrapped = chef.start_observing(openai_client, auto_learn=False)
response = wrapped.chat.completions.create(...)  # Observed automatically
chef.learn_rules()
chef.stop_observing()
```

### Pydantic Output Schemas

Use Pydantic models for type-safe, validated outputs with automatic label extraction:

```python
from pydantic import BaseModel
from typing import List, Literal

class Entity(BaseModel):
    text: str
    start: int
    end: int
    type: Literal["PERSON", "ORG", "LOCATION"]

class Output(BaseModel):
    entities: List[Entity]

task = Task(..., output_schema=Output, type=TaskType.NER)
# RuleChef automatically discovers labels: ["PERSON", "ORG", "LOCATION"]
```

### grex: Regex Pattern Suggestions

When `use_grex=True` (default), [grex](https://github.com/pemistahl/grex) analyzes your training examples and adds regex pattern hints to the synthesis prompt. The LLM sees concrete patterns alongside the raw examples, producing better rules — especially for structured data like dates, IDs, and amounts:

```
DATA EVIDENCE FROM TRAINING:
- DATE (5 unique): "2024-01-15", "2024-02-28", "2023-12-01", ...
  Exact pattern: (2023\-12\-01|2024\-01\-15|2024\-02\-28|...)
  Structural pattern: \d{4}\-\d{2}\-\d{2}
```

Install with `pip install rulechef[grex]`. Disable with `use_grex=False`.

## Benchmark: Banking77

On the [Banking77](https://huggingface.co/datasets/legacy-datasets/banking77) intent classification dataset (5-class subset, 5-shot per class, regex-only):

| Metric | Value |
|--------|-------|
| Accuracy | 60.5% |
| Micro Precision | 100% |
| Macro F1 | 71.7% |
| Rules learned | 108 |
| Per-query latency | 0.19ms |

With agentic coordinator guiding 15 refinement iterations against a dev set. Zero false positives — rules never give a wrong answer, they just abstain when unsure. Full results and learned rules: [`benchmarks/results_banking77.json`](benchmarks/results_banking77.json). Reproduce: `python benchmarks/benchmark_banking77.py --classes beneficiary_not_allowed,card_arrival,disposable_card_limits,exchange_rate,pending_cash_withdrawal --shots 5 --max-iterations 15 --agentic`.

## CLI

Interactive CLI for quick experimentation across all task types:

```bash
export OPENAI_API_KEY=your_key
rulechef
```

The CLI walks you through a setup wizard (task name, type, labels, model, base URL) and drops you into a command loop:

```
Commands:
  add        Add a training example
  correct    Add a correction
  extract    Run extraction on input
  learn      Learn rules (--iterations N, --incremental, --agentic, --prune)
  evaluate   Evaluate rules against dataset
  rules      List learned rules (rules <id> for detail)
  delete     Delete a rule by ID
  feedback   Add feedback (task/rule level)
  generate   Generate synthetic examples with LLM
  stats      Show dataset statistics
  help       Show commands
  quit       Exit
```

Works with any OpenAI-compatible API (Groq, Together, Ollama, etc.) via the base URL prompt.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
