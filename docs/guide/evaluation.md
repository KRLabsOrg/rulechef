# Evaluation & Feedback

RuleChef includes built-in evaluation with entity-level precision, recall, and F1 metrics.

## Dataset Evaluation

```python
eval_result = chef.evaluate()
```

This runs all rules against the dataset and computes:

- **Exact match accuracy** — percentage of inputs where output matches exactly
- **Micro precision/recall/F1** — aggregated across all predictions
- **Macro F1** — averaged across classes (for multi-class tasks)
- **Per-class breakdown** — precision, recall, F1 per label

## Per-Rule Evaluation

Find which rules are helping and which are hurting:

```python
metrics = chef.get_rule_metrics()
```

Each rule gets individual metrics:

- **TP / FP / FN** — true positives, false positives, false negatives
- **Sample matches** — which examples each rule matched
- **Dead rules** — rules that never fire on any example

```python
# Delete a rule that's causing false positives
chef.delete_rule("rule_id")
```

## Corrections

Corrections are the highest-value training signal. They show exactly where current rules fail:

```python
result = chef.extract({"text": "some input"})

# Result was wrong — correct it
chef.add_correction(
    {"text": "some input"},
    model_output=result,
    expected_output={"label": "correct_label"},
    feedback="The rule matched too broadly"
)

chef.learn_rules()  # Re-learns with corrections prioritized
```

## Feedback

Feedback provides guidance at different levels:

### Task-Level Feedback

General guidance for the entire task:

```python
chef.add_feedback("Drug names always follow 'take' or 'prescribe'")
chef.add_feedback("Ignore mentions in parentheses")
```

### Rule-Level Feedback

Guidance targeted at a specific rule:

```python
chef.add_feedback(
    "This rule is too broad — it matches common words",
    level="rule",
    target_id="rule_123"
)
```

Feedback is included in synthesis prompts during the next `learn_rules()` call.

## Matching Modes

### Extraction

```python
task = Task(
    ...,
    type=TaskType.EXTRACTION,
    matching_mode="text",   # Compare span text only (default)
)

task = Task(
    ...,
    type=TaskType.EXTRACTION,
    matching_mode="exact",  # Compare text + start/end offsets
)
```

### NER

Entity matching checks both text and type. Entities match if they have the same text and entity type.

### Classification

Label matching is case-insensitive and strips whitespace.

### Transformation

Dict matching compares values recursively. Array elements are matched order-independently.

## Custom Matchers

Override the default matching logic:

```python
def my_matcher(expected, actual):
    # Custom comparison logic
    return expected["label"].lower() == actual["label"].lower()

task = Task(
    ...,
    output_matcher=my_matcher,
)
```

## Stats

Get a summary of the current state:

```python
stats = chef.get_stats()
# {
#   "dataset_name": "default",
#   "total_examples": 25,
#   "total_corrections": 3,
#   "total_rules": 12,
#   "buffer_stats": {...},
# }

summary = chef.get_rules_summary()
# Human-readable summary of all rules
```
