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

## Rule Trust and Conflict Resolution

When learning completes, every rule is evaluated in isolation and stamped with a **validated precision** and **support** (the number of predictions behind the estimate). When a holdout is active these are measured on the dev split — data the rule was never tuned on.

```python
rule.validated_precision   # e.g. 0.86
rule.validated_support     # e.g. 22 predictions
```

For ranking and routing, precision is discounted by a **Wilson lower bound**, so a rule that was right 2/2 does not outrank one that was right 95/100. A rule that memorized a training lexicon flags itself this way: it transfers poorly to dev and ends up with a conspicuously low validated precision — you can spot it without reading the pattern.

### Conflict resolution

When several rules produce overlapping or conflicting matches, the executor orders them deterministically by **priority, then validated precision** (falling back to confidence for rules with no validated estimate). The higher-trust rule wins.

### Ranking and pruning

`rank_rules()` reports each rule's solo F1 and its leave-one-out marginal contribution to the ensemble, and can prune rules whose removal improves overall F1:

```python
ranked = chef.rank_rules(holdout_fraction=0.2)   # measured on a held-out split
```

See the [Ranking API](../api/ranking.md) for the full report structure.

## Repairing Rules with Feedback

Because rules are readable, their defects are too — and they can be fixed in plain English without re-synthesizing from scratch. Attach rule-level feedback and run one incremental round:

```python
chef.add_feedback(
    "Never match number/number patterns like '1432/03' — those are case numbers.",
    level="rule",
    target_id=quantity_rule.id,
)
chef.learn_rules(incremental_only=True, holdout_fraction=0.2)
```

The targeted rules are patched while untouched rules are preserved. Human-written and LLM-generated (critic) feedback flow through the same channel.

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
