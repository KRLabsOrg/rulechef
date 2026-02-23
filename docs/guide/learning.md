# Learning & Refinement

RuleChef uses a buffer-first architecture: examples are collected, then committed to a dataset and used for rule synthesis during `learn_rules()`.

## Buffer-First Architecture

```mermaid
flowchart TD
    A[add_example / add_correction] --> B[Buffer]
    B --> C[learn_rules]
    C --> D[Buffer → Dataset]
    D --> E[Rule Synthesis]
    E --> F[Evaluation & Refinement]
    F --> G[Persisted Rules]
```

```python
chef.add_example(input1, output1)   # Goes to buffer
chef.add_example(input2, output2)   # Goes to buffer
chef.add_correction(input3, wrong, correct)  # High-priority signal

chef.learn_rules()  # Buffer → Dataset → Synthesis → Refinement
```

## learn_rules()

The main learning method orchestrates the full pipeline:

```python
rules, eval_result = chef.learn_rules(
    max_refinement_iterations=3,  # Evaluation-refinement cycles
    incremental_only=False,       # If True, only patch (don't re-synthesize)
)
```

**What happens during `learn_rules()`:**

1. Buffer examples are committed to the dataset
2. Rules are synthesized using the LLM
3. Rules are evaluated against the dataset
4. Failed examples drive refinement iterations
5. Rules are persisted to disk

**Returns:** A tuple of `(List[Rule], Optional[EvalResult])`.

## Synthesis Strategies

For multi-class tasks (NER, classification), RuleChef can synthesize rules per-class for better coverage:

```python
# Auto-detect: per-class if >1 class, bulk otherwise (default)
chef = RuleChef(task, client, synthesis_strategy="auto")

# Force per-class synthesis
chef = RuleChef(task, client, synthesis_strategy="per_class")

# Force single-prompt bulk synthesis
chef = RuleChef(task, client, synthesis_strategy="bulk")
```

Per-class synthesis generates rules for each label separately — one LLM call per class. Each call receives:

- **Positive examples** for that class (capped to `max_samples`, sampled using `sampling_strategy`)
- **Counter-examples** from other classes (capped to `max_counter_examples`) to prevent false positives
- Up to `max_rules_per_class` rules are generated per call

For a 5-class classification task with default settings, this means 5 LLM calls, each producing up to 5 rules, with up to 50 positive examples and 10 counter-examples per prompt.

## Prompt Size Controls

These parameters control how much data goes into each LLM prompt:

```python
chef = RuleChef(task, client,
    max_samples=50,             # Max examples per prompt (per-class positives + patch failures)
    max_rules_per_class=5,      # Max rules generated per class in per-class synthesis
    max_counter_examples=10,    # Max negative examples per class prompt
    max_rules=10,               # Max rules per bulk synthesis call
)
```

| Parameter | Default | Applies to |
|-----------|---------|------------|
| `max_samples` | 50 | Per-class positive examples and patch failure sampling |
| `max_rules_per_class` | 5 | Per-class synthesis (rules generated per class) |
| `max_counter_examples` | 10 | Per-class synthesis (negative examples per prompt) |
| `max_rules` | 10 | Bulk synthesis (total rules per call) |

When a class has more examples than `max_samples`, examples are sampled using the configured `sampling_strategy`.

## Sampling Strategies

Control how training data is selected when there are more examples than `max_samples`:

```python
chef = RuleChef(task, client, sampling_strategy="balanced")
```

| Strategy | Behavior |
|----------|----------|
| `balanced` | Takes examples in order (default) |
| `recent` | Favor recently added examples |
| `diversity` | Evenly space picks across the dataset |
| `uncertain` | Low-confidence examples first |
| `varied` | Mix of recent, diverse, and uncertain |
| `corrections_first` | Corrections first, then recent examples |

All strategies always include corrections first — they're the highest-value signal.

## Incremental Patching

After the initial learn, you can patch rules for specific failures without re-synthesizing everything:

```python
# Initial synthesis
chef.learn_rules()

# Add corrections for failures
chef.add_correction(
    {"text": "some input"},
    model_output={"label": "wrong"},
    expected_output={"label": "correct"},
)

# Patch existing rules (don't re-synthesize)
chef.learn_rules(incremental_only=True)
```

Incremental patching:

- Generates targeted rules for known failures
- Merges new rules into the existing ruleset
- Prunes weak rules that don't contribute
- Preserves stable rules that are working

## Persistence

Rules and datasets are automatically saved to disk:

```python
chef = RuleChef(task, client,
    dataset_name="my_project",       # Filename for the dataset
    storage_path="./rulechef_data",  # Directory for JSON files
)
```

Files saved:

- `{storage_path}/{dataset_name}.json` — dataset with examples, corrections, rules

Loading happens automatically when a matching file exists at the storage path.
