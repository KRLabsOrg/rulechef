"""Offline demo: Evaluation and Per-Rule Metrics

No API key needed — manually creates rules and evaluates them.

Usage:
    python examples/evaluation_offline.py
"""

from rulechef.core import (
    Correction,
    Dataset,
    Example,
    Rule,
    RuleFormat,
    Task,
    TaskType,
)
from rulechef.evaluation import (
    evaluate_dataset,
    evaluate_rules_individually,
    print_eval_result,
    print_rule_metrics,
)
from rulechef.executor import RuleExecutor

# ── Build a small NER dataset ──────────────────────────────────────────────

task = Task(
    name="Medical NER",
    description="Extract drugs, dosages, frequencies, and conditions",
    input_schema={"text": "str"},
    output_schema={
        "entities": "List[{text: str, start: int, end: int, type: DRUG|DOSAGE|FREQUENCY|CONDITION}]"
    },
    type=TaskType.NER,
)

dataset = Dataset(name="medical_demo", task=task)

examples_raw = [
    {
        "text": "Metformin 500mg twice daily for type 2 diabetes.",
        "entities": [
            {"text": "Metformin", "start": 0, "end": 9, "type": "DRUG"},
            {"text": "500mg", "start": 10, "end": 15, "type": "DOSAGE"},
            {"text": "twice daily", "start": 16, "end": 27, "type": "FREQUENCY"},
            {"text": "type 2 diabetes", "start": 32, "end": 47, "type": "CONDITION"},
        ],
    },
    {
        "text": "Lisinopril 10mg once daily for hypertension.",
        "entities": [
            {"text": "Lisinopril", "start": 0, "end": 10, "type": "DRUG"},
            {"text": "10mg", "start": 11, "end": 15, "type": "DOSAGE"},
            {"text": "once daily", "start": 16, "end": 26, "type": "FREQUENCY"},
            {"text": "hypertension", "start": 31, "end": 43, "type": "CONDITION"},
        ],
    },
    {
        "text": "Ibuprofen 400mg every 6 hours for migraine.",
        "entities": [
            {"text": "Ibuprofen", "start": 0, "end": 9, "type": "DRUG"},
            {"text": "400mg", "start": 10, "end": 15, "type": "DOSAGE"},
            {"text": "every 6 hours", "start": 16, "end": 29, "type": "FREQUENCY"},
            {"text": "migraine", "start": 34, "end": 42, "type": "CONDITION"},
        ],
    },
    {
        "text": "Gabapentin 300mg three times daily for neuropathic pain.",
        "entities": [
            {"text": "Gabapentin", "start": 0, "end": 10, "type": "DRUG"},
            {"text": "300mg", "start": 11, "end": 16, "type": "DOSAGE"},
            {"text": "three times daily", "start": 17, "end": 34, "type": "FREQUENCY"},
            {"text": "neuropathic pain", "start": 39, "end": 55, "type": "CONDITION"},
        ],
    },
    {
        "text": "Aspirin 81mg daily for cardiac prevention.",
        "entities": [
            {"text": "Aspirin", "start": 0, "end": 7, "type": "DRUG"},
            {"text": "81mg", "start": 8, "end": 12, "type": "DOSAGE"},
            {"text": "daily", "start": 13, "end": 18, "type": "FREQUENCY"},
            {"text": "cardiac prevention", "start": 23, "end": 41, "type": "CONDITION"},
        ],
    },
]

for i, ex in enumerate(examples_raw):
    dataset.examples.append(
        Example(
            id=f"ex_{i}",
            input={"text": ex["text"]},
            expected_output={"entities": ex["entities"]},
            source="human_labeled",
        )
    )

dataset.corrections.append(
    Correction(
        id="corr_1",
        input={"text": "Prednisone 60mg once daily for asthma."},
        model_output={
            "entities": [
                {"text": "Prednisone", "start": 0, "end": 10, "type": "DRUG"},
                {"text": "60mg", "start": 11, "end": 15, "type": "DOSAGE"},
            ]
        },
        expected_output={
            "entities": [
                {"text": "Prednisone", "start": 0, "end": 10, "type": "DRUG"},
                {"text": "60mg", "start": 11, "end": 15, "type": "DOSAGE"},
                {"text": "once daily", "start": 16, "end": 26, "type": "FREQUENCY"},
                {"text": "asthma", "start": 31, "end": 37, "type": "CONDITION"},
            ]
        },
        feedback="Model missed the frequency and condition entirely",
    )
)

print(f"Dataset: {len(dataset.examples)} examples, {len(dataset.corrections)} corrections")

# ── Hand-crafted rules (good, decent, broad, and dead) ────────────────────

rules = [
    Rule(
        id="rule_dosage",
        name="dosage_pattern",
        description="Match dosage amounts like 500mg, 10mg",
        format=RuleFormat.REGEX,
        content=r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|units)\b",
        output_template={
            "text": "$0",
            "start": "$start",
            "end": "$end",
            "type": "DOSAGE",
        },
        output_key="entities",
        priority=8,
    ),
    Rule(
        id="rule_drug",
        name="drug_before_dosage",
        description="Match capitalized word before a dosage",
        format=RuleFormat.REGEX,
        content=r"\b([A-Z][a-z]+)\s+(?=\d+(?:\.\d+)?\s*(?:mg|mcg|units))",
        output_template={
            "text": "$0",
            "start": "$start",
            "end": "$end",
            "type": "DRUG",
        },
        output_key="entities",
        priority=7,
    ),
    Rule(
        id="rule_freq",
        name="frequency_pattern",
        description="Match frequency phrases like 'once daily', 'twice daily'",
        format=RuleFormat.REGEX,
        content=r"\b(?:once|twice|three times|every \d+ hours)\s*(?:daily|a day)?\b",
        output_template={
            "text": "$0",
            "start": "$start",
            "end": "$end",
            "type": "FREQUENCY",
        },
        output_key="entities",
        priority=6,
    ),
    Rule(
        id="rule_condition_broad",
        name="condition_after_for",
        description="Match everything after 'for' as condition (too broad!)",
        format=RuleFormat.REGEX,
        content=r"\bfor\s+(\w+)",
        output_template={
            "text": "$1",
            "start": "$start",
            "end": "$end",
            "type": "CONDITION",
        },
        output_key="entities",
        priority=5,
    ),
    Rule(
        id="rule_dead",
        name="iv_route",
        description="Match IV administration routes (never fires on this data)",
        format=RuleFormat.REGEX,
        content=r"\b(?:IV|intravenous|subcutaneous)\b",
        output_template={
            "text": "$0",
            "start": "$start",
            "end": "$end",
            "type": "ROUTE",
        },
        output_key="entities",
        priority=3,
    ),
]

dataset.rules = rules

# ── Executor ──────────────────────────────────────────────────────────────

executor = RuleExecutor()


def apply_rules_fn(rules, input_data, task_type, text_field):
    return executor.apply_rules(rules, input_data, task_type, text_field)


# ── 1. Dataset-level evaluation ───────────────────────────────────────────

print("\n" + "=" * 70)
print("1. Entity-level evaluation with per-class P/R/F1")
print("=" * 70)

eval_result = evaluate_dataset(rules, dataset, apply_rules_fn)
print_eval_result(eval_result, "Medical NER")

print(f"\n  {len(eval_result.failures)} document(s) with errors:")
for f in eval_result.failures[:3]:
    text = f["input"]["text"][:60]
    is_corr = f["is_correction"]
    print(f"    {'[CORRECTION] ' if is_corr else ''}{text}...")

# ── 2. Per-rule metrics ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("2. Per-rule metrics — find good, weak, and dead rules")
print("=" * 70)

rule_metrics = evaluate_rules_individually(rules, dataset, apply_rules_fn)
print_rule_metrics(rule_metrics)

print("Actionable insights:")
for rm in rule_metrics:
    if rm.f1 == 0 and rm.matches == 0:
        print(f"  DEAD: '{rm.rule_name}' — never fires, safe to delete")
    elif rm.precision < 0.5:
        print(f"  TOO BROAD: '{rm.rule_name}' — precision {rm.precision:.0%}")
    elif rm.precision >= 0.8 and rm.recall >= 0.5:
        print(f"  GOOD: '{rm.rule_name}' — P={rm.precision:.0%} R={rm.recall:.0%}")
    elif rm.precision >= 0.8:
        print(f"  NARROW: '{rm.rule_name}' — P={rm.precision:.0%} R={rm.recall:.0%}")

# ── 3. Delete bad rules, re-evaluate ─────────────────────────────────────

print("\n" + "=" * 70)
print("3. Clean up bad rules and re-evaluate")
print("=" * 70)

dead_rules = [rm for rm in rule_metrics if rm.matches == 0]
for rm in dead_rules:
    print(f"  Deleting dead rule: '{rm.rule_name}'")
    dataset.rules = [r for r in dataset.rules if r.id != rm.rule_id]

eval_after = evaluate_dataset(dataset.rules, dataset, apply_rules_fn)
print(f"\n  Before cleanup: micro_f1={eval_result.micro_f1:.3f}, {eval_result.total_fp} FP")
print(f"  After cleanup:  micro_f1={eval_after.micro_f1:.3f}, {eval_after.total_fp} FP")

print("\nDone!")
