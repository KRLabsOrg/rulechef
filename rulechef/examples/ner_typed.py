"""
Typed NER Example - Extract and classify named entities

Demonstrates the new schema-aware rules feature:
- Rules have output_template that maps matches to typed entities
- Evaluation uses typed metrics (entity must match both boundary AND type)
"""

import os
from openai import OpenAI
from rulechef import RuleChef, Task, TaskType
from rulechef.coordinator import SimpleCoordinator
from rulechef.evaluation import evaluate_typed_spans, print_typed_eval_report

# =============================================================================
# Define Typed NER Task
# =============================================================================

task = Task(
    name="Named Entity Recognition",
    description="Extract and classify named entities (PER, ORG, LOC) from text",
    input_schema={"text": "str"},
    output_schema={"entities": "List[{text: str, start: int, end: int, type: str}]"},
    type=TaskType.NER,  # Use the new NER task type
)

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("⚠️  OPENAI_API_KEY not set - running in demo mode (no rule learning)")
    print("Set it with: export OPENAI_API_KEY='your-key'")
    chef = RuleChef(
        task,
        dataset_name="ner_typed_demo",
        allowed_formats=["regex", "spacy"],
    )
else:
    client = OpenAI(
        api_key=api_key,
    )
    coordinator = SimpleCoordinator(trigger_threshold=10, verbose=False)
    chef = RuleChef(
        task,
        client=client,
        dataset_name="ner_typed",
        coordinator=coordinator,
        auto_trigger=False,
        allowed_formats=["regex", "spacy"],
        model="gpt-5.1",
    )

# =============================================================================
# Training Data - Typed Entities
# =============================================================================

training_examples = [
    {
        "text": "Apple Inc. announced its new iPhone at an event in Cupertino, California.",
        "entities": [
            {"text": "Apple Inc.", "start": 0, "end": 10, "type": "ORG"},
            {"text": "Cupertino", "start": 51, "end": 60, "type": "LOC"},
            {"text": "California", "start": 62, "end": 72, "type": "LOC"},
        ],
    },
    {
        "text": "Microsoft CEO Satya Nadella discussed partnerships at the Seattle headquarters.",
        "entities": [
            {"text": "Microsoft", "start": 0, "end": 9, "type": "ORG"},
            {"text": "Satya Nadella", "start": 14, "end": 27, "type": "PER"},
            {"text": "Seattle", "start": 58, "end": 65, "type": "LOC"},
        ],
    },
    {
        "text": "Google LLC and OpenAI signed a major collaboration deal in San Francisco.",
        "entities": [
            {"text": "Google LLC", "start": 0, "end": 10, "type": "ORG"},
            {"text": "OpenAI", "start": 15, "end": 21, "type": "ORG"},
            {"text": "San Francisco", "start": 59, "end": 72, "type": "LOC"},
        ],
    },
    {
        "text": "Tesla Motors released the Model 3 with advanced features. Elon Musk announced it.",
        "entities": [
            {"text": "Tesla Motors", "start": 0, "end": 12, "type": "ORG"},
            {"text": "Elon Musk", "start": 58, "end": 67, "type": "PER"},
        ],
    },
    {
        "text": "Amazon Web Services expanded cloud infrastructure in Tokyo and Singapore.",
        "entities": [
            {"text": "Amazon Web Services", "start": 0, "end": 19, "type": "ORG"},
            {"text": "Tokyo", "start": 52, "end": 57, "type": "LOC"},
            {"text": "Singapore", "start": 62, "end": 71, "type": "LOC"},
        ],
    },
    {
        "text": "Jeff Bezos founded Amazon in Seattle, Washington.",
        "entities": [
            {"text": "Jeff Bezos", "start": 0, "end": 10, "type": "PER"},
            {"text": "Amazon", "start": 19, "end": 25, "type": "ORG"},
            {"text": "Seattle", "start": 29, "end": 36, "type": "LOC"},
            {"text": "Washington", "start": 38, "end": 48, "type": "LOC"},
        ],
    },
    {
        "text": "Mark Zuckerberg leads Meta Platforms from Menlo Park, California.",
        "entities": [
            {"text": "Mark Zuckerberg", "start": 0, "end": 15, "type": "PER"},
            {"text": "Meta Platforms", "start": 22, "end": 36, "type": "ORG"},
            {"text": "Menlo Park", "start": 42, "end": 52, "type": "LOC"},
            {"text": "California", "start": 54, "end": 64, "type": "LOC"},
        ],
    },
    {
        "text": "IBM partnered with Red Hat on cloud solutions from Boston headquarters.",
        "entities": [
            {"text": "IBM", "start": 0, "end": 3, "type": "ORG"},
            {"text": "Red Hat", "start": 18, "end": 25, "type": "ORG"},
            {"text": "Boston", "start": 50, "end": 56, "type": "LOC"},
        ],
    },
    {
        "text": "Tim Cook is the CEO of Apple Inc. based in Cupertino.",
        "entities": [
            {"text": "Tim Cook", "start": 0, "end": 8, "type": "PER"},
            {"text": "Apple Inc.", "start": 23, "end": 33, "type": "ORG"},
            {"text": "Cupertino", "start": 43, "end": 52, "type": "LOC"},
        ],
    },
    {
        "text": "Sundar Pichai runs Google from Mountain View, California.",
        "entities": [
            {"text": "Sundar Pichai", "start": 0, "end": 13, "type": "PER"},
            {"text": "Google", "start": 19, "end": 25, "type": "ORG"},
            {"text": "Mountain View", "start": 31, "end": 44, "type": "LOC"},
            {"text": "California", "start": 46, "end": 56, "type": "LOC"},
        ],
    },
]

# Test examples (held out)
test_examples = [
    {
        "text": "Jensen Huang is the CEO of Nvidia Corporation in Santa Clara.",
        "entities": [
            {"text": "Jensen Huang", "start": 0, "end": 12, "type": "PER"},
            {"text": "Nvidia Corporation", "start": 27, "end": 45, "type": "ORG"},
            {"text": "Santa Clara", "start": 49, "end": 60, "type": "LOC"},
        ],
    },
    {
        "text": "Salesforce acquired Slack Technologies in San Francisco.",
        "entities": [
            {"text": "Salesforce", "start": 0, "end": 10, "type": "ORG"},
            {"text": "Slack Technologies", "start": 20, "end": 38, "type": "ORG"},
            {"text": "San Francisco", "start": 42, "end": 55, "type": "LOC"},
        ],
    },
    {
        "text": "Satya Nadella announced Microsoft's AI strategy in Redmond.",
        "entities": [
            {"text": "Satya Nadella", "start": 0, "end": 13, "type": "PER"},
            {"text": "Microsoft", "start": 24, "end": 33, "type": "ORG"},
            {"text": "Redmond", "start": 51, "end": 58, "type": "LOC"},
        ],
    },
    {
        "text": "Warren Buffett leads Berkshire Hathaway from Omaha, Nebraska.",
        "entities": [
            {"text": "Warren Buffett", "start": 0, "end": 14, "type": "PER"},
            {"text": "Berkshire Hathaway", "start": 21, "end": 39, "type": "ORG"},
            {"text": "Omaha", "start": 45, "end": 50, "type": "LOC"},
            {"text": "Nebraska", "start": 52, "end": 60, "type": "LOC"},
        ],
    },
    {
        "text": "Adobe Systems has offices in San Jose and New York City.",
        "entities": [
            {"text": "Adobe Systems", "start": 0, "end": 13, "type": "ORG"},
            {"text": "San Jose", "start": 29, "end": 37, "type": "LOC"},
            {"text": "New York City", "start": 42, "end": 55, "type": "LOC"},
        ],
    },
]

print("=" * 80)
print("TYPED NER EXAMPLE")
print("=" * 80)

print(f"\n📊 Dataset: {len(training_examples)} training, {len(test_examples)} test")
print("   Entity types: PER, ORG, LOC")

# =============================================================================
# Add Training Examples
# =============================================================================

print(f"\n📥 Adding {len(training_examples)} training examples...")
for example in training_examples:
    chef.add_example(
        input_data={"text": example["text"]},
        output_data={"entities": example["entities"]},
    )

print(f"✓ Buffer stats: {chef.buffer.get_stats()}")

# =============================================================================
# Learn Rules
# =============================================================================

if api_key:
    print("\n🤖 Learning typed NER rules...")
    chef.learn_rules(run_evaluation=False, max_refinement_iterations=2)

    # Show learned rules
    if chef.dataset.rules:
        print(f"\n✓ Learned {len(chef.dataset.rules)} rules:")
        for rule in chef.dataset.rules[:5]:
            print(f"  - {rule.name}")
            print(f"    Format: {rule.format.value}")
            if rule.output_template:
                print(f"    Template: {rule.output_template}")
            if rule.output_key:
                print(f"    Output key: {rule.output_key}")
else:
    print("\n⚠️  Skipping rule learning (no API key)")

# =============================================================================
# Evaluate on Test Set
# =============================================================================

print(f"\n📋 Evaluating on {len(test_examples)} test examples...")

all_predictions = []
all_gold = []

for example in test_examples:
    text = example["text"]
    gold = example["entities"]

    # Extract using learned rules
    result = chef.extract({"text": text})
    predictions = result.get("entities", [])

    all_predictions.extend(predictions)
    all_gold.extend(gold)

    # Show individual results
    print(f"\nText: {text[:60]}...")
    print(f"  Gold:      {[e['text'] + '/' + e['type'] for e in gold]}")
    print(f"  Predicted: {[e['text'] + '/' + e.get('type', '?') for e in predictions]}")

# =============================================================================
# Typed Evaluation
# =============================================================================

if all_predictions or all_gold:
    metrics = evaluate_typed_spans(all_predictions, all_gold)
    print_typed_eval_report(metrics, "Typed NER Test Set")
else:
    print("\n⚠️  No predictions to evaluate (need rules)")

print("=" * 80)
print("✓ Done!")
print("=" * 80)
