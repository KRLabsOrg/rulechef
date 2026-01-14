"""
Comprehensive NER Demo - Extract organizations, people, and locations from news text.

This demonstrates RuleChef's value proposition:
1. Learn rules from a few examples
2. Rules replace expensive LLM calls
3. Fast, cheap extraction with learned rules
"""

import os
from openai import OpenAI
from rulechef import RuleChef, Task, TaskType
from rulechef.coordinator import SimpleCoordinator

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not set")
    print("Set it with: export OPENAI_API_KEY='your-key'")
    exit(1)

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1/")

# =============================================================================
# Define NER Task
# =============================================================================

task = Task(
    name="Named Entity Recognition",
    description="Extract organizations, people, and locations from news articles",
    input_schema={"text": "str"},
    output_schema={"entities": "List[{text: str, start: int, end: int, type: str}]"},
    type=TaskType.NER,
)

# Low threshold for demo
coordinator = SimpleCoordinator(
    trigger_threshold=10,  # Learn after 10 examples
    correction_threshold=3,
    verbose=True,
)

chef = RuleChef(
    task,
    client=client,
    dataset_name="ner_demo",
    coordinator=coordinator,
    auto_trigger=True,  # Automatically learn when ready
    allowed_formats=["regex"],  # Regex only for this demo
    model="moonshotai/kimi-k2-instruct-0905",  # Specify the model to use
)

# =============================================================================
# Training Data - Real news snippets
# =============================================================================

training_examples = [
    {
        "text": "Apple Inc. announced its new iPhone at an event in Cupertino, California. CEO Tim Cook presented the features to a packed audience.",
        "entities": [
            {"text": "Apple Inc.", "start": 0, "end": 10, "type": "ORG"},
            {"text": "iPhone", "start": 29, "end": 35, "type": "PRODUCT"},
            {"text": "Cupertino", "start": 52, "end": 61, "type": "LOC"},
            {"text": "California", "start": 63, "end": 73, "type": "LOC"},
            {"text": "Tim Cook", "start": 79, "end": 87, "type": "PERSON"},
        ],
    },
    {
        "text": "Microsoft Corporation partnered with OpenAI to integrate AI capabilities. Satya Nadella expressed enthusiasm about the collaboration at the Seattle headquarters.",
        "entities": [
            {"text": "Microsoft Corporation", "start": 0, "end": 21, "type": "ORG"},
            {"text": "OpenAI", "start": 37, "end": 43, "type": "ORG"},
            {"text": "Satya Nadella", "start": 75, "end": 88, "type": "PERSON"},
            {"text": "Seattle", "start": 141, "end": 148, "type": "LOC"},
        ],
    },
    {
        "text": "Tesla Motors, based in Austin, Texas, released its quarterly earnings. Elon Musk discussed the company's expansion plans in Europe and Asia.",
        "entities": [
            {"text": "Tesla Motors", "start": 0, "end": 12, "type": "ORG"},
            {"text": "Austin", "start": 23, "end": 29, "type": "LOC"},
            {"text": "Texas", "start": 31, "end": 36, "type": "LOC"},
            {"text": "Elon Musk", "start": 72, "end": 81, "type": "PERSON"},
            {"text": "Europe", "start": 127, "end": 133, "type": "LOC"},
            {"text": "Asia", "start": 138, "end": 142, "type": "LOC"},
        ],
    },
    {
        "text": "Google LLC launched a new data center in Dublin, Ireland. Sundar Pichai emphasized the company's commitment to renewable energy.",
        "entities": [
            {"text": "Google LLC", "start": 0, "end": 10, "type": "ORG"},
            {"text": "Dublin", "start": 41, "end": 47, "type": "LOC"},
            {"text": "Ireland", "start": 49, "end": 56, "type": "LOC"},
            {"text": "Sundar Pichai", "start": 58, "end": 71, "type": "PERSON"},
        ],
    },
    {
        "text": "Amazon Web Services expanded its cloud infrastructure in Singapore and Tokyo. Jeff Bezos praised the engineering team for their work.",
        "entities": [
            {"text": "Amazon Web Services", "start": 0, "end": 19, "type": "ORG"},
            {"text": "Singapore", "start": 57, "end": 66, "type": "LOC"},
            {"text": "Tokyo", "start": 71, "end": 76, "type": "LOC"},
            {"text": "Jeff Bezos", "start": 78, "end": 88, "type": "PERSON"},
        ],
    },
    {
        "text": "Meta Platforms Inc., formerly Facebook, opened a new research lab in London, United Kingdom. Mark Zuckerberg announced the investment.",
        "entities": [
            {"text": "Meta Platforms Inc.", "start": 0, "end": 19, "type": "ORG"},
            {"text": "Facebook", "start": 30, "end": 38, "type": "ORG"},
            {"text": "London", "start": 70, "end": 76, "type": "LOC"},
            {"text": "United Kingdom", "start": 78, "end": 92, "type": "LOC"},
            {"text": "Mark Zuckerberg", "start": 94, "end": 109, "type": "PERSON"},
        ],
    },
    {
        "text": "Netflix Inc. is producing original content in Mumbai, India and Seoul, South Korea. Reed Hastings discussed the strategy.",
        "entities": [
            {"text": "Netflix Inc.", "start": 0, "end": 12, "type": "ORG"},
            {"text": "Mumbai", "start": 46, "end": 52, "type": "LOC"},
            {"text": "India", "start": 54, "end": 59, "type": "LOC"},
            {"text": "Seoul", "start": 64, "end": 69, "type": "LOC"},
            {"text": "South Korea", "start": 71, "end": 82, "type": "LOC"},
            {"text": "Reed Hastings", "start": 84, "end": 97, "type": "PERSON"},
        ],
    },
    {
        "text": "NVIDIA Corporation unveiled new GPUs at a conference in San Jose, California. Jensen Huang demonstrated the technology.",
        "entities": [
            {"text": "NVIDIA Corporation", "start": 0, "end": 18, "type": "ORG"},
            {"text": "San Jose", "start": 56, "end": 64, "type": "LOC"},
            {"text": "California", "start": 66, "end": 76, "type": "LOC"},
            {"text": "Jensen Huang", "start": 78, "end": 90, "type": "PERSON"},
        ],
    },
    {
        "text": "IBM and Red Hat announced a partnership at the Boston office. Arvind Krishna highlighted the benefits of the collaboration.",
        "entities": [
            {"text": "IBM", "start": 0, "end": 3, "type": "ORG"},
            {"text": "Red Hat", "start": 8, "end": 15, "type": "ORG"},
            {"text": "Boston", "start": 47, "end": 53, "type": "LOC"},
            {"text": "Arvind Krishna", "start": 62, "end": 76, "type": "PERSON"},
        ],
    },
    {
        "text": "Intel Corporation is building a new semiconductor factory in Phoenix, Arizona. Pat Gelsinger announced the $20 billion investment.",
        "entities": [
            {"text": "Intel Corporation", "start": 0, "end": 17, "type": "ORG"},
            {"text": "Phoenix", "start": 61, "end": 68, "type": "LOC"},
            {"text": "Arizona", "start": 70, "end": 77, "type": "LOC"},
            {"text": "Pat Gelsinger", "start": 79, "end": 92, "type": "PERSON"},
        ],
    },
]

print("=" * 80)
print("NER DEMO: Learning to Extract Organizations, People, and Locations")
print("=" * 80)

print(f"\nAdding {len(training_examples)} training examples...")
for i, example in enumerate(training_examples, 1):
    chef.add_example(
        input_data={"text": example["text"]},
        output_data={"entities": example["entities"]},
    )

    print(f"  [{i}/{len(training_examples)}] {len(example['entities'])} entities")

print(f"\nBuffer status: {chef.buffer.get_stats()}")

# =============================================================================
# Manual Learning (if auto-trigger didn't fire)
# =============================================================================

if chef.buffer.get_stats()["new_examples"] > 0:
    print("\nManually triggering learning...")
    chef.learn_rules(run_evaluation=True, max_refinement_iterations=2)

# =============================================================================
# Show Learned Rules
# =============================================================================

print("\n" + "=" * 80)
print("LEARNED RULES")
print("=" * 80)

if chef.dataset.rules:
    for i, rule in enumerate(chef.get_rules_summary(), 1):
        print(f"\n{i}. {rule['name']}")
        print(f"   {rule['description']}")
        print(f"   Format: {rule['format']}, Priority: {rule['priority']}")
        print(f"   Confidence: {rule['confidence']}")
else:
    print("No rules learned (may need API key)")

# =============================================================================
# Test on New Examples
# =============================================================================

test_examples = [
    "Salesforce acquired Slack in a deal announced in San Francisco. Marc Benioff praised the team.",
    "Adobe Systems launched Creative Cloud updates at the conference in Las Vegas, Nevada.",
    "Oracle Corporation moved its headquarters to Austin, Texas. Larry Ellison explained the decision.",
]

print("\n" + "=" * 80)
print("TESTING ON NEW EXAMPLES")
print("=" * 80)

if chef.dataset.rules:
    for i, text in enumerate(test_examples, 1):
        print(f"\n{i}. Text: {text}")
        result = chef.extract({"text": text})

        entities = result.get("entities", result.get("spans", []))
        print(f"   Extracted {len(entities)} entities:")
        for span in entities:
            span_type = span.get("type") or span.get("label", "")
            print(
                f"     - '{span.get('text')}' ({span_type}) at [{span.get('start')}:{span.get('end')}]"
            )
else:
    print("Cannot test - no rules learned (API key required)")

# =============================================================================
# Value Proposition
# =============================================================================

print("\n" + "=" * 80)
print("VALUE PROPOSITION")
print("=" * 80)

if chef.dataset.rules:
    print(f"""
✓ Learned {len(chef.dataset.rules)} rules from {len(training_examples)} examples
✓ Rules now extract entities WITHOUT calling LLM
✓ Cost: ~$0 per extraction (vs $0.001-0.01 per LLM call)
✓ Latency: <1ms (vs 500-2000ms for LLM)
✓ Scale: Can process millions of documents cheaply

Cost savings example:
- Processing 1M documents with LLM: $1,000-10,000
- Processing 1M documents with rules: ~$0
- One-time learning cost: ~$1-5 (for initial examples)

→ Rules pay for themselves after ~100-1000 documents!
    """)
else:
    print("""
To see full demo, set OPENAI_API_KEY:
  export OPENAI_API_KEY='your-key'
  python ner_demo.py

Expected results:
- Learn regex patterns for organizations (Corp., Inc., etc.)
- Learn patterns for people names (First Last format)
- Learn patterns for locations (capitalized place names)
- Extract entities from new text without LLM calls
    """)

print("=" * 80)
