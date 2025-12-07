"""
Schema-Aware Transformation Example - Transform text to structured JSON

Demonstrates the schema-aware rules feature for arbitrary transformations:
- Rules have output_template that maps patterns to custom JSON output
- Works with any output schema, not just span extraction
- Evaluation uses exact JSON matching
"""

import os
from openai import OpenAI
from rulechef import RuleChef, Task, TaskType
from rulechef.coordinator import SimpleCoordinator
from rulechef.evaluation import (
    evaluate_key_value_extraction,
    print_key_value_eval_report,
)
import json
# =============================================================================
# Define Transformation Task
# =============================================================================

task = Task(
    name="Email Metadata Extraction",
    description="Extract structured metadata from email headers and signatures",
    input_schema={"email_text": "str"},
    output_schema={"metadata": "List[{key: str, value: str}]"},
    type=TaskType.TRANSFORMATION,
)

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("⚠️  OPENAI_API_KEY not set - running in demo mode (no rule learning)")
    print("Set it with: export OPENAI_API_KEY='your-key'")
    chef = RuleChef(
        task,
        dataset_name="transformation_demo",
        allowed_formats=["regex", "spacy"],
    )
else:
    client = OpenAI(api_key=api_key)
    coordinator = SimpleCoordinator(trigger_threshold=10, verbose=False)
    chef = RuleChef(
        task,
        client=client,
        dataset_name="transformation",
        coordinator=coordinator,
        auto_trigger=False,
        allowed_formats=["regex", "spacy"],
        model="gpt-5.1",
    )

# =============================================================================
# Training Data - Structured Metadata
# =============================================================================

training_examples = [
    {
        "email_text": """From: john.doe@company.com
To: jane.smith@company.com
Date: 2024-01-15
Subject: Project Update

Hi Jane,

The project is on track. See details below.

Best regards,
John Doe
Senior Engineer
john.doe@company.com
(555) 123-4567""",
        "metadata": [
            {"key": "from_email", "value": "john.doe@company.com"},
            {"key": "to_email", "value": "jane.smith@company.com"},
            {"key": "date", "value": "2024-01-15"},
            {"key": "subject", "value": "Project Update"},
            {"key": "sender_name", "value": "John Doe"},
            {"key": "sender_title", "value": "Senior Engineer"},
            {"key": "sender_phone", "value": "(555) 123-4567"},
        ],
    },
    {
        "email_text": """From: alice@techcorp.net
To: bob@techcorp.net, charlie@techcorp.net
Date: 2024-01-20
Subject: Q1 Planning

Team,

Let's sync on Q1 strategy.

Cheers,
Alice Martinez
Manager, Product Team
alice@techcorp.net
Office: (555) 987-6543""",
        "metadata": [
            {"key": "from_email", "value": "alice@techcorp.net"},
            {"key": "to_email", "value": "bob@techcorp.net"},
            {"key": "date", "value": "2024-01-20"},
            {"key": "subject", "value": "Q1 Planning"},
            {"key": "sender_name", "value": "Alice Martinez"},
            {"key": "sender_title", "value": "Manager, Product Team"},
            {"key": "sender_phone", "value": "(555) 987-6543"},
        ],
    },
    {
        "email_text": """From: support@service.io
To: customer@example.com
Date: 2024-01-25
Subject: Ticket #12345 - Account Access Issue

Dear Customer,

Thank you for contacting us. We're looking into your account access issue.

Support Team
Technical Support Division
support@service.io
Phone: 1-800-555-0123""",
        "metadata": [
            {"key": "from_email", "value": "support@service.io"},
            {"key": "to_email", "value": "customer@example.com"},
            {"key": "date", "value": "2024-01-25"},
            {"key": "subject", "value": "Ticket #12345 - Account Access Issue"},
            {"key": "sender_name", "value": "Support Team"},
            {"key": "sender_title", "value": "Technical Support Division"},
            {"key": "sender_phone", "value": "1-800-555-0123"},
        ],
    },
    {
        "email_text": """From: robert.wilson@corp.com
To: sarah.johnson@corp.com
Date: 2024-02-01
Subject: Budget Review - FY2024

Sarah,

Please review the attached budget document.

Best,
Robert Wilson
CFO, Finance Department
robert.wilson@corp.com
Direct: (555) 111-2222""",
        "metadata": [
            {"key": "from_email", "value": "robert.wilson@corp.com"},
            {"key": "to_email", "value": "sarah.johnson@corp.com"},
            {"key": "date", "value": "2024-02-01"},
            {"key": "subject", "value": "Budget Review - FY2024"},
            {"key": "sender_name", "value": "Robert Wilson"},
            {"key": "sender_title", "value": "CFO, Finance Department"},
            {"key": "sender_phone", "value": "(555) 111-2222"},
        ],
    },
    {
        "email_text": """From: hr@company.org
To: newemployee@company.org
Date: 2024-02-05
Subject: Welcome to Our Team!

Welcome aboard!

Your onboarding begins Monday.

HR Department
Human Resources
hr@company.org
Ext: 5555""",
        "metadata": [
            {"key": "from_email", "value": "hr@company.org"},
            {"key": "to_email", "value": "newemployee@company.org"},
            {"key": "date", "value": "2024-02-05"},
            {"key": "subject", "value": "Welcome to Our Team!"},
            {"key": "sender_name", "value": "HR Department"},
            {"key": "sender_title", "value": "Human Resources"},
        ],
    },
]

# Test examples (held out)
test_examples = [
    {
        "email_text": """From: michael.brown@enterprise.com
To: lisa.anderson@enterprise.com
Date: 2024-02-10
Subject: Strategic Partnership Proposal

Lisa,

I'd like to discuss a new partnership opportunity.

Regards,
Michael Brown
VP of Business Development
michael.brown@enterprise.com
Phone: (555) 444-5555""",
        "metadata": [
            {"key": "from_email", "value": "michael.brown@enterprise.com"},
            {"key": "to_email", "value": "lisa.anderson@enterprise.com"},
            {"key": "date", "value": "2024-02-10"},
            {"key": "subject", "value": "Strategic Partnership Proposal"},
            {"key": "sender_name", "value": "Michael Brown"},
            {"key": "sender_title", "value": "VP of Business Development"},
            {"key": "sender_phone", "value": "(555) 444-5555"},
        ],
    },
    {
        "email_text": """From: design@studio.com
To: client@business.com
Date: 2024-02-15
Subject: Design Mockups - Project Alpha

Hi Client,

Attached are the design mockups for review.

Design Team
Creative Services
design@studio.com
Contact: (555) 666-7777""",
        "metadata": [
            {"key": "from_email", "value": "design@studio.com"},
            {"key": "to_email", "value": "client@business.com"},
            {"key": "date", "value": "2024-02-15"},
            {"key": "subject", "value": "Design Mockups - Project Alpha"},
            {"key": "sender_name", "value": "Design Team"},
            {"key": "sender_title", "value": "Creative Services"},
            {"key": "sender_phone", "value": "(555) 666-7777"},
        ],
    },
    {
        "email_text": """From: legal@lawfirm.net
To: corporate@company.com
Date: 2024-02-20
Subject: Contract Review - Non-Disclosure Agreement

Corporate Team,

Please sign and return the attached NDA.

Legal Department
Counsel Services
legal@lawfirm.net
Office: 1-800-555-1234""",
        "metadata": [
            {"key": "from_email", "value": "legal@lawfirm.net"},
            {"key": "to_email", "value": "corporate@company.com"},
            {"key": "date", "value": "2024-02-20"},
            {"key": "subject", "value": "Contract Review - Non-Disclosure Agreement"},
            {"key": "sender_name", "value": "Legal Department"},
            {"key": "sender_title", "value": "Counsel Services"},
            {"key": "sender_phone", "value": "1-800-555-1234"},
        ],
    },
]

print("=" * 80)
print("TRANSFORMATION EXAMPLE - Email Metadata Extraction")
print("=" * 80)

print(f"\n📊 Dataset: {len(training_examples)} training, {len(test_examples)} test")
print("   Output schema: metadata array with key-value pairs")

# =============================================================================
# Add Training Examples
# =============================================================================

print(f"\n📥 Adding {len(training_examples)} training examples...")
for example in training_examples:
    chef.add_example(
        input_data={"email_text": example["email_text"]},
        output_data={"metadata": example["metadata"]},
    )

print(f"✓ Buffer stats: {chef.buffer.get_stats()}")

# =============================================================================
# Learn Rules
# =============================================================================

if api_key:
    print("\n🤖 Learning transformation rules...")
    chef.learn_rules(run_evaluation=False, max_refinement_iterations=2)

    # Show learned rules
    if chef.dataset.rules:
        print(f"\n✓ Learned {len(chef.dataset.rules)} rules:")
        for rule in chef.dataset.rules[:5]:
            print(rule)
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
    text = example["email_text"]
    gold = example["metadata"]

    # Extract using learned rules
    result = chef.extract({"email_text": text})
    predictions = result.get("metadata", [])

    # Check if rules were used (rules add start/end, LLM doesn't)
    used_rules = predictions and "start" in predictions[0]

    all_predictions.extend(predictions)
    all_gold.extend(gold)

    # Show individual results
    source = "RULES" if used_rules else "LLM"
    print(
        f"\nEmail from: {[m['value'] for m in gold if m['key'] == 'from_email']}... [{source}]"
    )
    print(f"  Gold:      {len(gold)} metadata items")
    print(f"  Predicted: {len(predictions)} metadata items")
    if predictions:
        print(json.dumps(predictions, indent=2))

# =============================================================================
# Evaluation
# =============================================================================

if all_predictions or all_gold:
    # For transformation, evaluate by key-value matching
    metrics = evaluate_key_value_extraction(all_predictions, all_gold)
    print_key_value_eval_report(metrics, "Email Metadata Extraction")
else:
    print("\n⚠️  No predictions to evaluate (need rules)")

print("=" * 80)
print("✓ Done!")
print("=" * 80)
