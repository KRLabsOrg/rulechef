"""Test NER with Pydantic schema and labels"""

import os
from pydantic import BaseModel, Field
from typing import Literal, List
from openai import OpenAI
from rulechef import RuleChef, Task, TaskType
from rulechef.core import RuleFormat


# Define Pydantic schema with typed labels
class Entity(BaseModel):
    text: str = Field(description="The matched text span")
    start: int = Field(description="Start character offset")
    end: int = Field(description="End character offset")
    type: Literal["PER", "ORG", "LOC"] = Field(
        description="PER=Person, ORG=Organization, LOC=Location"
    )


class NEROutput(BaseModel):
    entities: List[Entity]


# Create task
task = Task(
    name="Named Entity Recognition",
    description="Extract person names, organizations, and locations from text",
    input_schema={"text": "str"},
    output_schema=NEROutput,
    type=TaskType.NER,
)

# Verify labels are extracted
print(f"Labels from schema: {task.get_labels()}")

# Create RuleChef
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1/",
)
chef = RuleChef(
    task,
    client,
    allowed_formats=[RuleFormat.REGEX],
    model="moonshotai/kimi-k2-instruct-0905",
)

# Add training examples
examples = [
    {
        "input": {"text": "John Smith works at Google in New York."},
        "output": {
            "entities": [
                {"text": "John Smith", "start": 0, "end": 10, "type": "PER"},
                {"text": "Google", "start": 20, "end": 26, "type": "ORG"},
                {"text": "New York", "start": 30, "end": 38, "type": "LOC"},
            ]
        },
    },
    {
        "input": {"text": "Jane Doe joined Microsoft last year."},
        "output": {
            "entities": [
                {"text": "Jane Doe", "start": 0, "end": 8, "type": "PER"},
                {"text": "Microsoft", "start": 16, "end": 25, "type": "ORG"},
            ]
        },
    },
    {
        "input": {"text": "Tim Cook announced the new iPhone in California."},
        "output": {
            "entities": [
                {"text": "Tim Cook", "start": 0, "end": 8, "type": "PER"},
                {
                    "text": "iPhone",
                    "start": 27,
                    "end": 33,
                    "type": "ORG",
                },  # product, but using ORG
                {"text": "California", "start": 37, "end": 47, "type": "LOC"},
            ]
        },
    },
]

print("\nAdding examples...")
for ex in examples:
    chef.add_example(ex["input"], ex["output"])

# Learn rules
print("\nLearning rules...")
chef.learn_rules()

# Show learned rules
print(f"\nLearned {len(chef.dataset.rules)} rules:")
for rule in chef.dataset.rules:
    print(f"  - {rule.name}: {rule.format.value}")
    print(
        f"    Pattern: {rule.content[:80]}..."
        if len(rule.content) > 80
        else f"    Pattern: {rule.content}"
    )
    print(f"    Template: {rule.output_template}")

# Test extraction
test_inputs = [
    {"text": "Bob Johnson visited Amazon headquarters in Seattle."},
    {"text": "Mary Williams is the CEO of Tesla."},
    {"text": "The meeting was held in London with Sarah Connor."},
]

print("\n" + "=" * 60)
print("Testing extraction:")
print("=" * 60)

for test in test_inputs:
    print(f"\nInput: {test['text']}")
    result = chef.extract(test)
    print(f"Output: {result}")

    # Validate
    is_valid, errors = task.validate_output(result)
    if not is_valid:
        print(f"Validation errors: {errors}")
