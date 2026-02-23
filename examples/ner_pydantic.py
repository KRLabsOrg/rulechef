"""NER with Pydantic schema — type-safe entities with automatic label discovery.

Usage:
    export OPENAI_API_KEY='your-key'
    python examples/ner_pydantic.py
"""

import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from rulechef import RuleChef, RuleFormat, Task, TaskType


# Define Pydantic schema with typed labels
class Entity(BaseModel):
    text: str = Field(description="The matched text span")
    start: int = Field(description="Start character offset")
    end: int = Field(description="End character offset")
    type: Literal["PER", "ORG", "LOC"] = Field(
        description="PER=Person, ORG=Organization, LOC=Location"
    )


class NEROutput(BaseModel):
    entities: list[Entity]


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    client_kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    model = os.environ.get("RULECHEF_MODEL", "gpt-4o-mini")

    task = Task(
        name="Named Entity Recognition",
        description="Extract person names, organizations, and locations from text",
        input_schema={"text": "str"},
        output_schema=NEROutput,
        type=TaskType.NER,
    )

    # RuleChef discovers labels automatically from the Pydantic Literal type
    print(f"Labels from schema: {task.get_labels()}")

    chef = RuleChef(
        task,
        client,
        allowed_formats=[RuleFormat.REGEX],
        model=model,
    )

    # Training examples
    chef.add_example(
        {"text": "John Smith works at Google in New York."},
        {
            "entities": [
                {"text": "John Smith", "start": 0, "end": 10, "type": "PER"},
                {"text": "Google", "start": 20, "end": 26, "type": "ORG"},
                {"text": "New York", "start": 30, "end": 38, "type": "LOC"},
            ]
        },
    )
    chef.add_example(
        {"text": "Jane Doe joined Microsoft last year."},
        {
            "entities": [
                {"text": "Jane Doe", "start": 0, "end": 8, "type": "PER"},
                {"text": "Microsoft", "start": 16, "end": 25, "type": "ORG"},
            ]
        },
    )
    chef.add_example(
        {"text": "Tim Cook announced the new iPhone in California."},
        {
            "entities": [
                {"text": "Tim Cook", "start": 0, "end": 8, "type": "PER"},
                {"text": "California", "start": 37, "end": 47, "type": "LOC"},
            ]
        },
    )

    # Learn rules
    print("\nLearning rules...")
    rules, eval_result = chef.learn_rules()
    print(f"Learned {len(rules)} rules")

    # Test on new inputs
    test_inputs = [
        {"text": "Bob Johnson visited Amazon headquarters in Seattle."},
        {"text": "Mary Williams is the CEO of Tesla."},
        {"text": "The meeting was held in London with Sarah Connor."},
    ]

    print("\nTesting on new inputs:")
    for test in test_inputs:
        result = chef.extract(test)
        entities = result.get("entities", [])
        print(f"\n  Input: {test['text']}")
        for e in entities:
            print(f"    {e['text']} ({e.get('type', '?')})")

        # Pydantic validation
        is_valid, errors = task.validate_output(result)
        if not is_valid:
            print(f"    Validation errors: {errors}")


if __name__ == "__main__":
    main()
