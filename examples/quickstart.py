"""Quickstart example for RuleChef (basic extraction)."""

import os
from openai import OpenAI
from rulechef import RuleChef, Task


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for this demo")

    client = OpenAI(api_key=api_key)
    task = Task(
        name="Q&A",
        description="Extract answer spans from text",
        input_schema={"question": "str", "context": "str"},
        output_schema={"spans": "List[Span]"},
        text_field="context",
    )

    chef = RuleChef(task, client, dataset_name="quickstart")

    chef.add_example(
        {"question": "When?", "context": "Built in 1991"},
        {"spans": [{"text": "1991", "start": 9, "end": 13}]},
    )
    chef.add_example(
        {"question": "When?", "context": "Construction finished in 2025"},
        {"spans": [{"text": "2025", "start": 25, "end": 29}]},
    )

    chef.learn_rules()

    result = chef.extract({"question": "When?", "context": "Renovated in 2033"})
    print("Output:", result)


if __name__ == "__main__":
    main()
