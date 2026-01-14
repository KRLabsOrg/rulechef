"""
Agentic coordinator showcase.

Demonstrates how the agentic coordinator selects a learning strategy and
triggers a ruleset update based on buffered examples and corrections.
"""

import os
import tempfile
from openai import OpenAI

from rulechef import RuleChef, Task
from rulechef.core import RuleFormat
from rulechef.coordinator import AgenticCoordinator


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for this demo")

    client = OpenAI(api_key=api_key)
    task = Task(
        name="Q&A",
        description="Extract years from text",
        input_schema={"question": "str", "context": "str"},
        output_schema={"spans": "List[Span]"},
        text_field="context",
    )

    coordinator = AgenticCoordinator(
        client,
        model="gpt-5.1",
        min_batch_size=1,
        min_correction_batch=0,
        verbose=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        chef = RuleChef(
            task,
            client,
            storage_path=tmpdir,
            allowed_formats=[RuleFormat.REGEX],
            coordinator=coordinator,
            auto_trigger=False,
            model="gpt-5.1",
        )

        chef.add_example(
            {"question": "When?", "context": "Built in 1991"},
            {"spans": [{"text": "1991", "start": 9, "end": 13}]},
        )
        chef.add_example(
            {"question": "When?", "context": "Construction finished in 2025"},
            {"spans": [{"text": "2025", "start": 25, "end": 29}]},
        )
        chef.add_correction(
            {"question": "When?", "context": "Celebrated in 2005."},
            model_output={"spans": []},
            expected_output={"spans": [{"text": "2005", "start": 13, "end": 17}]},
        )

        decision = coordinator.should_trigger_learning(chef.buffer, chef.dataset.rules)
        print(
            f"Coordinator decision: learn={decision.should_learn}, "
            f"strategy={decision.strategy}, reason={decision.reasoning}"
        )

        if decision.should_learn:
            chef.learn_rules(
                sampling_strategy=decision.strategy, max_refinement_iterations=2
            )

        result = chef.extract({"question": "When?", "context": "Renovated in 2033"})
        print("Output:", result)


if __name__ == "__main__":
    main()
