"""
Quick manual demo for incremental patch learning.

Requirements:
- Set OPENAI_API_KEY
- Optional: set OPENAI_BASE_URL if using a compatible endpoint

Run:
  python demo_incremental_patch.py
"""

import os
import tempfile

from openai import OpenAI

from rulechef import RuleChef, Task
from rulechef.core import RuleFormat


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for this demo")

    base_url = os.environ.get("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    task = Task(
        name="Q&A",
        description="Extract years",
        input_schema={"question": "str", "context": "str"},
        output_schema={"spans": "List[Span]"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        chef = RuleChef(
            task,
            client,
            storage_path=tmpdir,
            allowed_formats=[RuleFormat.REGEX],
            model="gpt-5.1",
        )

        print("Adding baseline example (1991)...")
        chef.add_example(
            {"question": "When?", "context": "Built in 1991"},
            {"spans": [{"text": "1991", "start": 9, "end": 13}]},
        )

        print("\n=== Baseline learn ===")
        chef.learn_rules(run_evaluation=False, max_refinement_iterations=1)
        print(f"Rules after baseline: {[r.name for r in chef.dataset.rules]}")

        new_input = {"question": "When?", "context": "Construction finished in 2025"}
        expected = {"spans": [{"text": "2025", "start": 26, "end": 30}]}

        print("\nExtracting before patch...")
        pre = chef.extract(new_input)
        print("Pre-patch output:", pre)

        print("\n=== Incremental patch learn ===")
        chef.learn_rules(
            run_evaluation=False, incremental_only=True, max_refinement_iterations=1
        )
        print(f"Rules after patch: {[r.name for r in chef.dataset.rules]}")

        print("\nExtracting after patch...")
        post = chef.extract(new_input)
        print("Post-patch output:", post)

        # Simple check for visibility
        if post.get("spans"):
            texts = [
                s["text"] if isinstance(s, dict) else s.text for s in post["spans"]
            ]
            print("Extracted spans:", texts)
            if any(t == "2025" for t in texts):
                print("✓ Patch appears to cover the new example.")
            else:
                print("⚠ Patch did not capture expected span; inspect outputs above.")
        else:
            print("⚠ No spans extracted after patch; inspect outputs above.")


if __name__ == "__main__":
    main()
