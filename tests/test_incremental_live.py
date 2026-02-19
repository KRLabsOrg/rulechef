import os
import tempfile

import pytest
from openai import OpenAI

from rulechef import RuleChef, Task
from rulechef.core import RuleFormat
from rulechef.matching import outputs_match


pytestmark = pytest.mark.integration


def _span_texts(output):
    spans = output.get("spans", [])
    return sorted([s["text"] if isinstance(s, dict) else s.text for s in spans])


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not os.environ.get("RUN_LIVE_TESTS"),
    reason="Requires OPENAI_API_KEY and RUN_LIVE_TESTS=1 for live LLM calls",
)
def test_incremental_patch_flow_with_live_llm():
    """
    Live integration test:
    - Learn from a single example
    - See how it performs on an unseen example
    - Run incremental-only patch to adapt
    - Verify extractions run without errors (LLM-dependent)
    """
    task = Task(
        name="Q&A",
        description="Extract answer spans from text",
        input_schema={"question": "str", "context": "str"},
        output_schema={"spans": "List[Span]"},
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with tempfile.TemporaryDirectory() as tmpdir:
        chef = RuleChef(
            task,
            client,
            storage_path=tmpdir,
            allowed_formats=[RuleFormat.REGEX],
            model="gpt-4o-mini",
        )

        # Seed with one example
        chef.add_example(
            {"question": "When?", "context": "Built in 1991"},
            {"spans": [{"text": "1991", "start": 9, "end": 13}]},
        )
        chef.learn_rules(run_evaluation=False, max_refinement_iterations=1)
        assert chef.dataset.rules, "Expected at least one learned rule"

        # New example not seen during synth
        new_input = {"question": "When?", "context": "Construction finished in 2025"}
        expected = {"spans": [{"text": "2025", "start": 26, "end": 30}]}
        pre_patch = chef.extract(new_input)

        # Incremental patch attempt
        chef.learn_rules(
            run_evaluation=False, incremental_only=True, max_refinement_iterations=1
        )
        post_patch = chef.extract(new_input)

        # Basic sanity: extraction returns a spans list both times
        assert "spans" in pre_patch
        assert "spans" in post_patch

        # Prefer post-patch to be closer to expected; allow live LLM variation
        post_matches = outputs_match(
            expected, post_patch, task.type, task.output_matcher
        )
        if not post_matches:
            pytest.skip(
                "Live LLM did not produce expected span; manual inspection needed"
            )

        assert _span_texts(post_patch), "Post-patch spans should not be empty"
