from __future__ import annotations

import asyncio
import traceback

from api.state import SessionState


async def run_learning(
    state: SessionState,
    sampling_strategy: str | None = None,
    max_iterations: int = 3,
    incremental_only: bool = True,
) -> None:
    """Run learn_rules() in a background thread for a specific session."""
    state.learning.progress = "Learning rules from examples..."

    try:
        async with state.write_lock:
            result = await asyncio.to_thread(
                state.chef.learn_rules,
                min_examples=2,
                max_refinement_iterations=max_iterations,
                sampling_strategy=sampling_strategy,
                incremental_only=incremental_only,
            )
        if not result:
            state.learning.last_metrics = None
            state.learning.progress = (
                "Not enough data yet. Add at least 2 annotated examples/corrections."
            )
            return
        rules, eval_result = result
        state.learning.last_metrics = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else eval_result
        )
        state.learning.progress = f"Done - learned {len(rules)} rule(s)"
    except Exception as exc:
        state.learning.error = f"{exc}\n{traceback.format_exc()}"
        state.learning.progress = "Failed"
    finally:
        async with state.learning_lock:
            state.learning.running = False
