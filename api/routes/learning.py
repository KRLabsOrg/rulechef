from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_session
from api.schemas import (
    AddFeedbackRequest,
    FeedbackItem,
    LearnRequest,
    LearnStatusResponse,
)
from api.state import SessionState
from api.tasks import run_learning
from rulechef.coordinator import AgenticCoordinator

router = APIRouter(prefix="/api/learn", tags=["learning"])


@router.post("")
async def trigger_learning(
    req: LearnRequest,
    state: Annotated[SessionState, Depends(get_session)],
):
    async with state.learning_lock:
        if state.learning.running:
            raise HTTPException(409, "Learning already in progress")
        state.learning.running = True
        state.learning.progress = "Queued learning..."
        state.learning.error = None

    if req.use_agentic:
        coordinator = AgenticCoordinator(
            llm_client=state.chef.llm,
            model=state.chef.model,
        )
        state.chef.coordinator = coordinator

    asyncio.create_task(
        run_learning(
            state,
            sampling_strategy=req.sampling_strategy,
            max_iterations=req.max_iterations,
            incremental_only=req.incremental_only,
        )
    )

    return {"ok": True, "message": "Learning started"}


@router.get("/status", response_model=LearnStatusResponse)
async def learning_status(state: Annotated[SessionState, Depends(get_session)]):
    return LearnStatusResponse(
        running=state.learning.running,
        progress=state.learning.progress,
        error=state.learning.error,
        metrics=state.learning.last_metrics,
    )


@router.get("/evaluate")
async def evaluate_rules(state: Annotated[SessionState, Depends(get_session)]):
    async with state.write_lock:
        result = state.chef.evaluate(verbose=False)
    return result.to_dict()


@router.post("/feedback")
async def add_feedback(
    req: AddFeedbackRequest,
    state: Annotated[SessionState, Depends(get_session)],
):
    async with state.write_lock:
        state.chef.add_feedback(req.text, level=req.level, target_id=req.target_id)
    return {"ok": True}


@router.delete("/feedback/{feedback_id}")
async def delete_feedback(
    feedback_id: str,
    state: Annotated[SessionState, Depends(get_session)],
):
    async with state.write_lock:
        fb_list = state.chef.dataset.structured_feedback
        for i, fb in enumerate(fb_list):
            if fb.id == feedback_id:
                fb_list.pop(i)
                return {"ok": True}
    raise HTTPException(404, f"Feedback {feedback_id} not found")


@router.get("/feedback")
async def list_feedback(state: Annotated[SessionState, Depends(get_session)]):
    async with state.write_lock:
        items = [
            FeedbackItem(
                id=fb.id,
                text=fb.text,
                level=fb.level,
                target_id=fb.target_id,
                timestamp=fb.timestamp.isoformat(),
            )
            for fb in state.chef.dataset.structured_feedback
        ]
    return {"feedback": items}
