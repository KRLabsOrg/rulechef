from __future__ import annotations

import re
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAI

from api.config import settings
from api.deps import get_session
from api.schemas import ConfigureRequest, StatusResponse
from api.state import LearningStatus, SessionState
from rulechef import RuleChef, Task
from rulechef.core import RuleFormat, TaskType

router = APIRouter(prefix="/api/project", tags=["project"])

TASK_TYPE_MAP = {
    "ner": TaskType.NER,
    "extraction": TaskType.EXTRACTION,
    "classification": TaskType.CLASSIFICATION,
    "transformation": TaskType.TRANSFORMATION,
}

FORMAT_MAP = {
    "regex": RuleFormat.REGEX,
    "code": RuleFormat.CODE,
    "spacy": RuleFormat.SPACY,
}


def _safe_dataset_name(raw: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", (raw or "").strip().lower())
    normalized = normalized.strip("_")[:64]
    return normalized or "task"


def _build_status(state: SessionState) -> StatusResponse:
    chef = state.chef
    task = state.task
    stats = chef.get_stats()
    buffer_stats = chef.get_buffer_stats()
    stats["pending_examples"] = buffer_stats.get("new_examples", 0)
    stats["pending_corrections"] = buffer_stats.get("new_corrections", 0)
    stats["draft_examples"] = len(state.draft_examples)
    return StatusResponse(
        configured=True,
        task_name=task.name,
        task_type=task.type.value,
        stats=stats,
    )


@router.post("/configure", response_model=StatusResponse)
async def configure(
    req: ConfigureRequest,
    state: Annotated[SessionState, Depends(get_session)],
):
    task_type = TASK_TYPE_MAP.get(req.task_type)
    if task_type is None:
        raise HTTPException(400, f"Invalid task_type: {req.task_type}")

    api_key = req.openai_api_key or settings.OPENAI_API_KEY
    base_url = req.openai_base_url or settings.OPENAI_BASE_URL

    task = Task(
        name=req.task_name,
        description=req.task_description,
        input_schema=req.input_schema,
        output_schema=req.output_schema,
        type=task_type,
        text_field=req.text_field or None,
    )

    kwargs: dict = {"api_key": (api_key or "demo-key").strip()}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    formats = [FORMAT_MAP[f] for f in req.allowed_formats if f in FORMAT_MAP] or None

    dataset_suffix = _safe_dataset_name(req.task_name)
    chef = RuleChef(
        task,
        client,
        dataset_name=f"workspace_{state.workspace_id}_{dataset_suffix}",
        storage_path=settings.STORAGE_PATH,
        model=req.model,
        allowed_formats=formats,
    )

    async with state.write_lock:
        if state.learning.running:
            raise HTTPException(409, "Cannot reconfigure while learning is running")
        state.task = task
        state.chef = chef
        state.allowed_formats = req.allowed_formats

    return _build_status(state)


@router.get("/status", response_model=StatusResponse)
async def status(state: Annotated[SessionState, Depends(get_session)]):
    return _build_status(state)


@router.post("/reset", response_model=StatusResponse)
async def reset_workspace(state: Annotated[SessionState, Depends(get_session)]):
    async with state.write_lock:
        if state.learning.running:
            raise HTTPException(409, "Cannot reset while learning is running")

        task = state.task
        client = state.chef.llm
        formats = [FORMAT_MAP[f] for f in state.allowed_formats if f in FORMAT_MAP] or None
        state.workspace_id = f"{state.session_id[:4]}_{int(time.time())}"
        dataset_suffix = _safe_dataset_name(task.name)
        dataset_name = f"workspace_{state.workspace_id}_{dataset_suffix}"
        chef = RuleChef(
            task,
            client,
            dataset_name=dataset_name,
            storage_path=settings.STORAGE_PATH,
            model=state.chef.model,
            allowed_formats=formats,
        )
        state.chef = chef
        state.draft_examples = []
        state.learning = LearningStatus()

    return _build_status(state)
