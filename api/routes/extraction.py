from __future__ import annotations

from copy import deepcopy
from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_session
from api.schemas import (
    BatchExtractRequest,
    BatchExtractResponse,
    ExtractRequest,
    ExtractResponse,
)
from api.state import SessionState

router = APIRouter(prefix="/api/extract", tags=["extraction"])


def _normalize_entities_in_result(result: dict) -> dict:
    """Collapse overlapping spans for clearer UI and deterministic output."""
    normalized = deepcopy(result)

    for key in ("entities", "spans"):
        items = normalized.get(key)
        if not isinstance(items, list):
            continue

        candidates: list[tuple[int, int, dict]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            start = item.get("start")
            end = item.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                continue
            candidates.append((start, end, item))

        # Prefer longer spans when starts are equal, then keep left-to-right.
        candidates.sort(
            key=lambda t: (
                t[0],
                -(t[1] - t[0]),
                str(t[2].get("type") or t[2].get("label") or ""),
            )
        )

        selected: list[tuple[int, int, dict]] = []
        for start, end, item in candidates:
            has_overlap = any(start < s_end and end > s_start for s_start, s_end, _ in selected)
            if not has_overlap:
                selected.append((start, end, item))

        selected.sort(key=lambda t: t[0])
        normalized[key] = [item for _, _, item in selected]

    return normalized


@router.post("", response_model=ExtractResponse)
async def extract(
    req: ExtractRequest,
    state: Annotated[SessionState, Depends(get_session)],
):
    async with state.write_lock:
        result = _normalize_entities_in_result(state.chef.extract(req.input_data))
    return ExtractResponse(result=result)


@router.post("/batch", response_model=BatchExtractResponse)
async def batch_extract(
    req: BatchExtractRequest,
    state: Annotated[SessionState, Depends(get_session)],
):
    async with state.write_lock:
        results = [_normalize_entities_in_result(state.chef.extract(inp)) for inp in req.inputs]
    return BatchExtractResponse(results=results)
