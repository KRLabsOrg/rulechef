from __future__ import annotations

import copy
import csv
import io
import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from api.deps import get_session
from api.schemas import (
    AddCorrectionRequest,
    AddExampleRequest,
    AddRawLinesRequest,
    AnnotateExampleRequest,
    UploadResponse,
)
from api.state import SessionState

router = APIRouter(prefix="/api/data", tags=["data"])


def _task_text_field(chef) -> str:
    task = chef.dataset.task
    return task.text_field or "text"


def _default_empty_output(chef) -> dict:
    schema = getattr(chef.dataset.task, "output_schema", {}) or {}
    if "entities" in schema:
        return {"entities": []}

    empty: dict = {}
    for key, type_name in schema.items():
        normalized = str(type_name).lower()
        if "list" in normalized:
            empty[key] = []
        elif "dict" in normalized or "{" in normalized:
            empty[key] = {}
        elif "bool" in normalized:
            empty[key] = False
        elif "int" in normalized or "float" in normalized:
            empty[key] = 0
        else:
            empty[key] = ""
    return empty


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _new_draft_id() -> str:
    return f"draft:{uuid.uuid4().hex[:10]}"


@router.post("/example")
async def add_example(
    req: AddExampleRequest,
    state: Annotated[SessionState, Depends(get_session)],
):
    async with state.write_lock:
        state.chef.add_example(req.input_data, req.output_data)
    return {"ok": True}


@router.post("/correction")
async def add_correction(
    req: AddCorrectionRequest,
    state: Annotated[SessionState, Depends(get_session)],
):
    async with state.write_lock:
        state.chef.add_correction(
            req.input_data,
            req.model_output,
            req.expected_output,
            feedback=req.feedback,
        )
    return {"ok": True}


@router.post("/raw-lines", response_model=UploadResponse)
async def add_raw_lines(
    req: AddRawLinesRequest,
    state: Annotated[SessionState, Depends(get_session)],
):
    lines = [line.strip() for line in req.lines if line and line.strip()]
    if not lines:
        raise HTTPException(400, "No non-empty lines provided")

    async with state.write_lock:
        text_field = _task_text_field(state.chef)
        empty_output = _default_empty_output(state.chef)
        for line in lines:
            state.draft_examples.append(
                {
                    "id": _new_draft_id(),
                    "input": {text_field: line},
                    "expected_output": copy.deepcopy(empty_output),
                    "source": "draft",
                }
            )

    return UploadResponse(count=len(lines), message=f"Added {len(lines)} draft example(s)")


@router.post("/example/{example_id}/annotation")
async def annotate_example(
    example_id: str,
    req: AnnotateExampleRequest,
    state: Annotated[SessionState, Depends(get_session)],
):
    chef = state.chef

    async with state.write_lock:
        if example_id.startswith("draft:"):
            for i, draft in enumerate(state.draft_examples):
                if draft.get("id") == example_id:
                    input_data = _as_dict(draft.get("input", {}))
                    chef.add_example(input_data, req.output_data)
                    state.draft_examples.pop(i)
                    return {"ok": True, "message": f"Annotated and added {example_id}"}
            raise HTTPException(404, f"Draft example {example_id} not found")

        if example_id.startswith("buffer:"):
            try:
                index = int(example_id.split(":", 1)[1])
            except ValueError as exc:
                raise HTTPException(400, "Invalid buffered example id") from exc

            with chef.buffer.lock:
                if index < 0 or index >= len(chef.buffer.examples):
                    raise HTTPException(404, f"Buffered example {example_id} not found")

                observed = chef.buffer.examples[index]
                if observed.is_correction:
                    raise HTTPException(400, "Cannot annotate a correction as an example")
                observed.output = req.output_data

            return {"ok": True, "message": f"Updated {example_id}"}

        if example_id.startswith("buffer-correction:"):
            raise HTTPException(400, "Cannot annotate a correction as an example")

        for example in chef.dataset.examples:
            if example.id == example_id:
                example.expected_output = req.output_data
                # Persist edited annotations immediately for existing dataset items.
                if hasattr(chef, "_save_dataset"):
                    chef._save_dataset()
                return {"ok": True, "message": f"Updated {example_id}"}

    raise HTTPException(404, f"Example {example_id} not found")


@router.delete("/example/{example_id}")
async def delete_example(
    example_id: str,
    state: Annotated[SessionState, Depends(get_session)],
):
    async with state.write_lock:
        if example_id.startswith("draft:"):
            for i, draft in enumerate(state.draft_examples):
                if draft.get("id") == example_id:
                    state.draft_examples.pop(i)
                    return {"ok": True, "message": f"Deleted draft {example_id}"}
            raise HTTPException(404, f"Draft example {example_id} not found")

        if example_id.startswith("buffer:"):
            try:
                index = int(example_id.split(":", 1)[1])
            except ValueError as exc:
                raise HTTPException(400, "Invalid buffered example id") from exc

            chef = state.chef
            with chef.buffer.lock:
                if index < 0 or index >= len(chef.buffer.examples):
                    raise HTTPException(404, f"Buffered example {example_id} not found")
                chef.buffer.examples.pop(index)

            return {"ok": True, "message": f"Deleted {example_id}"}

        chef = state.chef
        for i, example in enumerate(chef.dataset.examples):
            if example.id == example_id:
                chef.dataset.examples.pop(i)
                if hasattr(chef, "_save_dataset"):
                    chef._save_dataset()
                return {"ok": True, "message": f"Deleted {example_id}"}

    raise HTTPException(404, f"Example {example_id} not found")


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile,
    state: Annotated[SessionState, Depends(get_session)],
):
    chef = state.chef

    content = await file.read()
    text = content.decode("utf-8")

    rows: list[dict] = []
    filename = (file.filename or "").lower()

    if filename.endswith(".txt"):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise HTTPException(400, "No non-empty lines found in file")

        async with state.write_lock:
            text_field = _task_text_field(chef)
            empty_output = _default_empty_output(chef)
            for line in lines:
                state.draft_examples.append(
                    {
                        "id": _new_draft_id(),
                        "input": {text_field: line},
                        "expected_output": copy.deepcopy(empty_output),
                        "source": "draft",
                    }
                )
        return UploadResponse(
            count=len(lines),
            message=f"Added {len(lines)} draft example(s) from text file",
        )

    if filename.endswith(".json") or filename.endswith(".jsonl"):
        try:
            data = json.loads(text)
            rows = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            for line in text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise HTTPException(400, f"Invalid JSONL row: {line[:120]}") from exc
    elif filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    else:
        raise HTTPException(400, "Unsupported file type. Use .txt, .json, .jsonl, or .csv")

    count = 0
    async with state.write_lock:
        for row in rows:
            if "input" in row and "output" in row:
                inp = row["input"] if isinstance(row["input"], dict) else json.loads(row["input"])
                out = (
                    row["output"] if isinstance(row["output"], dict) else json.loads(row["output"])
                )
                chef.add_example(inp, out)
                count += 1
            else:
                input_keys = set(chef.dataset.task.input_schema.keys())
                inp = {k: v for k, v in row.items() if k in input_keys}
                out = {k: v for k, v in row.items() if k not in input_keys}
                if inp and out:
                    for k, v in out.items():
                        if isinstance(v, str) and v.startswith("["):
                            try:
                                out[k] = json.loads(v)
                            except json.JSONDecodeError:
                                pass
                    chef.add_example(inp, out)
                    count += 1

    return UploadResponse(count=count, message=f"Added {count} example(s)")


@router.get("/examples")
async def list_examples(state: Annotated[SessionState, Depends(get_session)]):
    async with state.write_lock:
        chef = state.chef
        examples = [
            {
                "id": ex.id,
                "input": ex.input,
                "expected_output": ex.expected_output,
                "source": ex.source,
            }
            for ex in chef.dataset.examples
        ]
        examples = copy.deepcopy(state.draft_examples) + examples

        corrections = [
            {
                "id": c.id,
                "input": c.input,
                "model_output": c.model_output,
                "expected_output": c.expected_output,
                "feedback": c.feedback,
            }
            for c in chef.dataset.corrections
        ]

        with chef.buffer.lock:
            start_index = chef.buffer.last_learn_index
            buffered = chef.buffer.examples[start_index:].copy()

        for offset, ex in enumerate(buffered):
            index = start_index + offset
            output = _as_dict(getattr(ex, "output", {}))
            metadata = _as_dict(getattr(ex, "metadata", {}))
            if getattr(ex, "is_correction", False):
                feedback = metadata.get("feedback")
                if isinstance(feedback, str):
                    feedback = feedback.strip() or None

                corrections.append(
                    {
                        "id": f"buffer-correction:{index}",
                        "input": _as_dict(getattr(ex, "input", {})),
                        "model_output": _as_dict(output.get("actual", {})),
                        "expected_output": _as_dict(output.get("expected", {})),
                        "feedback": feedback,
                    }
                )
            else:
                examples.append(
                    {
                        "id": f"buffer:{index}",
                        "input": _as_dict(getattr(ex, "input", {})),
                        "expected_output": output,
                        "source": f"buffer_{getattr(ex, 'source', 'human')}",
                    }
                )

    return {"examples": examples, "corrections": corrections}
