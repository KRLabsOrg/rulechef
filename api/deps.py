from __future__ import annotations

from typing import Annotated

from fastapi import Cookie, Response
from openai import OpenAI

from api.config import settings
from api.security import (
    decode_session_cookie,
    encode_session_cookie,
    generate_session_id,
)
from api.state import SessionState, sessions
from rulechef import RuleChef, Task
from rulechef.core import TaskType


def _make_client() -> OpenAI:
    key = (settings.OPENAI_API_KEY or "demo-key").strip()
    kwargs: dict = {"api_key": key}
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    return OpenAI(**kwargs)


def _ensure_default_project(state: SessionState) -> None:
    """Auto-provision a default NER project for the session."""
    if state.configured:
        return

    task = Task(
        name="RuleChef Demo",
        description=(
            "Extract named entities from text. "
            "Prefer labels that are present in user examples/corrections."
        ),
        input_schema={"text": "str"},
        output_schema={"entities": "List[{text: str, start: int, end: int, type: str}]"},
        type=TaskType.NER,
        text_field="text",
    )

    chef = RuleChef(
        task,
        _make_client(),
        dataset_name=f"workspace_{state.workspace_id}_default",
        storage_path=settings.STORAGE_PATH,
        model=settings.OPENAI_MODEL,
        use_grex=True,
    )

    state.task = task
    state.chef = chef


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=encode_session_cookie(session_id, settings.SESSION_SECRET),
        max_age=settings.SESSION_TTL_SECONDS,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        path=settings.SESSION_COOKIE_PATH,
    )


def get_session(
    response: Response,
    session_cookie: Annotated[
        str | None,
        Cookie(alias=settings.SESSION_COOKIE_NAME),
    ] = None,
) -> SessionState:
    """Resolve session from signed cookie; create one if missing/invalid."""
    session_id = decode_session_cookie(session_cookie, settings.SESSION_SECRET)

    if session_id is None:
        session_id = generate_session_id()

    _set_session_cookie(response, session_id)

    state = sessions.get(session_id)
    _ensure_default_project(state)
    return state
