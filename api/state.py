from __future__ import annotations

import asyncio
import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field

from api.config import settings
from rulechef import RuleChef, Task

SESSION_TTL = settings.SESSION_TTL_SECONDS


@dataclass
class LearningStatus:
    running: bool = False
    progress: str = ""
    error: str | None = None
    last_metrics: dict | None = None


@dataclass
class SessionState:
    session_id: str
    workspace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    chef: RuleChef | None = None
    task: Task | None = None
    draft_examples: list[dict] = field(default_factory=list)
    allowed_formats: list[str] = field(default_factory=lambda: ["regex", "code"])
    learning: LearningStatus = field(default_factory=LearningStatus)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    learning_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def configured(self) -> bool:
        return self.chef is not None

    def touch(self) -> None:
        self.last_accessed = time.time()


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                # Derive workspace_id deterministically so data survives server restarts.
                # session_id is a 32-char hex UUID stored in the signed cookie — same
                # cookie always maps to the same workspace_id and therefore the same
                # dataset file on disk.
                workspace_id = hashlib.sha256(session_id.encode()).hexdigest()[:10]
                self._sessions[session_id] = SessionState(
                    session_id=session_id,
                    workspace_id=workspace_id,
                )
            state = self._sessions[session_id]
            state.touch()
            return state

    def cleanup(self) -> int:
        with self._lock:
            now = time.time()
            expired = [
                sid
                for sid, state in self._sessions.items()
                if now - state.last_accessed > SESSION_TTL and not state.learning.running
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


sessions = SessionStore()
