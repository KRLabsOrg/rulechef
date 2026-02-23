from __future__ import annotations

import hashlib
import hmac
import re
import uuid

SID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def generate_session_id() -> str:
    return uuid.uuid4().hex


def _sign_session_id(session_id: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), session_id.encode("utf-8"), hashlib.sha256).hexdigest()


def encode_session_cookie(session_id: str, secret: str) -> str:
    signature = _sign_session_id(session_id, secret)
    return f"{session_id}.{signature}"


def decode_session_cookie(cookie_value: str | None, secret: str) -> str | None:
    if not cookie_value or "." not in cookie_value:
        return None

    session_id, provided_sig = cookie_value.rsplit(".", 1)
    if not SID_PATTERN.fullmatch(session_id):
        return None

    expected_sig = _sign_session_id(session_id, secret)
    if not hmac.compare_digest(provided_sig, expected_sig):
        return None

    return session_id
