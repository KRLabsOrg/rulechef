from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_session
from api.state import SessionState

router = APIRouter(prefix="/api/rules", tags=["rules"])


def _rule_sort_key(rule) -> tuple:
    name_key = rule.name.casefold() if rule.name else ""
    return (-rule.priority, name_key, rule.id or "")


@router.get("")
async def list_rules(state: Annotated[SessionState, Depends(get_session)]):
    async with state.write_lock:
        rules = sorted(state.chef.dataset.rules, key=_rule_sort_key)
        payload = []
        for rule in rules:
            success_rate = (rule.successes / rule.times_applied) if rule.times_applied > 0 else 0.0
            payload.append(
                {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "format": rule.format.value,
                    "priority": rule.priority,
                    "confidence": rule.confidence,
                    "times_applied": rule.times_applied,
                    "success_rate": success_rate,
                    "content": rule.content,
                }
            )

    return {"rules": payload}


@router.get("/metrics")
async def rule_metrics(state: Annotated[SessionState, Depends(get_session)]):
    async with state.write_lock:
        metrics = state.chef.get_rule_metrics(verbose=False)
    return {"rule_metrics": [m.to_dict() for m in metrics]}


@router.delete("/{rule_id}")
async def delete_rule(
    rule_id: str,
    state: Annotated[SessionState, Depends(get_session)],
):
    async with state.write_lock:
        deleted = state.chef.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(404, f"Rule {rule_id} not found")
    return {"ok": True, "message": f"Deleted rule {rule_id}"}
