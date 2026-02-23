"""RuleChef - Learn rule-based models from examples and LLM interactions"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rulechef.coordinator import (
    AuditAction,
    AuditResult,
    CoordinatorProtocol,
    SimpleCoordinator,
)
from rulechef.core import (
    Correction,
    Dataset,
    Example,
    Feedback,
    Rule,
    RuleFormat,
    Span,
    Task,
    TaskType,
)
from rulechef.evaluation import ClassMetrics, EvalResult, RuleMetrics

if TYPE_CHECKING:
    from rulechef.coordinator import AgenticCoordinator as AgenticCoordinator
    from rulechef.engine import RuleChef as RuleChef
    from rulechef.training_logger import TrainingDataLogger as TrainingDataLogger

__version__ = "0.1.0"
__all__ = [
    "RuleChef",
    "Task",
    "Dataset",
    "Example",
    "Correction",
    "Feedback",
    "Span",
    "Rule",
    "TaskType",
    "RuleFormat",
    "EvalResult",
    "RuleMetrics",
    "ClassMetrics",
    "CoordinatorProtocol",
    "SimpleCoordinator",
    "AgenticCoordinator",
    "AuditResult",
    "AuditAction",
    "TrainingDataLogger",
]


def __getattr__(name: str):
    # Lazy imports to avoid importing heavy dependencies (e.g. openai) unless needed.
    if name == "RuleChef":
        from rulechef.engine import RuleChef

        return RuleChef
    if name == "AgenticCoordinator":
        from rulechef.coordinator import AgenticCoordinator

        return AgenticCoordinator
    if name == "TrainingDataLogger":
        from rulechef.training_logger import TrainingDataLogger

        return TrainingDataLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
