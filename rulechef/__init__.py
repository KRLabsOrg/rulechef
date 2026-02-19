"""RuleChef - Learn rule-based models from examples and LLM interactions"""

from rulechef.core import (
    Task,
    Dataset,
    Example,
    Correction,
    Feedback,
    Span,
    Rule,
    TaskType,
    RuleFormat,
)
from rulechef.evaluation import EvalResult, RuleMetrics, ClassMetrics
from rulechef.coordinator import (
    CoordinatorProtocol,
    SimpleCoordinator,
    AuditResult,
    AuditAction,
)

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
]


def __getattr__(name: str):
    # Lazy imports to avoid importing heavy dependencies (e.g. openai) unless needed.
    if name == "RuleChef":
        from rulechef.engine import RuleChef

        return RuleChef
    if name == "AgenticCoordinator":
        from rulechef.coordinator import AgenticCoordinator

        return AgenticCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
