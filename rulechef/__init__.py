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
]


def __getattr__(name: str):
    # Lazy import to avoid importing heavy dependencies (e.g. openai) unless needed.
    if name == "RuleChef":
        from rulechef.engine import RuleChef

        return RuleChef
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
