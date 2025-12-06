"""Core data structures for RuleChef"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class RuleFormat(Enum):
    """Rule representation formats"""

    REGEX = "regex"
    CODE = "code"
    SPACY = "spacy"  # spaCy token matcher patterns


@dataclass
class Span:
    """A text span with position"""

    text: str
    start: int
    end: int
    score: float = 1.0

    def overlaps(self, other: "Span") -> bool:
        """Check if spans overlap"""
        return not (self.end <= other.start or self.start >= other.end)

    def overlap_ratio(self, other: "Span") -> float:
        """Calculate overlap ratio (IoU)"""
        if not self.overlaps(other):
            return 0.0
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        overlap_len = overlap_end - overlap_start
        union_len = max(self.end, other.end) - min(self.start, other.start)
        return overlap_len / union_len if union_len > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }


class TaskType(Enum):
    """Type of task being performed"""

    EXTRACTION = "extraction"  # Returns List[Span]
    CLASSIFICATION = "classification"  # Returns str (Label)
    TRANSFORMATION = "transformation"  # Returns Any (JSON/String)


@dataclass
class Task:
    """
    Abstract task definition
    Describes what we're trying to accomplish
    """

    name: str
    description: str  # Free text description
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    type: TaskType = TaskType.EXTRACTION

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "type": self.type.value,
        }


@dataclass
class Example:
    """
    Regular training example
    Lower priority than corrections
    """

    id: str
    input: Dict[str, Any]
    expected_output: Dict[str, Any]
    source: str  # "human_labeled" | "llm_generated"
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input": self.input,
            "expected_output": self.expected_output,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class Correction:
    """
    User correction - HIGHEST value signal
    Contains both wrong output and correct output
    """

    id: str
    input: Dict[str, Any]
    model_output: Dict[str, Any]  # What was WRONG
    expected_output: Dict[str, Any]  # What it SHOULD be
    feedback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input": self.input,
            "model_output": self.model_output,
            "expected_output": self.expected_output,
            "feedback": self.feedback,
        }


@dataclass
class Rule:
    """Learned extraction rule"""

    id: str
    name: str
    description: str
    format: RuleFormat
    content: str
    priority: int = 5
    confidence: float = 0.5
    times_applied: int = 0
    successes: int = 0
    failures: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def update_stats(self, success: bool):
        """Update performance stats and adjust confidence"""
        self.times_applied += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

        # Adjust confidence based on success rate
        if self.times_applied > 0:
            success_rate = self.successes / self.times_applied
            self.confidence = 0.3 + (success_rate * 0.7)  # Range: 0.3 to 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "format": self.format.value,
            "content": self.content,
            "priority": self.priority,
            "confidence": self.confidence,
            "times_applied": self.times_applied,
            "successes": self.successes,
            "failures": self.failures,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Dataset:
    """Complete training dataset"""

    name: str
    task: Task
    description: str = ""
    examples: List[Example] = field(default_factory=list)
    corrections: List[Correction] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    version: int = 1

    def get_all_training_data(self) -> List:
        """Get all examples and corrections combined"""
        return self.corrections + self.examples

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": self.task.to_dict(),
            "description": self.description,
            "examples": [e.to_dict() for e in self.examples],
            "corrections": [c.to_dict() for c in self.corrections],
            "feedback": self.feedback,
            "rules": [r.to_dict() for r in self.rules],
            "version": self.version,
        }
