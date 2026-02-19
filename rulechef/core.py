"""Core data structures for RuleChef"""

from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Callable,
    Union,
    Type,
    Tuple,
    Literal,
    get_args,
    get_origin,
)
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ValidationError

# Type alias for output matcher functions
# Takes (expected_output, actual_output) -> bool
OutputMatcher = Callable[[Dict[str, Any], Dict[str, Any]], bool]

# Type for output schema - can be dict or Pydantic model
OutputSchema = Union[Dict[str, Any], Type[BaseModel]]


def get_labels_from_model(
    model: Type[BaseModel], field_name: str = "type"
) -> List[str]:
    """
    Extract Literal values from a Pydantic model.

    Looks for a field with the given name (default "type") that has a Literal type annotation.
    Handles nested models (e.g., List[Entity] -> Entity -> type field).
    """
    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation

        # Handle Optional types
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            annotation = args[0] if args else annotation
            origin = get_origin(annotation)

        # Handle List[Entity] -> recurse into Entity
        if origin is list:
            args = get_args(annotation)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                result = get_labels_from_model(args[0], field_name)
                if result:
                    return result

        # Handle nested BaseModel
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            result = get_labels_from_model(annotation, field_name)
            if result:
                return result

        # Found the target field with Literal type
        if name == field_name and get_origin(annotation) is Literal:
            return list(get_args(annotation))

    return []


def is_pydantic_schema(schema: OutputSchema) -> bool:
    """Check if schema is a Pydantic model class"""
    return isinstance(schema, type) and issubclass(schema, BaseModel)


class RuleFormat(Enum):
    """Rule representation formats"""

    REGEX = "regex"
    CODE = "code"
    SPACY = "spacy"  # spaCy token matcher patterns


@dataclass
class Span:
    """A text span with character-level position information.

    Attributes:
        text: The matched text content.
        start: Start character offset (inclusive) in the source string.
        end: End character offset (exclusive) in the source string.
        score: Confidence score for the match, between 0.0 and 1.0.
    """

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
    """Type of task being performed.

    EXTRACTION:      Find text spans (untyped). Output: {"spans": [{"text", "start", "end"}]}
    NER:             Find typed entities.       Output: {"entities": [{"text", "start", "end", "type"}]}
    CLASSIFICATION:  Classify input into a label. Output: {"label": "class_name"}
    TRANSFORMATION:  Extract structured data to a custom output schema you define.
                     Like GLiNER — define {"company": "str", "amount": "str"} and get exactly that.
    """

    EXTRACTION = "extraction"
    NER = "ner"
    CLASSIFICATION = "classification"
    TRANSFORMATION = "transformation"


# Canonical output keys per task type
DEFAULT_OUTPUT_KEYS = {
    TaskType.EXTRACTION: "spans",
    TaskType.NER: "entities",
    TaskType.CLASSIFICATION: "label",
    TaskType.TRANSFORMATION: None,  # Uses output_schema keys
}


@dataclass
class Task:
    """
    Abstract task definition.
    Describes what we're trying to accomplish.

    Attributes:
        name: Task name
        description: Free text description
        input_schema: Dict describing input fields
        output_schema: Dict or Pydantic model describing output fields.
            - Dict: Simple string descriptions (e.g., {"spans": "List[Span]"})
            - Pydantic model: Full type validation with Literal labels
        type: TaskType enum (EXTRACTION, NER, CLASSIFICATION, TRANSFORMATION)
        output_matcher: Optional custom function to compare outputs.
                       Signature: (expected: Dict, actual: Dict) -> bool
                       If not provided, uses default matcher for the task type.
        matching_mode: For extraction tasks, choose "text" (default) or "exact"
                       to control how span matches are evaluated.
        text_field: Optional input key to use for regex/spaCy matching. If not set,
                    the longest string field is used.
    """

    name: str
    description: str
    input_schema: Dict[str, str]
    output_schema: OutputSchema
    type: TaskType = TaskType.EXTRACTION
    output_matcher: Optional[OutputMatcher] = None
    matching_mode: Literal["text", "exact"] = "text"
    text_field: Optional[str] = None

    def get_labels(self, field_name: str = "type") -> List[str]:
        """
        Get label values from output schema.

        For Pydantic schemas, extracts Literal values from the specified field.
        For dict schemas, returns empty list (labels not defined).
        """
        if is_pydantic_schema(self.output_schema):
            return get_labels_from_model(self.output_schema, field_name)
        return []

    def validate_output(self, output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate output against schema.

        For Pydantic schemas, uses model validation.
        For dict schemas, returns (True, []) - no validation.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        if not is_pydantic_schema(self.output_schema):
            return (True, [])

        try:
            self.output_schema.model_validate(output)
            return (True, [])
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            return (False, errors)

    def get_schema_for_prompt(self) -> str:
        """
        Render schema for inclusion in LLM prompts.

        For Pydantic schemas, generates a readable representation with descriptions.
        For dict schemas, returns the dict as a string.
        """
        if not is_pydantic_schema(self.output_schema):
            return str(self.output_schema)

        # Use Pydantic's JSON schema with descriptions
        schema = self.output_schema.model_json_schema()
        return _format_json_schema_for_prompt(schema)

    def to_dict(self) -> dict:
        # For Pydantic schemas, store the JSON schema representation
        if is_pydantic_schema(self.output_schema):
            output_schema_dict = self.output_schema.model_json_schema()
        else:
            output_schema_dict = self.output_schema

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": output_schema_dict,
            "type": self.type.value,
            "matching_mode": self.matching_mode,
            "text_field": self.text_field,
            # Note: output_matcher is not serializable, so not included
        }


def _format_json_schema_for_prompt(schema: Dict[str, Any], indent: int = 0) -> str:
    """Format a JSON schema into a readable string for LLM prompts"""
    lines = []
    prefix = "  " * indent

    if "properties" in schema:
        for prop_name, prop_schema in schema.get("properties", {}).items():
            prop_type = prop_schema.get("type", "any")
            description = prop_schema.get("description", "")

            # Handle arrays
            if prop_type == "array" and "items" in prop_schema:
                items = prop_schema["items"]
                if "$ref" in items:
                    # Reference to another definition
                    ref_name = items["$ref"].split("/")[-1]
                    lines.append(f"{prefix}{prop_name}: List[{ref_name}]")
                    if description:
                        lines.append(f"{prefix}  # {description}")
                else:
                    lines.append(f"{prefix}{prop_name}: List[...]")

            # Handle enums (Literal types)
            elif "enum" in prop_schema:
                values = prop_schema["enum"]
                lines.append(f"{prefix}{prop_name}: one of {values}")
                if description:
                    lines.append(f"{prefix}  # {description}")

            else:
                line = f"{prefix}{prop_name}: {prop_type}"
                if description:
                    line += f"  # {description}"
                lines.append(line)

    # Handle definitions (nested models)
    if "$defs" in schema:
        for def_name, def_schema in schema.get("$defs", {}).items():
            lines.append(f"\n{prefix}{def_name}:")
            lines.append(_format_json_schema_for_prompt(def_schema, indent + 1))

    return "\n".join(lines)


@dataclass
class Example:
    """Regular training example. Lower priority than corrections.

    Attributes:
        id: Unique identifier.
        input: Input data dict matching the task's input_schema.
        expected_output: Expected output dict matching the task's output_schema.
        source: Origin of the example ('human_labeled' or 'llm_generated').
        confidence: Confidence score for this example (0.0-1.0).
        timestamp: When the example was created.
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
    """User correction -- the highest value training signal.

    Contains both the wrong output and the correct output so the learner
    can understand what to fix.

    Attributes:
        id: Unique identifier.
        input: Input data dict that was processed.
        model_output: The incorrect output that was produced.
        expected_output: The correct output the model should have produced.
        feedback: Optional free-text explanation of what went wrong.
        timestamp: When the correction was created.
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
class Feedback:
    """User feedback at any level: task, example, or rule.

    - task: general guidance ("drugs usually follow dosage like 'mg'")
    - example: feedback on a specific training item
    - rule: feedback on a specific rule ("too broad", "too specific")

    Attributes:
        id: Unique identifier.
        text: The feedback text.
        level: Feedback scope -- 'task', 'example', or 'rule'.
        target_id: Empty for task-level; example_id or rule_id otherwise.
        timestamp: When the feedback was created.
    """

    id: str
    text: str
    level: str  # "task" | "example" | "rule"
    target_id: str = ""  # empty for task-level, example_id or rule_id otherwise
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "level": self.level,
            "target_id": self.target_id,
        }


@dataclass
class Rule:
    """Learned extraction rule.

    For schema-aware rules (NER, TRANSFORMATION), use output_template and
    output_key to control how matches are mapped to structured output.
    For legacy rules (EXTRACTION), content holds the pattern directly.

    Attributes:
        id: Unique identifier.
        name: Human-readable rule name (used for merge-by-name in patching).
        description: What this rule matches or does.
        format: Rule format (REGEX, CODE, or SPACY).
        content: Pattern string (regex, code, or JSON-encoded spaCy pattern).
            Also accessible via the ``pattern`` property.
        priority: Execution priority (1-10, higher runs first).
        confidence: Confidence score (0.0-1.0), adjusted based on success rate.
        times_applied: Total number of times this rule has been applied.
        successes: Number of successful applications.
        failures: Number of failed applications.
        created_at: When the rule was created.
        output_template: JSON template for each match, using variables like
            $0, $1, $start, $end, $ent_type. None for plain span extraction.
        output_key: Which key in the output dict to populate (e.g. 'entities').
            Inferred from task type if not set.
    """

    id: str
    name: str
    description: str
    format: RuleFormat
    content: str  # Pattern string (kept for backward compat)
    priority: int = 5
    confidence: float = 0.5
    times_applied: int = 0
    successes: int = 0
    failures: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    # Schema-aware rule fields (optional, for NER/TRANSFORMATION)
    output_template: Optional[Dict[str, Any]] = None  # Template for output JSON
    output_key: Optional[str] = None  # Which output key to populate (e.g., "entities")

    @property
    def pattern(self) -> str:
        """Alias for content - clearer semantics for regex/spaCy patterns"""
        return self.content

    @pattern.setter
    def pattern(self, value: str):
        self.content = value

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
        result = {
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
        # Include schema-aware fields if set
        if self.output_template is not None:
            result["output_template"] = self.output_template
        if self.output_key is not None:
            result["output_key"] = self.output_key
        return result


@dataclass
class Dataset:
    """Complete training dataset containing examples, corrections, feedback, and rules.

    Attributes:
        name: Dataset name, used as the persistence filename.
        task: Task definition describing the extraction/classification goal.
        description: Optional human-readable description.
        examples: List of labeled training examples.
        corrections: List of user corrections (highest-value training signal).
        feedback: Legacy list of plain-text feedback strings (task-level only).
        structured_feedback: Structured feedback entries at task/example/rule level.
        rules: Learned rules (populated by learn_rules).
        version: Dataset schema version for forward compatibility.
    """

    name: str
    task: Task
    description: str = ""
    examples: List[Example] = field(default_factory=list)
    corrections: List[Correction] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)
    structured_feedback: List[Feedback] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    version: int = 1

    def get_all_training_data(self) -> List:
        """Get all examples and corrections combined"""
        return self.corrections + self.examples

    def get_feedback_for(self, level: str, target_id: str = "") -> List[Feedback]:
        """Get feedback filtered by level and optional target."""
        return [
            f
            for f in self.structured_feedback
            if f.level == level and (not target_id or f.target_id == target_id)
        ]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": self.task.to_dict(),
            "description": self.description,
            "examples": [e.to_dict() for e in self.examples],
            "corrections": [c.to_dict() for c in self.corrections],
            "feedback": self.feedback,
            "structured_feedback": [f.to_dict() for f in self.structured_feedback],
            "rules": [r.to_dict() for r in self.rules],
            "version": self.version,
        }
