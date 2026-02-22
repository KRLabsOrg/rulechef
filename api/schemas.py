from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# --- Project ---


class ConfigureRequest(BaseModel):
    task_name: str
    task_description: str
    task_type: str = "ner"  # ner | extraction | classification | transformation
    input_schema: dict[str, str] = {"text": "str"}
    output_schema: dict[str, str] = {
        "entities": "List[{text: str, start: int, end: int, type: str}]"
    }
    entity_labels: list[str] = []
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    model: str = "moonshotai/kimi-k2-instruct-0905"
    text_field: str = "text"
    allowed_formats: list[str] = ["regex", "code"]  # "regex" | "code" | "spacy"


class StatusResponse(BaseModel):
    configured: bool
    task_name: str | None = None
    task_type: str | None = None
    stats: dict | None = None


# --- Data ---


class AddExampleRequest(BaseModel):
    input_data: dict[str, Any]
    output_data: dict[str, Any]


class AddCorrectionRequest(BaseModel):
    input_data: dict[str, Any]
    model_output: dict[str, Any]
    expected_output: dict[str, Any]
    feedback: str | None = None


class AddRawLinesRequest(BaseModel):
    lines: list[str]


class AnnotateExampleRequest(BaseModel):
    output_data: dict[str, Any]


class UploadResponse(BaseModel):
    count: int
    message: str


# --- Learning ---


class LearnRequest(BaseModel):
    sampling_strategy: str | None = None
    max_iterations: int = 3
    incremental_only: bool = True
    use_agentic: bool = False


class LearnStatusResponse(BaseModel):
    running: bool
    progress: str = ""
    error: str | None = None
    metrics: dict | None = None


# --- Extraction ---


class ExtractRequest(BaseModel):
    input_data: dict[str, Any]


class ExtractResponse(BaseModel):
    result: dict[str, Any]


class BatchExtractRequest(BaseModel):
    inputs: list[dict[str, Any]]


class BatchExtractResponse(BaseModel):
    results: list[dict[str, Any]]


# --- Rules ---


class RuleSummary(BaseModel):
    id: str
    name: str
    description: str
    format: str
    priority: int
    confidence: float
    times_applied: int
    success_rate: float
    content: str | None = None


# --- Feedback ---


class AddFeedbackRequest(BaseModel):
    text: str
    level: str = "task"  # "task" | "example" | "rule"
    target_id: str = ""


class FeedbackItem(BaseModel):
    id: str
    text: str
    level: str
    target_id: str
    timestamp: str
