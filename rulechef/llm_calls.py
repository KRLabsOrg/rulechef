"""Shared LLM call helpers with provider-agnostic prompt budgeting."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class LLMCallConfig(BaseModel):
    """Configuration for OpenAI-compatible LLM calls."""

    context_window: int | None = None
    output_token_param: str | None = "max_completion_tokens"
    synthesis_output_tokens: int = 16384
    patch_output_tokens: int = 8192
    default_output_tokens: int | None = None
    safety_margin_tokens: int = 512
    response_format_json: bool = True

    @field_validator(
        "context_window",
        "synthesis_output_tokens",
        "patch_output_tokens",
        "default_output_tokens",
        "safety_margin_tokens",
    )
    @classmethod
    def _positive_or_none(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("token budget values must be non-negative")
        return value

    @field_validator("output_token_param")
    @classmethod
    def _non_empty_param(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("output_token_param must be a non-empty string or None")
        return value


class PromptVariant(BaseModel):
    """One possible prompt representation for a call."""

    name: str
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMCallResult(BaseModel):
    """Text response plus prompt-budget metadata."""

    response_text: str
    variant: PromptVariant
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]


class PromptTooLargeError(ValueError):
    """Raised when no prompt variant fits the configured context window."""


class LLMCallManager:
    """Thin OpenAI-compatible call wrapper with prompt budgeting."""

    def __init__(self, client: Any, model: str, config: LLMCallConfig | None = None):
        self.client = client
        self.model = model
        self.config = config or LLMCallConfig()

    def estimate_text_tokens(self, text: str) -> int:
        """Conservative provider-agnostic token estimate.

        We intentionally avoid provider-specific tokenizers here. UTF-8 bytes / 3
        overestimates English text enough for a safety guard while keeping the
        dependency surface small.
        """
        if not text:
            return 0
        return max(1, (len(text.encode("utf-8")) + 2) // 3)

    def estimate_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            total += 4  # small chat-format overhead
            total += self.estimate_text_tokens(str(message.get("role", "")))
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            total += self.estimate_text_tokens(str(content))
        return total

    def resolve_output_tokens(self, output_tokens: int | None) -> int | None:
        return output_tokens if output_tokens is not None else self.config.default_output_tokens

    def output_kwargs(self, output_tokens: int | None) -> dict[str, int]:
        resolved = self.resolve_output_tokens(output_tokens)
        if resolved is None or self.config.output_token_param is None:
            return {}
        return {self.config.output_token_param: resolved}

    def fits(self, messages: list[dict[str, Any]], output_tokens: int | None) -> bool:
        if self.config.context_window is None:
            return True
        reserved = self.resolve_output_tokens(output_tokens) or 0
        return (
            self.estimate_messages_tokens(messages) + reserved + self.config.safety_margin_tokens
            <= self.config.context_window
        )

    def select_variant(
        self, variants: list[PromptVariant], output_tokens: int | None
    ) -> tuple[PromptVariant, list[dict[str, Any]], dict[str, Any]]:
        if not variants:
            raise PromptTooLargeError("No prompt variants were provided")

        estimates = []
        resolved_output = self.resolve_output_tokens(output_tokens)
        for variant in variants:
            messages = [{"role": "user", "content": variant.prompt}]
            estimated_input = self.estimate_messages_tokens(messages)
            total = estimated_input + (resolved_output or 0) + self.config.safety_margin_tokens
            metadata = {
                "variant_name": variant.name,
                "estimated_input_tokens": estimated_input,
                "reserved_output_tokens": resolved_output,
                "safety_margin_tokens": self.config.safety_margin_tokens,
                "context_window": self.config.context_window,
                "estimated_total_tokens": total,
                "fits_context": self.config.context_window is None
                or total <= self.config.context_window,
            }
            estimates.append(metadata)
            if metadata["fits_context"]:
                return variant, messages, {**metadata, "variant_estimates": estimates}

        raise PromptTooLargeError(
            "No prompt variant fits the configured context window "
            f"({self.config.context_window} tokens); smallest estimate was "
            f"{min(e['estimated_total_tokens'] for e in estimates)} tokens"
        )

    def complete_with_variants(
        self,
        variants: list[PromptVariant],
        output_tokens: int | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> LLMCallResult:
        variant, messages, metadata = self.select_variant(variants, output_tokens)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.output_kwargs(output_tokens),
            **(extra_kwargs or {}),
        }
        if self.config.response_format_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content
        return LLMCallResult(
            response_text=response_text,
            variant=variant,
            messages=messages,
            metadata=metadata,
        )
