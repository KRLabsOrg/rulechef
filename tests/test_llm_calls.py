from unittest.mock import MagicMock

import pytest

from rulechef.llm_calls import LLMCallConfig, LLMCallManager, PromptTooLargeError, PromptVariant


def _mock_client(response_text: str = '{"ok": true}'):
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = response
    return client


def test_output_token_kwarg_names():
    client = _mock_client()

    manager = LLMCallManager(
        client, "test-model", LLMCallConfig(output_token_param="max_completion_tokens")
    )
    assert manager.output_kwargs(123) == {"max_completion_tokens": 123}

    manager = LLMCallManager(client, "test-model", LLMCallConfig(output_token_param="max_tokens"))
    assert manager.output_kwargs(123) == {"max_tokens": 123}

    manager = LLMCallManager(
        client, "test-model", LLMCallConfig(output_token_param="max_new_tokens")
    )
    assert manager.output_kwargs(123) == {"max_new_tokens": 123}

    manager = LLMCallManager(client, "test-model", LLMCallConfig(output_token_param=None))
    assert manager.output_kwargs(123) == {}


def test_context_window_none_selects_first_variant():
    manager = LLMCallManager(_mock_client(), "test-model", LLMCallConfig(context_window=None))
    variants = [
        PromptVariant(name="full", prompt="x" * 10000),
        PromptVariant(name="compact", prompt="small"),
    ]

    variant, _, metadata = manager.select_variant(variants, output_tokens=8192)

    assert variant.name == "full"
    assert metadata["context_window"] is None


def test_selects_first_variant_that_fits_budget():
    manager = LLMCallManager(
        _mock_client(),
        "test-model",
        LLMCallConfig(context_window=1000, safety_margin_tokens=0),
    )
    variants = [
        PromptVariant(name="too_large", prompt="x" * 5000),
        PromptVariant(name="fits", prompt="small"),
    ]

    variant, _, metadata = manager.select_variant(variants, output_tokens=100)

    assert variant.name == "fits"
    assert metadata["fits_context"] is True
    assert metadata["variant_estimates"][0]["fits_context"] is False


def test_raises_when_no_variant_fits():
    manager = LLMCallManager(
        _mock_client(),
        "test-model",
        LLMCallConfig(context_window=10, safety_margin_tokens=0),
    )

    with pytest.raises(PromptTooLargeError):
        manager.select_variant([PromptVariant(name="large", prompt="x" * 1000)], output_tokens=100)


def test_complete_with_variants_uses_configured_provider_kwargs():
    client = _mock_client()
    manager = LLMCallManager(
        client,
        "test-model",
        LLMCallConfig(output_token_param="max_tokens", response_format_json=False),
    )

    result = manager.complete_with_variants(
        [PromptVariant(name="full", prompt="hello")],
        output_tokens=77,
        extra_kwargs={"temperature": 0},
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert result.response_text == '{"ok": true}'
    assert kwargs["model"] == "test-model"
    assert kwargs["max_tokens"] == 77
    assert kwargs["temperature"] == 0
    assert "max_completion_tokens" not in kwargs
    assert "response_format" not in kwargs
