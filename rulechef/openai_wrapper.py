"""OpenAI protocol wrapper for observing LLM interactions.

Supports three modes:
- Auto mode (task=None): raw capture → LLM discovers schema → LLM maps observations
- Mapped mode (task provided): raw capture → LLM maps to known schema
- Custom extractor mode (task + extractors): user functions parse directly
"""

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rulechef.buffer import ExampleBuffer
from rulechef.core import Task


@dataclass
class RawObservation:
    """A captured LLM call before schema mapping.

    response_content is normalized to a plain string at capture time
    because the OpenAI response object is ephemeral.
    """

    messages: list[dict[str, Any]]
    response_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def _extract_response_content(response) -> str | None:
    """Normalize an OpenAI response to a plain string.

    Handles plain text, tool_calls, and Pydantic structured output.
    Returns None if content cannot be extracted (caller should skip).
    """
    try:
        msg = response.choices[0].message
    except (AttributeError, IndexError):
        return None

    # Pydantic structured output (response_format=SomeModel)
    if hasattr(msg, "parsed") and msg.parsed is not None:
        try:
            return msg.parsed.model_dump_json()
        except Exception:
            return str(msg.parsed)

    # Tool calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        calls = []
        for tc in msg.tool_calls:
            calls.append(
                {
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )
        return json.dumps(calls)

    # Plain text (preserve empty string — use `is not None`, not truthiness)
    if msg.content is not None:
        return msg.content

    return None


class _StreamWrapper:
    """Wraps an OpenAI Stream to capture content after streaming completes.

    Passes through every chunk to the caller unchanged. When the stream
    finishes (StopIteration, context manager exit, or close()), the
    accumulated content is stored as a RawObservation.
    """

    def __init__(self, stream, observer, api_kwargs: dict):
        self._stream = stream
        self._observer = observer
        self._api_kwargs = api_kwargs
        self._content_parts: list = []
        self._tool_call_parts: dict = {}
        self._finalized = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._accumulate(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        self._finalize()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return False

    def close(self):
        self._finalize()
        if hasattr(self._stream, "close"):
            self._stream.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def _accumulate(self, chunk):
        try:
            if not chunk.choices:
                return
            delta = chunk.choices[0].delta

            if hasattr(delta, "content") and delta.content:
                self._content_parts.append(delta.content)

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in self._tool_call_parts:
                        self._tool_call_parts[idx] = {
                            "name": "",
                            "arguments": "",
                        }
                    if hasattr(tc, "function") and tc.function:
                        if tc.function.name:
                            self._tool_call_parts[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            self._tool_call_parts[idx]["arguments"] += tc.function.arguments
        except (AttributeError, IndexError):
            pass

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True

        content = None
        if self._content_parts:
            content = "".join(self._content_parts)
        elif self._tool_call_parts:
            calls = []
            for idx in sorted(self._tool_call_parts):
                tc = self._tool_call_parts[idx]
                calls.append({"function": tc["name"], "arguments": tc["arguments"]})
            content = json.dumps(calls)

        if content is None:
            self._observer._skipped_count += 1
            return

        # Custom extractor mode: can't apply extractors to a stream,
        # so store as raw observation for later mapping
        obs = RawObservation(
            messages=list(self._api_kwargs.get("messages", [])),
            response_content=content,
            metadata={
                "model": self._api_kwargs.get("model"),
                "temperature": self._api_kwargs.get("temperature"),
                "streamed": True,
            },
        )
        with self._observer._lock:
            self._observer._raw_observations.append(obs)
        self._observer._streaming_captured_count += 1


class OpenAIObserver:
    """Wraps OpenAI-compatible clients to observe calls and extract training data.

    In auto/mapped modes, raw observations are stored and mapped to the task
    schema later (at learn_rules() time) via LLM calls. In custom extractor
    mode, user functions parse input/output immediately with zero overhead.
    """

    def __init__(
        self,
        buffer: ExampleBuffer,
        task: Task | None,
        original_create: Callable,
        extract_input: Callable | None = None,
        extract_output: Callable | None = None,
        min_observations_for_discovery: int = 5,
        training_logger=None,
    ):
        """
        Args:
            buffer: ExampleBuffer to store mapped examples.
            task: Task definition (None for auto-discovery mode).
            original_create: The unwrapped chat.completions.create method.
                Must be saved by the engine BEFORE monkey-patching.
            extract_input: Custom function (api_kwargs → input dict). Optional.
            extract_output: Custom function (response → output dict). Optional.
            min_observations_for_discovery: Minimum raw observations needed
                before discover_task() can run.
            training_logger: Optional TrainingDataLogger for capturing LLM calls.
        """
        self.buffer = buffer
        self.task = task
        self._original_create = original_create
        self._client = None
        self.training_logger = training_logger

        # Mode detection
        self._custom_extractors = extract_input is not None and extract_output is not None
        self._extract_input_fn = extract_input
        self._extract_output_fn = extract_output

        # Raw observation store (auto/mapped modes)
        self._raw_observations: list[RawObservation] = []
        self._mapped_index: int = 0
        self._lock = threading.Lock()

        # Self-observation prevention
        self._skip: bool = False

        self.min_observations_for_discovery = min_observations_for_discovery

        # Stats
        self._skipped_count: int = 0
        self._failed_count: int = 0
        self._streaming_captured_count: int = 0

    def attach(self, client):
        """Monkey-patch client.chat.completions.create to observe calls.

        Args:
            client: OpenAI client (or compatible).

        Returns:
            The same client object (mutated in place).
        """
        self._client = client
        observer = self  # closure reference

        def observed_create(*args, **kwargs):
            # Self-observation guard
            if observer._skip:
                return observer._original_create(*args, **kwargs)

            # Call original
            response = observer._original_create(*args, **kwargs)

            # Streaming: wrap the iterator to capture after completion
            if kwargs.get("stream", False):
                return _StreamWrapper(response, observer, kwargs)

            # Non-streaming: capture immediately (never break user's code)
            try:
                observer._capture(kwargs, response)
            except Exception as e:
                print(f"Warning: RuleChef observation failed: {e}")
                observer._failed_count += 1

            return response

        client.chat.completions.create = observed_create
        return client

    def detach(self):
        """Restore original create method."""
        if self._client and self._original_create:
            self._client.chat.completions.create = self._original_create
        self._client = None

    def _capture(self, api_kwargs: dict, response) -> None:
        """Process an observed call.

        Custom extractor mode: parse immediately, add to buffer.
        Auto/mapped mode: store as RawObservation for later mapping.
        """
        if self._custom_extractors:
            input_data = self._extract_input_fn(api_kwargs)
            output_data = self._extract_output_fn(response)
            if input_data and output_data:
                self.buffer.add_llm_observation(
                    input_data,
                    output_data,
                    metadata={
                        "model": api_kwargs.get("model"),
                        "temperature": api_kwargs.get("temperature"),
                    },
                )
            return

        # Auto/mapped mode: store raw observation
        content = _extract_response_content(response)
        if content is None:
            self._skipped_count += 1
            return

        obs = RawObservation(
            messages=list(api_kwargs.get("messages", [])),
            response_content=content,
            metadata={
                "model": api_kwargs.get("model"),
                "temperature": api_kwargs.get("temperature"),
            },
        )
        with self._lock:
            self._raw_observations.append(obs)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_task(self, llm_client, model: str) -> Task:
        """Discover task schema from accumulated raw observations via LLM.

        Args:
            llm_client: Client for the discovery LLM call (should be unwrapped).
            model: Model name.

        Returns:
            A Task instance with discovered schema.

        Raises:
            ValueError: If not enough observations or LLM returns invalid JSON.
        """
        with self._lock:
            observations = self._raw_observations[:20]

        if len(observations) < self.min_observations_for_discovery:
            raise ValueError(
                f"Need at least {self.min_observations_for_discovery} observations "
                f"for task discovery (have {len(observations)})."
            )

        obs_text = self._format_observations_for_prompt(observations)

        prompt = f"""You are analyzing LLM API calls to discover the underlying task pattern.

Here are {len(observations)} sample LLM interactions:

{obs_text}

Based on these interactions, identify:
1. What task is being performed? (name and description)
2. What type of task? Choose ONE: extraction, ner, classification, transformation
   - extraction: finding text spans (untyped)
   - ner: finding typed entities with labels
   - classification: assigning a label to input text
   - transformation: extracting structured fields from text
3. What are the input fields and their types?
4. What are the output fields and their types?
5. Which input field contains the main text?

Return ONLY valid JSON:
{{
  "name": "task_name",
  "description": "one sentence description",
  "type": "classification",
  "input_schema": {{"text": "str"}},
  "output_schema": {{"label": "str"}},
  "text_field": "text"
}}"""

        self._skip = True
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = response.choices[0].message.content
        finally:
            self._skip = False

        task_dict = self._parse_json(raw, "discovery")

        if self.training_logger:
            self.training_logger.log(
                "task_discovery",
                [{"role": "user", "content": prompt}],
                raw,
                {
                    "num_observations": len(observations),
                    "discovered_task_name": task_dict.get("name"),
                    "discovered_task_type": task_dict.get("type"),
                },
            )

        return Task.from_dict(task_dict)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def map_pending(self, task: Task, llm_client, model: str) -> int:
        """Map all unmapped raw observations to task schema via LLM.

        Uses batch LLM calls (up to 10 observations per call) for efficiency.

        Args:
            task: The task schema to map against.
            llm_client: Client for mapping LLM calls (should be unwrapped).
            model: Model name.

        Returns:
            Count of observations successfully added to buffer.
        """
        with self._lock:
            to_map = self._raw_observations[self._mapped_index :]
            start_index = self._mapped_index

        if not to_map:
            return 0

        added = 0
        batch_size = 10

        for batch_start in range(0, len(to_map), batch_size):
            batch = to_map[batch_start : batch_start + batch_size]
            try:
                results = self._map_batch(task, batch, llm_client, model)
                for obs, result in zip(batch, results):
                    if result.get("relevant", False):
                        inp = result.get("input")
                        out = result.get("output")
                        if inp is not None and out is not None:
                            self.buffer.add_llm_observation(inp, out, metadata=obs.metadata)
                            added += 1
                        else:
                            self._skipped_count += 1
                    else:
                        self._skipped_count += 1
            except Exception as e:
                print(f"Warning: Batch mapping failed: {e}")
                self._failed_count += len(batch)

        # Advance cursor (don't re-map on next call)
        with self._lock:
            self._mapped_index = start_index + len(to_map)

        return added

    def _map_batch(
        self,
        task: Task,
        batch: list[RawObservation],
        llm_client,
        model: str,
    ) -> list[dict]:
        """Map a batch of observations via one LLM call.

        Returns a list of dicts: [{relevant, input, output}, ...].
        Length matches len(batch).
        """
        obs_text = self._format_observations_for_prompt(batch)
        task_info = task.to_dict()

        prompt = f"""You are extracting structured training data from LLM interactions.

Task definition:
  Name: {task_info["name"]}
  Description: {task_info["description"]}
  Type: {task_info["type"]}
  Input schema: {json.dumps(task_info["input_schema"])}
  Output schema: {json.dumps(task_info["output_schema"])}

Here are {len(batch)} LLM interactions to analyze:

{obs_text}

For EACH interaction, determine:
1. Is it relevant to the task above? (relevant: true/false)
2. If relevant, extract the input (matching input_schema keys) and output (matching output_schema keys).

Return ONLY a JSON array with exactly {len(batch)} objects:
[
  {{"relevant": true, "input": {{...}}, "output": {{...}}}},
  {{"relevant": false, "input": null, "output": null}},
  ...
]"""

        self._skip = True
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = response.choices[0].message.content
        finally:
            self._skip = False

        results = self._parse_json(raw, "mapping")

        if not isinstance(results, list):
            raise ValueError(f"Expected JSON array, got {type(results).__name__}")

        # Pad or truncate to match batch size
        while len(results) < len(batch):
            results.append({"relevant": False, "input": None, "output": None})
        results = results[: len(batch)]

        if self.training_logger:
            relevant_count = sum(1 for r in results if r.get("relevant", False))
            self.training_logger.log(
                "observation_mapping",
                [{"role": "user", "content": prompt}],
                raw,
                {
                    "task_name": task.name if task else None,
                    "task_type": task.type.value if task else None,
                    "batch_size": len(batch),
                    "relevant_count": relevant_count,
                },
            )

        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        """Get observation statistics."""
        with self._lock:
            total_raw = len(self._raw_observations)
            mapped = self._mapped_index
            pending = total_raw - mapped
        return {
            "observed": total_raw,
            "mapped": mapped,
            "pending": pending,
            "skipped": self._skipped_count,
            "failed": self._failed_count,
            "streaming_captured": self._streaming_captured_count,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_observations_for_prompt(
        self,
        observations: list[RawObservation],
        max_content_chars: int = 500,
    ) -> str:
        """Format observations into readable prompt text."""
        lines = []
        for i, obs in enumerate(observations, 1):
            lines.append(f"--- Interaction {i} ---")
            for msg in obs.messages:
                role = msg.get("role", "?")
                content = str(msg.get("content", ""))
                if len(content) > max_content_chars:
                    content = content[:max_content_chars] + "...[truncated]"
                lines.append(f"[{role}]: {content}")
            resp = obs.response_content
            if len(resp) > max_content_chars:
                resp = resp[:max_content_chars] + "...[truncated]"
            lines.append(f"Response: {resp}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(raw: str, context: str) -> Any:
        """Parse JSON from LLM response, handling markdown code blocks."""
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError) as e:
            raise ValueError(
                f"{context} LLM returned invalid JSON: {e}\nResponse: {raw[:500]}"
            ) from e
