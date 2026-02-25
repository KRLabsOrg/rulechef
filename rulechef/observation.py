"""LLM observation lifecycle and background auto-learning."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from rulechef.gliner_wrapper import GlinerObserver
from rulechef.openai_wrapper import OpenAIObserver, RawObservation

if TYPE_CHECKING:
    from rulechef.engine import RuleChef


class ObservationManager:
    """Manages LLM observation, raw observation staging, and background learning."""

    def __init__(self, chef: RuleChef):
        self._chef = chef
        self._observer: OpenAIObserver | None = None
        self._gliner_observer: GlinerObserver | None = None
        self._pending_raw_observations: list[RawObservation] = []
        self._learning_thread: threading.Thread | None = None
        self._stop_learning = threading.Event()

    def start_observing(
        self,
        openai_client,
        auto_learn: bool = True,
        check_interval: int = 60,
        extract_input: Callable | None = None,
        extract_output: Callable | None = None,
        min_observations_for_discovery: int = 5,
    ):
        """
        Start observing OpenAI-compatible client calls to collect training data.

        Works in three modes:
        - Auto mode (task=None): raw capture, schema discovered at learn_rules() time
        - Mapped mode (task provided): raw capture, LLM maps to schema at learn_rules()
        - Custom extractor mode (task + extractors): immediate parsing, zero overhead

        Args:
            openai_client: OpenAI client (or compatible API).
            auto_learn: If True, automatically triggers learning when coordinator decides.
            check_interval: Seconds between coordinator checks (default 60).
            extract_input: Custom function (api_kwargs -> input dict). Optional.
            extract_output: Custom function (response -> output dict). Optional.
            min_observations_for_discovery: Min raw observations before auto-discovery
                can run (default 5). Only relevant when task=None.

        Returns:
            The same client (monkey-patched in place).

        Example:
            chef = RuleChef(client=client)  # No task needed
            wrapped = chef.start_observing(client, auto_learn=False)
            # Use wrapped as normal - RuleChef captures calls
            response = wrapped.chat.completions.create(...)
            chef.learn_rules()  # Discovers task + maps + learns
        """
        # Save original create BEFORE patching — critical for:
        # 1. Self-observation prevention (internal calls use this)
        # 2. Discovery/mapping LLM calls use this
        original_create = openai_client.chat.completions.create

        self._observer = OpenAIObserver(
            buffer=self._chef.buffer,
            task=self._chef.task,
            original_create=original_create,
            extract_input=extract_input,
            extract_output=extract_output,
            min_observations_for_discovery=min_observations_for_discovery,
            training_logger=self._chef.training_logger,
        )

        self._observer.attach(openai_client)

        if auto_learn:
            self._start_learning_loop(check_interval)
            print(f"✓ Started observing with auto-learning (check every {check_interval}s)")
        else:
            mode = "auto-discover" if self._chef.task is None else "mapped"
            print(f"✓ Started observing ({mode} mode, manual learning)")

        return openai_client

    def discover_task(self):
        """Discover the task schema from accumulated raw observations.

        Uses LLM to analyze captured LLM interactions and infer the task type,
        input/output schema, and text field. Requires at least
        min_observations_for_discovery raw observations (from start_observing()
        or add_raw_observation()).

        Returns:
            The discovered Task instance (also stored as self.task).

        Raises:
            RuntimeError: If no raw observations available or using custom extractors.
            ValueError: If not enough observations or LLM returns invalid JSON.
        """
        # Merge any pending raw observations
        self._merge_pending_into_observer()

        if self._observer is None:
            raise RuntimeError(
                "No raw observations available. Call start_observing() or "
                "add_raw_observation() first."
            )
        if self._observer._custom_extractors:
            raise RuntimeError(
                "Cannot discover task in custom extractor mode — "
                "task must be provided with custom extractors."
            )

        self._observer._skip = True
        try:
            task = self._observer.discover_task(self._chef.llm, self._chef.model)
        finally:
            self._observer._skip = False

        self._chef._initialize_with_task(task)
        self._observer.task = task

        print(f"✓ Discovered task: {task.name}")
        print(f"  Type: {task.type.value}")
        print(f"  Input: {task.input_schema}")
        print(f"  Output: {task.output_schema}")
        if task.text_field:
            print(f"  Text field: {task.text_field}")

        return task

    def start_observing_gliner(
        self,
        gliner_model,
        method: str | None = None,
        auto_learn: bool = True,
        check_interval: int = 60,
    ):
        """Monkey-patch a GLiNER/GLiNER2 model to observe predictions.

        Args:
            gliner_model: A GLiNER or GLiNER2 model instance.
            method: Which method to observe (required for GLiNER2 non-NER tasks).
            auto_learn: If True, trigger learning automatically.
            check_interval: Seconds between coordinator checks.

        Returns:
            The same model (monkey-patched in place).
        """
        self._gliner_observer = GlinerObserver(
            buffer=self._chef.buffer, task=self._chef.task, method=method
        )
        self._gliner_observer.attach(gliner_model)

        kind = self._gliner_observer._task_kind
        method_name = self._gliner_observer._method_name
        if auto_learn:
            self._start_learning_loop(check_interval)
            print(f"✓ Observing GLiNER ({kind} via {method_name})")
        else:
            print(f"✓ Observing GLiNER ({kind} via {method_name}, manual learning)")

        return gliner_model

    def stop_observing_gliner(self):
        """Detach GLiNER observer."""
        if self._gliner_observer:
            self._gliner_observer.detach()
            self._gliner_observer = None

    def stop_observing(self):
        """Stop observing LLM calls and background learning."""
        # Stop background thread
        if self._learning_thread:
            self._stop_learning.set()
            self._learning_thread.join(timeout=5)
            self._learning_thread = None
            self._stop_learning.clear()

        # Detach observers
        if self._observer:
            self._observer.detach()
            self._observer = None
        if self._gliner_observer:
            self._gliner_observer.detach()
            self._gliner_observer = None

        print("✓ Stopped observing")

    def add_raw_observation(
        self,
        messages: list[dict],
        response: str,
        metadata: dict | None = None,
    ):
        """Add a raw LLM interaction for auto-discovery. Works with any LLM.

        Use this when you don't know the task schema yet. Pass the raw
        messages and response text — RuleChef will analyze these at
        learn_rules() time to discover the task type, input/output schema,
        and extract structured training data.

        Args:
            messages: List of message dicts
                (e.g. [{"role": "user", "content": "..."}]).
            response: The LLM's response as a plain string.
            metadata: Optional metadata (e.g. {"model": "gpt-4o"}).
        """
        self._pending_raw_observations.append(
            RawObservation(
                messages=list(messages),
                response_content=response,
                metadata=metadata or {},
            )
        )
        total = len(self._pending_raw_observations)
        if self._observer:
            total += self._observer.get_stats()["observed"]
        print(f"✓ Added raw observation ({total} total)")

    def get_buffer_stats(self) -> dict:
        """Get statistics about buffered examples and observations."""
        stats = {
            **self._chef.buffer.get_stats(),
            "coordinator_analysis": self._chef.coordinator.analyze_buffer(self._chef.buffer),
        }
        if self._observer:
            stats["observer"] = self._observer.get_stats()
        if self._gliner_observer:
            stats["gliner_observer"] = self._gliner_observer.get_stats()
        if self._pending_raw_observations:
            stats["pending_raw_observations"] = len(self._pending_raw_observations)
        return stats

    def check_and_trigger_learning(self):
        """
        Check coordinator and trigger learning if ready.

        Called after add_example() or add_correction() when auto_trigger=True.
        """
        current_rules = self._chef.dataset.rules if self._chef.dataset else []
        decision = self._chef.coordinator.should_trigger_learning(self._chef.buffer, current_rules)

        if decision.should_learn:
            print(f"\n{'=' * 60}")
            print(f"Auto-triggering learning: {decision.reasoning}")
            print(f"{'=' * 60}")
            self._auto_learn(decision)

    def trigger_manual_learning(self):
        """Manually trigger learning from buffered examples."""
        current_rules = self._chef.dataset.rules if self._chef.dataset else []
        decision = self._chef.coordinator.should_trigger_learning(self._chef.buffer, current_rules)

        if decision.should_learn:
            print(f"✓ Triggering learning: {decision.reasoning}")
            self._auto_learn(decision)
            return True
        else:
            print(f"✗ Not ready to learn: {decision.reasoning}")
            return False

    def merge_pending_into_observer(self):
        """Merge pending raw observations into the observer (public for pipeline)."""
        self._merge_pending_into_observer()

    def _merge_pending_into_observer(self):
        """Merge pending raw observations into the observer."""
        if self._pending_raw_observations:
            if self._observer is None:
                self._observer = OpenAIObserver(
                    buffer=self._chef.buffer,
                    task=self._chef.task,
                    original_create=self._chef.llm.chat.completions.create,
                    training_logger=self._chef.training_logger,
                )
            with self._observer._lock:
                self._observer._raw_observations.extend(self._pending_raw_observations)
            self._pending_raw_observations.clear()

    def _start_learning_loop(self, interval: int):
        """Background thread that periodically checks if learning should trigger."""

        def loop():
            while not self._stop_learning.is_set():
                try:
                    should_learn = False
                    reason = ""

                    # In auto/mapped mode, check raw observation count
                    if self._observer and not self._observer._custom_extractors:
                        obs_stats = self._observer.get_stats()
                        pending = obs_stats["pending"]
                        threshold = self._observer.min_observations_for_discovery
                        if pending >= threshold:
                            should_learn = True
                            reason = f"{pending} raw observations ready"
                    else:
                        # Custom extractors or no observer — check buffer
                        current_rules = self._chef.dataset.rules if self._chef.dataset else []
                        decision = self._chef.coordinator.should_trigger_learning(
                            self._chef.buffer, current_rules
                        )
                        should_learn = decision.should_learn
                        reason = decision.reasoning

                    if should_learn:
                        print(f"\n{'=' * 60}")
                        print(f"Auto-triggering learning: {reason}")
                        print(f"{'=' * 60}")
                        self._auto_learn(
                            decision
                            if not self._observer or self._observer._custom_extractors
                            else None
                        )

                except Exception as e:
                    print(f"Error in learning loop: {e}")

                # Wait for next check
                self._stop_learning.wait(interval)

        self._learning_thread = threading.Thread(target=loop, daemon=True)
        self._learning_thread.start()

    def _auto_learn(self, decision=None):
        """Execute learning based on coordinator decision."""
        old_rules = (
            self._chef.dataset.rules.copy()
            if self._chef.dataset and self._chef.dataset.rules
            else None
        )

        try:
            kwargs = {}
            if decision is not None:
                kwargs["sampling_strategy"] = decision.strategy
                kwargs["max_refinement_iterations"] = decision.max_iterations
            result = self._chef.learn_rules(**kwargs)
            if result:
                rules, eval_result = result
                self._chef.coordinator.on_learning_complete(old_rules, rules, eval_result)

        except Exception as e:
            print(f"Error during auto-learning: {e}")
