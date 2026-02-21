"""Main RuleChef orchestrator"""

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI

from rulechef.buffer import ExampleBuffer
from rulechef.coordinator import (
    AgenticCoordinator,
    CoordinatorProtocol,
    SimpleCoordinator,
)
from rulechef.core import (
    DEFAULT_OUTPUT_KEYS,
    Correction,
    Dataset,
    Example,
    Feedback,
    Rule,
    RuleFormat,
    Task,
    TaskType,
)
from rulechef.evaluation import (
    EvalResult,
    RuleMetrics,
    evaluate_dataset,
    evaluate_rules_individually,
    print_eval_result,
    print_rule_metrics,
)
from rulechef.learner import RuleLearner
from rulechef.openai_wrapper import OpenAIObserver


class RuleChef:
    """Main orchestrator for learning and applying rules"""

    def __init__(
        self,
        task: Task | None = None,
        client: OpenAI | None = None,
        dataset_name: str = "default",
        storage_path: str = "./rulechef_data",
        allowed_formats: list[RuleFormat] | None = None,
        sampling_strategy: str = "balanced",
        coordinator: CoordinatorProtocol | None = None,
        auto_trigger: bool = False,
        model: str = "gpt-4o-mini",
        llm_fallback: bool = False,
        use_spacy_ner: bool = False,
        use_grex: bool = True,
        max_rules: int = 10,
        max_samples: int = 50,
        synthesis_strategy: str = "auto",
        training_logger=None,
    ):
        """Initialize a RuleChef instance.

        Args:
            task: Task definition describing what to extract/classify.
                Optional — if None, use start_observing() and the task will
                be auto-discovered from observed LLM calls.
            client: OpenAI client instance. Creates a default client if not provided.
            dataset_name: Name for the dataset, used as the filename for persistence.
            storage_path: Directory path for saving/loading dataset JSON files.
            allowed_formats: Rule formats to allow (e.g. REGEX, CODE, SPACY).
                Defaults to [REGEX, CODE].
            sampling_strategy: How to sample training data for prompts.
                Options: 'balanced', 'recent', 'diversity', 'uncertain', 'varied'.
            coordinator: CoordinatorProtocol implementation for learning decisions.
                Defaults to SimpleCoordinator.
            auto_trigger: If True, check coordinator after each add_example/add_correction
                and trigger learning automatically when ready.
            model: OpenAI model name for LLM calls.
            llm_fallback: If True, fall back to direct LLM extraction when rules
                produce no results.
            use_spacy_ner: If True, enable spaCy NER entity recognition during
                rule execution (requires spaCy and a model).
            use_grex: If True, use grex for regex pattern suggestion in prompts.
            max_rules: Maximum number of rules to generate per synthesis call.
            max_samples: Maximum training examples to include in synthesis prompts.
            synthesis_strategy: Strategy for multi-class synthesis.
                'auto' uses per-class when multiple classes detected, 'per_class'
                always uses per-class, any other value uses bulk synthesis.
            training_logger: Optional TrainingDataLogger instance for capturing
                all LLM calls as training data for model distillation.
        """
        self.task = task
        self.llm = client or OpenAI()
        self.model = model
        self.llm_fallback = llm_fallback
        self.use_spacy_ner = use_spacy_ner
        self.use_grex = use_grex
        self.storage_path = Path(storage_path)
        self.sampling_strategy = sampling_strategy
        self.synthesis_strategy = synthesis_strategy
        self.training_logger = training_logger

        # Save constructor args for lazy initialization (when task=None)
        self._dataset_name = dataset_name
        self._allowed_formats_raw = allowed_formats
        self._max_rules = max_rules
        self._max_samples = max_samples

        # Coordinator for learning decisions (swappable simple/agentic)
        self.coordinator = coordinator or SimpleCoordinator()

        # Propagate training logger to coordinator if it's agentic
        if self.training_logger and isinstance(self.coordinator, AgenticCoordinator):
            self.coordinator.training_logger = self.training_logger

        # Buffer for observed examples (buffer-first architecture)
        self.buffer = ExampleBuffer()

        # Auto-trigger: coordinator checks after each add_example/add_correction
        self.auto_trigger = auto_trigger

        # Observation mode components
        self._observer: OpenAIObserver | None = None
        self._pending_raw_observations: list = []
        self._learning_thread: threading.Thread | None = None
        self._stop_learning = threading.Event()

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Dataset and learner: created now if task is provided, lazily otherwise
        if task is not None:
            self._initialize_with_task(task)
        else:
            self.dataset = None
            self.learner = None
            self.allowed_formats = None

    def _initialize_with_task(self, task: Task) -> None:
        """Create dataset and learner from task.

        Called from __init__ when task is provided, or after auto-discovery.
        """
        self.task = task

        # Convert string format names to RuleFormat enums
        if self._allowed_formats_raw:
            self.allowed_formats = [
                RuleFormat(f) if isinstance(f, str) else f
                for f in self._allowed_formats_raw
            ]
        else:
            self.allowed_formats = [RuleFormat.REGEX, RuleFormat.CODE]

        self.dataset = Dataset(name=self._dataset_name, task=task)
        self.learner = RuleLearner(
            self.llm,
            allowed_formats=self.allowed_formats,
            sampling_strategy=self.sampling_strategy,
            model=self.model,
            use_spacy_ner=self.use_spacy_ner,
            use_grex=self.use_grex,
            max_rules=self._max_rules,
            max_samples=self._max_samples,
            training_logger=self.training_logger,
        )

        # Load existing dataset if on disk
        self._load_dataset()

    def _require_task(self, method_name: str) -> None:
        """Raise if task has not been set yet."""
        if self.task is None:
            raise RuntimeError(
                f"Cannot call {method_name}() before a task is defined. "
                "Either pass task= to RuleChef(), call discover_task(), "
                "or call learn_rules() after start_observing() which triggers "
                "auto-discovery."
            )

    # ========================================
    # Data Collection
    # ========================================

    def add_example(
        self, input_data: dict, output_data: dict, source: str = "human_labeled"
    ):
        """Add a labeled training example.

        Uses buffer-first architecture: example goes to buffer, then coordinator
        decides when to trigger learning.

        Args:
            input_data: Input dict matching the task's input_schema.
            output_data: Expected output dict matching the task's output_schema.
            source: Origin of the example, e.g. 'human_labeled' or 'llm_generated'.
        """
        self._require_task("add_example")
        # Add to buffer (not dataset directly)
        self.buffer.add_human_example(input_data, output_data)

        stats = self.buffer.get_stats()
        print(
            f"✓ Added example (buffer: {stats['new_examples']} new, {stats['total_examples']} total)"
        )

        # If auto-trigger enabled, check coordinator
        if self.auto_trigger:
            self._check_and_trigger_learning()

    def add_correction(
        self,
        input_data: dict,
        model_output: dict,
        expected_output: dict,
        feedback: str | None = None,
    ):
        """Add a user correction (high value signal).

        Uses buffer-first architecture: correction goes to buffer, then coordinator
        decides when to trigger learning. Corrections are high-priority signals.

        Args:
            input_data: Input dict that was processed.
            model_output: The incorrect output that was produced.
            expected_output: The correct output the model should have produced.
            feedback: Optional free-text explanation of what went wrong.
        """
        self._require_task("add_correction")
        # Add to buffer (not dataset directly)
        self.buffer.add_human_correction(
            input_data,
            expected_output,
            model_output,
            feedback=feedback,
        )

        stats = self.buffer.get_stats()
        print(
            f"✓ Added correction (buffer: {stats['new_corrections']} corrections, {stats['new_examples']} new total)"
        )

        # If auto-trigger enabled, check coordinator (corrections are high-value!)
        if self.auto_trigger:
            self._check_and_trigger_learning()

    def add_feedback(
        self,
        feedback: str,
        level: str = "task",
        target_id: str = "",
    ):
        """
        Add user feedback at any level.

        Args:
            feedback: The feedback text
            level: "task" (general guidance), "example" (specific example),
                   or "rule" (specific rule)
            target_id: Required for "example" and "rule" levels — the id of
                       the example or rule this feedback applies to
        """
        self._require_task("add_feedback")
        fb = Feedback(
            id=self._generate_id(),
            text=feedback,
            level=level,
            target_id=target_id,
        )
        self.dataset.structured_feedback.append(fb)

        # Also keep legacy list for backward compat
        if level == "task":
            self.dataset.feedback.append(feedback)

        self._save_dataset()
        print(
            f"✓ Added {level}-level feedback"
            + (f" for {target_id}" if target_id else "")
        )

    def generate_llm_examples(self, num_examples: int = 5, seed: int = 42):
        """
        Generate synthetic training examples using LLM.

        Examples go to buffer and can trigger learning if auto_trigger=True.
        """
        self._require_task("generate_llm_examples")
        print(f"\n🤖 Generating {num_examples} examples with LLM...")
        for i in range(num_examples):
            input_data = self.learner.generate_synthetic_input(self.task, seed + i)
            # Add to buffer directly to avoid N coordinator checks
            self.buffer.add_llm_observation(
                input_data,
                {"spans": []},  # Empty output, just for training variety
                metadata={"generated": True, "seed": seed + i},
            )

        stats = self.buffer.get_stats()
        print(
            f"✓ Generated {num_examples} examples (buffer: {stats['new_examples']} new)"
        )

        # Check coordinator once after generating all
        if self.auto_trigger:
            self._check_and_trigger_learning()

    def add_observation(
        self,
        input_data: dict,
        output_data: dict,
        metadata: dict | None = None,
    ):
        """Add a structured LLM observation. Works with any LLM provider.

        Use this to feed RuleChef data from any source — Anthropic, Groq,
        local models, LangChain, etc. You extract the input/output yourself.

        Unlike add_example(), no task definition is required — observations
        can be collected before the task is known.

        Args:
            input_data: Input dict (e.g. {"text": "the query"}).
            output_data: Output dict (e.g. {"label": "the_class"}).
            metadata: Optional metadata (e.g. {"model": "claude-3"}).
        """
        self.buffer.add_llm_observation(
            input_data, output_data, metadata=metadata or {}
        )

        stats = self.buffer.get_stats()
        print(
            f"✓ Added observation (buffer: {stats['new_examples']} new, "
            f"{stats['total_examples']} total)"
        )

        if self.auto_trigger:
            self._check_and_trigger_learning()

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
        from rulechef.openai_wrapper import RawObservation

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

    # ========================================
    # Learning
    # ========================================

    def learn_rules(
        self,
        run_evaluation: bool | None = None,
        min_examples: int = 1,
        max_refinement_iterations: int = 3,
        sampling_strategy: str | None = None,
        incremental_only: bool = False,
    ):
        """Learn rules from all collected data.

        This is the core batch learning process. Buffered examples are first
        committed to the dataset, then rules are synthesized and optionally
        refined through an evaluate-and-patch loop.

        Args:
            run_evaluation: Whether to run evaluation/refinement loop.
                None (default) auto-enables if total_data >= 3. True always
                enables refinement. False disables it (faster, synthesis only).
            min_examples: Minimum training items required to proceed.
            max_refinement_iterations: Max iterations in refinement loop (1-3).
            sampling_strategy: Override default sampling strategy for this run.
                Options: 'balanced', 'recent', 'diversity', 'uncertain', 'varied'.
            incremental_only: If True and rules already exist, only generate
                patch rules for current failures instead of full re-synthesis.

        Returns:
            Optional[Tuple[List[Rule], Optional[EvalResult]]]: A tuple of
                (learned_rules, eval_result) on success. eval_result is None
                when refinement is disabled. Returns None if not enough data.
        """

        start_time = time.time()

        # Merge any manually-added raw observations into the observer
        if self._pending_raw_observations:
            if self._observer is None:
                self._observer = OpenAIObserver(
                    buffer=self.buffer,
                    task=self.task,
                    original_create=self.llm.chat.completions.create,
                )
            with self._observer._lock:
                self._observer._raw_observations.extend(self._pending_raw_observations)
            self._pending_raw_observations.clear()

        # OBSERVATION INTEGRATION: discover task and/or map raw observations
        if self._observer is not None:
            self._observer._skip = True
            try:
                # Auto-discover task if not yet defined
                if self.task is None:
                    obs_stats = self._observer.get_stats()
                    if obs_stats["observed"] == 0:
                        raise RuntimeError(
                            "No observations captured yet. Make some LLM calls "
                            "through the observed client before calling learn_rules()."
                        )
                    print("\n🔍 Auto-discovering task from observed LLM calls...")
                    task = self._observer.discover_task(self.llm, self.model)
                    self._initialize_with_task(task)
                    self._observer.task = task
                    print(f"✓ Discovered task: {task.name} ({task.type.value})")

                # Map raw observations to task schema (auto/mapped modes)
                if not self._observer._custom_extractors:
                    pending = self._observer.get_stats()["pending"]
                    if pending > 0:
                        print(
                            f"\n📋 Mapping {pending} raw observations to task schema..."
                        )
                        added = self._observer.map_pending(
                            self.task, self.llm, self.model
                        )
                        print(f"✓ Mapped {added} observations to buffer")
            finally:
                self._observer._skip = False

        # Guard: task must be defined by now (either provided or discovered)
        if self.task is None:
            raise RuntimeError(
                "Task not defined. Either pass task= to RuleChef(), "
                "call discover_task(), or use start_observing() to enable "
                "auto-discovery."
            )

        # FIRST: Convert any buffered examples to dataset
        buffer_stats = self.buffer.get_stats()
        if buffer_stats["new_examples"] > 0:
            print(
                f"\n📥 Converting {buffer_stats['new_examples']} buffered examples to dataset..."
            )
            print(
                f"   ({buffer_stats['new_corrections']} corrections, {buffer_stats['llm_observations']} LLM, {buffer_stats['human_examples']} human)"
            )

            for example in self.buffer.get_new_examples():
                if example.is_correction:
                    metadata = getattr(example, "metadata", {}) or {}
                    correction_feedback = metadata.get("feedback")
                    if isinstance(correction_feedback, str):
                        correction_feedback = correction_feedback.strip() or None

                    # Add as Correction to dataset
                    correction = Correction(
                        id=self._generate_id(),
                        input=example.input,
                        model_output=example.output.get("actual", {}),
                        expected_output=example.output.get("expected", example.output),
                        feedback=correction_feedback,
                    )
                    self.dataset.corrections.append(correction)
                else:
                    # Add as Example to dataset
                    ex = Example(
                        id=self._generate_id(),
                        input=example.input,
                        expected_output=example.output,
                        source=example.source,
                    )
                    self.dataset.examples.append(ex)

            # Mark buffer as processed
            self.buffer.mark_learned()

            # Save dataset with new examples
            self._save_dataset()

            print(
                f"✓ Converted to dataset: {len(self.dataset.corrections)} corrections, {len(self.dataset.examples)} examples"
            )

        total_data = len(self.dataset.get_all_training_data())

        if total_data < min_examples:
            print(f"Need at least {min_examples} examples/corrections")
            print(
                f"Currently have: {len(self.dataset.corrections)} corrections, "
                f"{len(self.dataset.examples)} examples"
            )
            return

        # Smart default: disable evaluation for tiny datasets
        if run_evaluation is None:
            run_evaluation = total_data >= 3

        print(f"\n{'=' * 60}")
        print(f"Learning rules from {total_data} training items")
        print(f"  Corrections: {len(self.dataset.corrections)} (high value)")
        print(f"  Examples: {len(self.dataset.examples)}")
        if run_evaluation:
            print(
                f"  Mode: Synthesis + Refinement (max {max_refinement_iterations} iterations)"
            )
        else:
            print("  Mode: Synthesis only (no refinement)")

        # Show sampling strategy
        strategy = sampling_strategy or self.sampling_strategy
        if strategy != "balanced":
            print(f"  Sampling: {strategy}")
        print(f"{'=' * 60}\n")

        # Temporarily override sampling strategy if provided
        original_strategy = self.learner.sampling_strategy
        if sampling_strategy:
            self.learner.sampling_strategy = sampling_strategy

        # Prevent self-observation during synthesis/refinement LLM calls
        if self._observer:
            self._observer._skip = True

        try:
            if incremental_only and self.dataset.rules:
                print("Incremental mode: patching existing rules")
                pre_eval = self.learner._evaluate_rules(
                    self.dataset.rules, self.dataset
                )

                # If there's feedback but no failures, create a synthetic
                # failure entry so the patch prompt still fires with feedback
                failures = pre_eval.failures
                has_feedback = len(self.dataset.structured_feedback) > 0
                if not failures and has_feedback:
                    print(
                        "  No failures found but feedback exists — "
                        "forcing refinement with feedback"
                    )
                    # Use a small sample of training data as context
                    sample = self.dataset.get_all_training_data()[:3]
                    failures = [
                        {
                            "input": item.input,
                            "expected": item.expected_output,
                            "got": item.expected_output,  # same — no actual error
                            "is_correction": False,
                            "feedback_driven": True,
                        }
                        for item in sample
                    ]

                patch_rules = self.learner.synthesize_patch_ruleset(
                    self.dataset.rules,
                    failures,
                    dataset=self.dataset,
                )
                rules = self._merge_rules(self.dataset.rules, patch_rules)
            else:
                # Choose synthesis method based on strategy
                use_per_class = False
                if self.synthesis_strategy == "per_class":
                    use_per_class = True
                elif self.synthesis_strategy == "auto":
                    classes = self.learner._get_classes(self.dataset)
                    use_per_class = len(classes) > 1

                if use_per_class:
                    rules = self.learner.synthesize_ruleset_per_class(self.dataset)
                else:
                    rules = self.learner.synthesize_ruleset(self.dataset)

                if not rules:
                    print("Failed to synthesize rules")
                    return None

                print(f"✓ Generated {len(rules)} rules")

            # Evaluate and refine (refinement uses failures to patch)
            if run_evaluation:
                rules, eval_result = self.learner.evaluate_and_refine(
                    rules,
                    self.dataset,
                    max_iterations=max_refinement_iterations,
                    coordinator=self.coordinator,
                )
            else:
                eval_result = None

            # Save learned rules
            self.dataset.rules = rules
            self._save_dataset()

            # Audit rules for redundancy/merging (coordinator-driven)
            audit = self.coordinator.audit_rules(
                rules, self.get_rule_metrics(verbose=False)
            )
            if audit.actions:
                rules = self._apply_audit(audit, eval_result)
                eval_result = self.evaluate(verbose=False) if eval_result else None

            elapsed = time.time() - start_time

            print(f"\n{'=' * 60}")
            print(f"Learning complete! ({elapsed:.1f}s)")
            print(f"  Rules: {len(rules)}")
            if eval_result and eval_result.total_docs > 0:
                print(f"  Exact match: {eval_result.exact_match:.1%}")
                print(
                    f"  Entity P/R/F1: {eval_result.micro_precision:.1%} / "
                    f"{eval_result.micro_recall:.1%} / {eval_result.micro_f1:.1%}"
                )
                for cm in eval_result.per_class:
                    print(
                        f"    {cm.label}: F1={cm.f1:.0%} "
                        f"(P={cm.precision:.0%} R={cm.recall:.0%})"
                    )
            print(f"{'=' * 60}\n")

            return rules, eval_result
        finally:
            # Re-enable observation
            if self._observer:
                self._observer._skip = False
            # Restore original sampling strategy
            if sampling_strategy:
                self.learner.sampling_strategy = original_strategy

    # ========================================
    # Execution
    # ========================================

    def extract(self, input_data: dict, validate: bool = True) -> dict:
        """Extract from input using learned rules.

        Works for all task types: EXTRACTION, NER, CLASSIFICATION, TRANSFORMATION.
        Returns empty result if no rules learned, unless llm_fallback=True.

        Args:
            input_data: Input data dict matching the task's input_schema.
            validate: If True and output_schema is Pydantic, validate output.

        Returns:
            Output dict whose shape depends on the task type:
                - EXTRACTION: {"spans": [{"text", "start", "end", ...}]}
                - NER: {"entities": [{"text", "start", "end", "type", ...}]}
                - CLASSIFICATION: {"label": "class_name"}
                - TRANSFORMATION: Dict with keys defined by output_schema.
            Returns an empty structure if no rules matched or none are learned.
        """
        self._require_task("extract")
        if not self.dataset.rules:
            # No rules learned yet
            if self.llm_fallback:
                return self._execute_with_llm(input_data)
            # Return empty result based on task type
            if self.task.type == TaskType.CLASSIFICATION:
                return {"label": ""}  # Classification expects string, not list
            default_key = DEFAULT_OUTPUT_KEYS.get(self.task.type, "spans")
            if default_key:
                return {default_key: []}
            return {}

        # Apply rules with task type for proper output key inference
        output = self.learner._apply_rules(
            self.dataset.rules, input_data, self.task.type, self.task.text_field
        )

        # Store current extraction for potential correction
        self.current_extraction = output

        result = output or {}

        # LLM fallback if rules produced empty result
        if not result and self.llm_fallback:
            return self._execute_with_llm(input_data)

        # Validate output against schema if enabled
        if validate and result:
            is_valid, errors = self.task.validate_output(result)
            if not is_valid:
                for error in errors:
                    print(f"   Schema validation warning: {error}")

        return result

    def _execute_with_llm(self, input_data: dict) -> dict:
        """Execute extraction using LLM directly (fallback when rules don't work)"""
        prompt = f"""Task: {self.task.name}
Description: {self.task.description}

Input: {json.dumps(input_data)}

Output schema: {self.task.get_schema_for_prompt()}

Return ONLY valid JSON matching the output schema, no explanation."""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            text = response.choices[0].message.content
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception as e:
            print(f"LLM fallback error: {e}")
            # Return empty result
            if self.task.type == TaskType.CLASSIFICATION:
                return {"label": ""}
            default_key = DEFAULT_OUTPUT_KEYS.get(self.task.type, "spans")
            if default_key:
                return {default_key: []}
            return {}

    # ========================================
    # Observation Mode (LLM Middleware)
    # ========================================

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
            extract_input: Custom function (api_kwargs → input dict). Optional.
            extract_output: Custom function (response → output dict). Optional.
            min_observations_for_discovery: Min raw observations before auto-discovery
                can run (default 5). Only relevant when task=None.

        Returns:
            The same client (monkey-patched in place).

        Example:
            chef = RuleChef(client=client)  # No task needed
            wrapped = chef.start_observing(client, auto_learn=False)
            # Use wrapped as normal — RuleChef captures calls
            response = wrapped.chat.completions.create(...)
            chef.learn_rules()  # Discovers task + maps + learns
        """
        # Save original create BEFORE patching — critical for:
        # 1. Self-observation prevention (internal calls use this)
        # 2. Discovery/mapping LLM calls use this
        original_create = openai_client.chat.completions.create

        self._observer = OpenAIObserver(
            buffer=self.buffer,
            task=self.task,
            original_create=original_create,
            extract_input=extract_input,
            extract_output=extract_output,
            min_observations_for_discovery=min_observations_for_discovery,
        )

        self._observer.attach(openai_client)

        if auto_learn:
            self._start_learning_loop(check_interval)
            print(
                f"✓ Started observing with auto-learning (check every {check_interval}s)"
            )
        else:
            mode = "auto-discover" if self.task is None else "mapped"
            print(f"✓ Started observing ({mode} mode, manual learning)")

        return openai_client

    def discover_task(self) -> Task:
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
        if self._pending_raw_observations:
            if self._observer is None:
                self._observer = OpenAIObserver(
                    buffer=self.buffer,
                    task=self.task,
                    original_create=self.llm.chat.completions.create,
                )
            with self._observer._lock:
                self._observer._raw_observations.extend(self._pending_raw_observations)
            self._pending_raw_observations.clear()

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
            task = self._observer.discover_task(self.llm, self.model)
        finally:
            self._observer._skip = False

        self._initialize_with_task(task)
        self._observer.task = task

        print(f"✓ Discovered task: {task.name}")
        print(f"  Type: {task.type.value}")
        print(f"  Input: {task.input_schema}")
        print(f"  Output: {task.output_schema}")
        if task.text_field:
            print(f"  Text field: {task.text_field}")

        return task

    def stop_observing(self):
        """Stop observing LLM calls and background learning"""
        # Stop background thread
        if self._learning_thread:
            self._stop_learning.set()
            self._learning_thread.join(timeout=5)
            self._learning_thread = None
            self._stop_learning.clear()

        # Detach observer
        if self._observer:
            self._observer.detach()
            self._observer = None

        print("✓ Stopped observing")

    def _start_learning_loop(self, interval: int):
        """Background thread that periodically checks if learning should trigger"""

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
                        current_rules = self.dataset.rules if self.dataset else []
                        decision = self.coordinator.should_trigger_learning(
                            self.buffer, current_rules
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
        """Execute learning based on coordinator decision"""
        old_rules = (
            self.dataset.rules.copy() if self.dataset and self.dataset.rules else None
        )

        try:
            kwargs = {}
            if decision is not None:
                kwargs["sampling_strategy"] = decision.strategy
                kwargs["max_refinement_iterations"] = decision.max_iterations
            result = self.learn_rules(**kwargs)
            if result:
                rules, eval_result = result
                self.coordinator.on_learning_complete(old_rules, rules, eval_result)

        except Exception as e:
            print(f"Error during auto-learning: {e}")

    def _check_and_trigger_learning(self):
        """
        Check coordinator and trigger learning if ready.

        Called after add_example() or add_correction() when auto_trigger=True.
        """
        current_rules = self.dataset.rules if self.dataset else []
        decision = self.coordinator.should_trigger_learning(self.buffer, current_rules)

        if decision.should_learn:
            print(f"\n{'=' * 60}")
            print(f"Auto-triggering learning: {decision.reasoning}")
            print(f"{'=' * 60}")
            self._auto_learn(decision)

    def trigger_manual_learning(self):
        """Manually trigger learning from buffered examples"""
        current_rules = self.dataset.rules if self.dataset else []
        decision = self.coordinator.should_trigger_learning(self.buffer, current_rules)

        if decision.should_learn:
            print(f"✓ Triggering learning: {decision.reasoning}")
            self._auto_learn(decision)
            return True
        else:
            print(f"✗ Not ready to learn: {decision.reasoning}")
            return False

    def get_buffer_stats(self) -> dict:
        """Get statistics about buffered examples and observations."""
        stats = {
            **self.buffer.get_stats(),
            "coordinator_analysis": self.coordinator.analyze_buffer(self.buffer),
        }
        if self._observer:
            stats["observer"] = self._observer.get_stats()
        if self._pending_raw_observations:
            stats["pending_raw_observations"] = len(self._pending_raw_observations)
        return stats

    # ========================================
    # Utils
    # ========================================

    def get_stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dict with keys: 'task' (str), 'dataset' (str), 'corrections' (int),
            'examples' (int), 'feedback' (int), 'rules' (int),
            'description' (str).
        """
        self._require_task("get_stats")
        return {
            "task": self.dataset.task.name,
            "dataset": self.dataset.name,
            "corrections": len(self.dataset.corrections),
            "examples": len(self.dataset.examples),
            "feedback": len(self.dataset.feedback),
            "rules": len(self.dataset.rules),
            "description": self.dataset.description,
        }

    def evaluate(self, verbose: bool = True) -> EvalResult:
        """Run full entity-level evaluation of current rules against the dataset.

        Args:
            verbose: If True, print a formatted evaluation summary to stdout.

        Returns:
            EvalResult with micro/macro P/R/F1, per-class breakdown,
            exact match rate, and failure details.
        """
        self._require_task("evaluate")
        if not self.dataset.rules:
            print("No rules to evaluate")
            return EvalResult()

        result = evaluate_dataset(
            self.dataset.rules,
            self.dataset,
            self.learner._apply_rules,
        )
        if verbose:
            print_eval_result(result, self.dataset.name)
        return result

    def get_rule_metrics(self, verbose: bool = True) -> list[RuleMetrics]:
        """Evaluate each rule individually against the dataset.

        Useful for identifying dead or harmful rules.

        Args:
            verbose: If True, print a formatted per-rule metrics table to stdout.

        Returns:
            List[RuleMetrics] with per-rule precision/recall/F1, sample matches,
            and per-class breakdown.
        """
        self._require_task("get_rule_metrics")
        if not self.dataset.rules:
            print("No rules to evaluate")
            return []

        metrics = evaluate_rules_individually(
            self.dataset.rules,
            self.dataset,
            self.learner._apply_rules,
        )
        if verbose:
            print_rule_metrics(metrics)
        return metrics

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by id.

        Args:
            rule_id: The unique identifier of the rule to delete.

        Returns:
            True if the rule was found and deleted, False otherwise.
        """
        self._require_task("delete_rule")
        before = len(self.dataset.rules)
        self.dataset.rules = [r for r in self.dataset.rules if r.id != rule_id]
        if len(self.dataset.rules) < before:
            self._save_dataset()
            print(f"✓ Deleted rule {rule_id}")
            return True
        print(f"Rule {rule_id} not found")
        return False

    def get_rules_summary(self) -> list[dict]:
        """Get formatted summary of learned rules.

        Returns:
            List of dicts sorted by priority (descending), each with keys:
            'name' (str), 'description' (str), 'format' (str),
            'priority' (int), 'confidence' (str), 'times_applied' (int),
            'success_rate' (str).
        """
        self._require_task("get_rules_summary")
        summaries = []
        for rule in sorted(self.dataset.rules, key=lambda r: r.priority, reverse=True):
            success_rate = (
                rule.successes / rule.times_applied * 100
                if rule.times_applied > 0
                else 0
            )
            summaries.append(
                {
                    "name": rule.name,
                    "description": rule.description,
                    "format": rule.format.value,
                    "priority": rule.priority,
                    "confidence": f"{rule.confidence:.2f}",
                    "times_applied": rule.times_applied,
                    "success_rate": f"{success_rate:.1f}%"
                    if rule.times_applied > 0
                    else "N/A",
                }
            )
        return summaries

    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid

        return str(uuid.uuid4())[:8]

    def _merge_rules(self, existing: list[Rule], patches: list[Rule]) -> list[Rule]:
        """Merge patch rules into existing set and prune weak rules."""
        by_name = {r.name: r for r in existing}

        for pr in patches:
            if pr.name in by_name:
                # Preserve observed stats to reduce churn
                current = by_name[pr.name]
                pr.times_applied = current.times_applied
                pr.successes = current.successes
                pr.failures = current.failures
                pr.confidence = current.confidence
            by_name[pr.name] = pr

        merged = list(by_name.values())

        # Simple prune: drop rules with repeated failures and no successes
        pruned = [
            r
            for r in merged
            if not (r.times_applied >= 3 and r.successes == 0 and r.failures >= 3)
        ]
        if len(pruned) < len(merged):
            print(f"🧹 Pruned {len(merged) - len(pruned)} weak rules")
        return pruned

    def _apply_audit(self, audit, eval_result) -> list[Rule]:
        """Apply audit actions (merge/remove) with F1 safety net."""
        import re as re_mod

        pre_audit_rules = list(self.dataset.rules)
        pre_f1 = eval_result.micro_f1 if eval_result else None
        rules_by_id = {r.id: r for r in self.dataset.rules}
        changed = False

        for action in audit.actions:
            if action.action == "merge" and len(action.rule_ids) >= 2:
                # Find the source rules
                sources = [
                    rules_by_id[rid] for rid in action.rule_ids if rid in rules_by_id
                ]
                if len(sources) < 2:
                    continue

                # Validate the merged pattern compiles
                if action.merged_pattern:
                    try:
                        re_mod.compile(action.merged_pattern)
                    except re_mod.error:
                        continue

                # Create merged rule from the highest-priority source
                base = max(sources, key=lambda r: r.priority)
                merged = Rule(
                    id=self._generate_id(),
                    name=action.merged_name or base.name,
                    description=f"Merged: {action.reason}",
                    format=base.format,
                    content=action.merged_pattern or base.content,
                    priority=base.priority,
                    output_template=base.output_template,
                    output_key=base.output_key,
                )

                # Remove sources, add merged
                self.dataset.rules = [
                    r for r in self.dataset.rules if r.id not in action.rule_ids
                ]
                self.dataset.rules.append(merged)
                rules_by_id = {r.id: r for r in self.dataset.rules}
                changed = True

            elif action.action == "remove":
                for rid in action.rule_ids:
                    if rid in rules_by_id:
                        self.dataset.rules = [
                            r for r in self.dataset.rules if r.id != rid
                        ]
                        del rules_by_id[rid]
                        changed = True

        if not changed:
            return self.dataset.rules

        # Safety net: revert if F1 dropped
        if pre_f1 is not None:
            post_eval = self.evaluate(verbose=False)
            if post_eval.micro_f1 < pre_f1 - 0.01:
                print(
                    f"⚠ Audit dropped F1 ({pre_f1:.2f} → "
                    f"{post_eval.micro_f1:.2f}), reverting"
                )
                self.dataset.rules = pre_audit_rules
                self._save_dataset()
                return self.dataset.rules

        before = len(pre_audit_rules)
        after = len(self.dataset.rules)
        if after != before:
            print(f"🧹 Audit: {before} → {after} rules")
        self._save_dataset()
        return self.dataset.rules

    # ========================================
    # Persistence
    # ========================================

    def _save_dataset(self):
        """Save dataset to disk"""
        filepath = self.storage_path / f"{self.dataset.name}.json"
        with open(filepath, "w") as f:
            json.dump(self.dataset.to_dict(), f, indent=2, default=str)

    def _load_dataset(self):
        """Load dataset from disk if it exists"""
        filepath = self.storage_path / f"{self.dataset.name}.json"

        if not filepath.exists():
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Restore examples
            for ex in data.get("examples", []):
                example = Example(
                    id=ex["id"],
                    input=ex["input"],
                    expected_output=ex["expected_output"],
                    source=ex["source"],
                    confidence=ex.get("confidence", 0.8),
                )
                self.dataset.examples.append(example)

            # Restore corrections
            for corr in data.get("corrections", []):
                correction = Correction(
                    id=corr["id"],
                    input=corr["input"],
                    model_output=corr["model_output"],
                    expected_output=corr["expected_output"],
                    feedback=corr.get("feedback"),
                )
                self.dataset.corrections.append(correction)

            # Restore feedback
            self.dataset.feedback = data.get("feedback", [])

            # Restore structured feedback
            for fb_data in data.get("structured_feedback", []):
                fb = Feedback(
                    id=fb_data["id"],
                    text=fb_data["text"],
                    level=fb_data["level"],
                    target_id=fb_data.get("target_id", ""),
                )
                self.dataset.structured_feedback.append(fb)

            # Restore rules
            for rule_data in data.get("rules", []):
                rule = Rule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    description=rule_data["description"],
                    format=RuleFormat(rule_data["format"]),
                    content=rule_data["content"],
                    priority=rule_data.get("priority", 5),
                    confidence=rule_data.get("confidence", 0.5),
                    times_applied=rule_data.get("times_applied", 0),
                    successes=rule_data.get("successes", 0),
                    failures=rule_data.get("failures", 0),
                    output_template=rule_data.get("output_template"),
                    output_key=rule_data.get("output_key"),
                )
                self.dataset.rules.append(rule)

            print(
                f"✓ Loaded dataset: {len(self.dataset.corrections)} corrections, {len(self.dataset.examples)} examples"
            )

        except Exception as e:
            print(f"Error loading dataset: {e}")
