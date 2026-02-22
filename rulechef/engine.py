"""Main RuleChef orchestrator"""

import json
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
    Dataset,
    Feedback,
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
from rulechef.observation import ObservationManager
from rulechef.pipeline import LearningPipeline
from rulechef.storage import DatasetStore


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
        max_rules_per_class: int = 5,
        max_counter_examples: int = 10,
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
            max_samples: Maximum training examples to include in prompts
                (per-class positives and patch failures are both capped to this).
            max_rules_per_class: Maximum rules to generate per class in per-class synthesis.
            max_counter_examples: Maximum counter-examples from other classes per prompt.
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
        self._max_rules_per_class = max_rules_per_class
        self._max_counter_examples = max_counter_examples

        # Coordinator for learning decisions (swappable simple/agentic)
        self.coordinator = coordinator or SimpleCoordinator()

        # Propagate training logger to coordinator if it's agentic
        if self.training_logger and isinstance(self.coordinator, AgenticCoordinator):
            self.coordinator.training_logger = self.training_logger

        # Buffer for observed examples (buffer-first architecture)
        self.buffer = ExampleBuffer()

        # Auto-trigger: coordinator checks after each add_example/add_correction
        self.auto_trigger = auto_trigger

        # Composed subsystems
        self._store = DatasetStore(self.storage_path)
        self._observations = ObservationManager(self)
        self._pipeline = LearningPipeline(self)

        # Dataset and learner: created now if task is provided, lazily otherwise
        if task is not None:
            self._initialize_with_task(task)
        else:
            self.dataset = None
            self.learner = None
            self.allowed_formats = None

    # ========================================
    # Backward-compat property bridges
    # ========================================

    @property
    def _observer(self):
        return self._observations._observer

    @_observer.setter
    def _observer(self, value):
        self._observations._observer = value

    @property
    def _pending_raw_observations(self):
        return self._observations._pending_raw_observations

    @_pending_raw_observations.setter
    def _pending_raw_observations(self, value):
        self._observations._pending_raw_observations = value

    @property
    def _learning_thread(self):
        return self._observations._learning_thread

    @property
    def _stop_learning(self):
        return self._observations._stop_learning

    # ========================================
    # Initialization
    # ========================================

    def _initialize_with_task(self, task: Task) -> None:
        """Create dataset and learner from task.

        Called from __init__ when task is provided, or after auto-discovery.
        """
        self.task = task

        # Convert string format names to RuleFormat enums
        if self._allowed_formats_raw:
            self.allowed_formats = [
                RuleFormat(f) if isinstance(f, str) else f for f in self._allowed_formats_raw
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
            max_rules_per_class=self._max_rules_per_class,
            max_counter_examples=self._max_counter_examples,
            training_logger=self.training_logger,
        )

        # Load existing dataset if on disk
        self._store.load(self.dataset)

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

    def add_example(self, input_data: dict, output_data: dict, source: str = "human_labeled"):
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
            self._observations.check_and_trigger_learning()

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
            self._observations.check_and_trigger_learning()

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
            id=self._store.generate_id(),
            text=feedback,
            level=level,
            target_id=target_id,
        )
        self.dataset.structured_feedback.append(fb)

        # Also keep legacy list for backward compat
        if level == "task":
            self.dataset.feedback.append(feedback)

        self._store.save(self.dataset)
        print(f"✓ Added {level}-level feedback" + (f" for {target_id}" if target_id else ""))

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
        print(f"✓ Generated {num_examples} examples (buffer: {stats['new_examples']} new)")

        # Check coordinator once after generating all
        if self.auto_trigger:
            self._observations.check_and_trigger_learning()

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
        self.buffer.add_llm_observation(input_data, output_data, metadata=metadata or {})

        stats = self.buffer.get_stats()
        print(
            f"✓ Added observation (buffer: {stats['new_examples']} new, "
            f"{stats['total_examples']} total)"
        )

        if self.auto_trigger:
            self._observations.check_and_trigger_learning()

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
        self._observations.add_raw_observation(messages, response, metadata)

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
        return self._pipeline.run(
            run_evaluation=run_evaluation,
            min_examples=min_examples,
            max_refinement_iterations=max_refinement_iterations,
            sampling_strategy=sampling_strategy,
            incremental_only=incremental_only,
        )

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
    # Observation Mode (delegates to ObservationManager)
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
        return self._observations.start_observing(
            openai_client,
            auto_learn=auto_learn,
            check_interval=check_interval,
            extract_input=extract_input,
            extract_output=extract_output,
            min_observations_for_discovery=min_observations_for_discovery,
        )

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
        return self._observations.discover_task()

    def start_observing_gliner(
        self,
        gliner_model,
        method: str | None = None,
        auto_learn: bool = True,
        check_interval: int = 60,
    ):
        """Observe GLiNER/GLiNER2 predictions to learn rules from them.

        GLiNER:  auto-patches predict_entities() → NER observations.
        GLiNER2: patches the specified method:
          - "extract_entities" → NER
          - "classify_text"    → CLASSIFICATION
          - "extract_json"     → TRANSFORMATION

        Args:
            gliner_model: A GLiNER or GLiNER2 model instance.
            method: Which method to observe (required for GLiNER2 non-NER tasks).
            auto_learn: If True, trigger learning automatically.
            check_interval: Seconds between coordinator checks.

        Returns:
            The same model (monkey-patched in place).
        """
        return self._observations.start_observing_gliner(
            gliner_model,
            method=method,
            auto_learn=auto_learn,
            check_interval=check_interval,
        )

    def stop_observing_gliner(self):
        """Stop observing GLiNER predictions."""
        self._observations.stop_observing_gliner()

    def stop_observing(self):
        """Stop observing LLM calls and background learning."""
        self._observations.stop_observing()

    def trigger_manual_learning(self):
        """Manually trigger learning from buffered examples."""
        return self._observations.trigger_manual_learning()

    def get_buffer_stats(self) -> dict:
        """Get statistics about buffered examples and observations."""
        return self._observations.get_buffer_stats()

    # ========================================
    # Analysis
    # ========================================

    def get_stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dict with keys: 'task' (str), 'dataset' (str), 'corrections' (int),
            'examples' (int), 'feedback' (int), 'rules' (int),
            'description' (str).
        """
        self._require_task("get_stats")
        return self._store.get_stats(self.dataset)

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
            self._store.save(self.dataset)
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
        return self._store.get_rules_summary(self.dataset)
