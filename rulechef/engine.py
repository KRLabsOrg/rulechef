"""Main RuleChef orchestrator"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from openai import OpenAI

from rulechef.core import Task, Dataset, Example, Correction, Rule, RuleFormat, TaskType
from rulechef.learner import RuleLearner
from rulechef.buffer import ExampleBuffer
from rulechef.coordinator import CoordinatorProtocol, SimpleCoordinator
from rulechef.openai_wrapper import OpenAIObserver


class RuleChef:
    """Main orchestrator for learning and applying rules"""

    def __init__(
        self,
        task: Task,
        client: Optional[OpenAI] = None,
        dataset_name: str = "default",
        storage_path: str = "./rulechef_data",
        allowed_formats: Optional[List[RuleFormat]] = None,
        sampling_strategy: str = "balanced",
        coordinator: Optional[CoordinatorProtocol] = None,
        auto_trigger: bool = False,
        model: str = "gpt-4o-mini",
    ):
        self.task = task
        self.llm = client or OpenAI()
        self.model = model
        self.dataset = Dataset(name=dataset_name, task=task)
        self.storage_path = Path(storage_path)
        # Convert string format names to RuleFormat enums if needed
        if allowed_formats:
            self.allowed_formats = [
                RuleFormat(f) if isinstance(f, str) else f for f in allowed_formats
            ]
        else:
            self.allowed_formats = [RuleFormat.REGEX, RuleFormat.CODE]
        self.sampling_strategy = sampling_strategy
        self.learner = RuleLearner(
            self.llm,
            allowed_formats=self.allowed_formats,
            sampling_strategy=sampling_strategy,
            model=model,
        )

        # Coordinator for learning decisions (swappable simple/agentic)
        self.coordinator = coordinator or SimpleCoordinator()

        # Buffer for observed examples (buffer-first architecture)
        self.buffer = ExampleBuffer()

        # Auto-trigger: coordinator checks after each add_example/add_correction
        self.auto_trigger = auto_trigger

        # Observation mode components
        self._observer: Optional[OpenAIObserver] = None
        self._learning_thread: Optional[threading.Thread] = None
        self._stop_learning = threading.Event()

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing dataset if available
        self._load_dataset()

    # ========================================
    # Data Collection
    # ========================================

    def add_example(
        self, input_data: Dict, output_data: Dict, source: str = "human_labeled"
    ):
        """
        Add a labeled training example.

        Uses buffer-first architecture: example goes to buffer, then coordinator
        decides when to trigger learning.
        """
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
        input_data: Dict,
        model_output: Dict,
        expected_output: Dict,
        feedback: Optional[str] = None,
    ):
        """
        Add a user correction (high value signal).

        Uses buffer-first architecture: correction goes to buffer, then coordinator
        decides when to trigger learning. Corrections are high-priority signals.
        """
        # Add to buffer (not dataset directly)
        self.buffer.add_human_correction(input_data, expected_output, model_output)

        stats = self.buffer.get_stats()
        print(
            f"✓ Added correction (buffer: {stats['new_corrections']} corrections, {stats['new_examples']} new total)"
        )

        # If auto-trigger enabled, check coordinator (corrections are high-value!)
        if self.auto_trigger:
            self._check_and_trigger_learning()

    def add_feedback(self, feedback: str):
        """Add general user feedback"""
        self.dataset.feedback.append(feedback)
        self._save_dataset()

    def generate_llm_examples(self, num_examples: int = 5, seed: int = 42):
        """
        Generate synthetic training examples using LLM.

        Examples go to buffer and can trigger learning if auto_trigger=True.
        """
        print(f"\n🤖 Generating {num_examples} examples with LLM...")
        for i in range(num_examples):
            input_data = self.learner._generate_input(self.task, self.dataset, seed + i)
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

    # ========================================
    # Learning
    # ========================================

    def learn_rules(
        self,
        run_evaluation: Optional[bool] = None,
        min_examples: int = 1,
        max_refinement_iterations: int = 3,
        sampling_strategy: Optional[str] = None,
    ):
        """
        Learn rules from all collected data

        This is the core batch learning process

        Args:
            run_evaluation: Whether to run evaluation/refinement loop
                - None (default): Auto-enable if total_data >= 3, disable otherwise
                - True: Always enable refinement (3 iterations)
                - False: Disable refinement (faster, synthesis only)
            min_examples: Minimum training items required
            max_refinement_iterations: Max iterations in refinement loop (1-3, default 3)
            sampling_strategy: Override default sampling strategy for this run
                - Options: 'balanced', 'recent', 'diversity', 'uncertain', 'varied'
        """

        start_time = time.time()

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
                    # Add as Correction to dataset
                    correction = Correction(
                        id=self._generate_id(),
                        input=example.input,
                        model_output=example.output.get("actual", {}),
                        expected_output=example.output.get("expected", example.output),
                        feedback=None,
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

        try:
            # Synthesize initial ruleset
            rules = self.learner.synthesize_ruleset(self.dataset)

            if not rules:
                print("Failed to synthesize rules")
                return None

            print(f"✓ Generated {len(rules)} rules")

            # Evaluate and refine
            if run_evaluation:
                rules, metrics = self.learner.evaluate_and_refine(
                    rules, self.dataset, max_iterations=max_refinement_iterations
                )
            else:
                metrics = None

            # Save learned rules
            self.dataset.rules = rules
            self._save_dataset()

            elapsed = time.time() - start_time

            print(f"\n{'=' * 60}")
            print(f"Learning complete! ({elapsed:.1f}s)")
            print(f"  Rules: {len(rules)}")
            if metrics and metrics.get("total", 0) > 0:
                print(
                    f"  Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})"
                )
            print(f"{'=' * 60}\n")

            return rules, metrics
        finally:
            # Restore original sampling strategy
            if sampling_strategy:
                self.learner.sampling_strategy = original_strategy

    # ========================================
    # Execution
    # ========================================

    def extract(self, input_data: Dict) -> Dict:
        """
        Extract from input using learned rules.
        Falls back to LLM if no rules or low confidence.

        Works for all task types: EXTRACTION, NER, CLASSIFICATION, TRANSFORMATION.
        """

        if not self.dataset.rules:
            # No rules learned, use LLM fallback
            return self._execute_with_llm(input_data)

        # Apply rules
        output = self.learner._apply_rules(self.dataset.rules, input_data)

        # Store current extraction for potential correction
        self.current_extraction = output

        # Check confidence based on task type
        if self.task.type == TaskType.EXTRACTION:
            spans = output.get("spans", [])
            if spans:
                avg_confidence = sum(s.get("score", 0.5) for s in spans) / len(spans)
                if avg_confidence < 0.3:
                    print(
                        f"Low confidence ({avg_confidence:.1%}), using LLM fallback..."
                    )
                    return self._execute_with_llm(input_data)

        elif self.task.type == TaskType.NER:
            # For NER, check if we got any entities
            entities = output.get("entities", output.get("spans", []))
            if not entities:
                # No entities found, might want LLM fallback
                # But empty is valid for texts with no entities, so don't fallback
                pass

        # For other types, we assume rule execution is binary (success/fail)
        # If output is empty/None, fallback
        if not output:
            return self._execute_with_llm(input_data)

        return output

    def _execute_with_llm(self, input_data: Dict) -> Any:
        """Execute extraction using LLM directly"""

        if self.task.type == TaskType.EXTRACTION:
            # Format input for prompt
            prompt = f"""Extract answer spans from the following:

Question: {input_data.get("question", "")}
Context: {input_data.get("context", "")}

Return JSON:
{{
  "spans": [
    {{"text": "...", "start": 0, "end": 10}}
  ]
}}
"""
        elif self.task.type == TaskType.NER:
            # NER-specific prompt
            text = input_data.get("text", "")
            prompt = f"""Extract named entities from the following text.

Text: {text}

Output schema: {self.task.output_schema}

Return JSON with entities including text, character offsets (start, end), and entity type.
Example:
{{
  "entities": [
    {{"text": "Apple Inc.", "start": 0, "end": 10, "type": "ORG"}},
    {{"text": "Tim Cook", "start": 20, "end": 28, "type": "PER"}}
  ]
}}
"""
        elif self.task.type == TaskType.TRANSFORMATION:
            # TRANSFORMATION needs examples to learn the exact output format
            examples_prompt = ""
            if self.dataset.examples:
                examples_prompt = "\nExamples of expected output format:\n"
                for ex in self.dataset.examples[:3]:  # Show up to 3 examples
                    examples_prompt += f"\nInput: {json.dumps(ex.input)[:200]}...\n"
                    examples_prompt += f"Output: {json.dumps(ex.expected_output)}\n"

            prompt = f"""Task: {self.task.name}
Description: {self.task.description}

Input: {json.dumps(input_data)}

Output schema: {self.task.output_schema}
{examples_prompt}
IMPORTANT: Return JSON that EXACTLY matches the schema and key names shown in the examples.
Return ONLY valid JSON, no explanation.
"""
        else:
            # Generic prompt for other tasks (CLASSIFICATION)
            prompt = f"""Task: {self.task.name}
Description: {self.task.description}

Input: {json.dumps(input_data)}

Return JSON matching this schema:
{self.task.output_schema}
"""

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
            print(f"Error in LLM extraction: {e}")
            if self.task.type == TaskType.EXTRACTION:
                return {"spans": []}
            elif self.task.type == TaskType.NER:
                return {"entities": []}
            else:
                return {}

    # ========================================
    # Observation Mode (LLM Middleware)
    # ========================================

    def start_observing(
        self,
        openai_client,
        auto_learn: bool = True,
        check_interval: int = 60,
        extract_input: Optional[Callable] = None,
        extract_output: Optional[Callable] = None,
    ):
        """
        Start observing OpenAI-compatible client calls to collect training examples.

        Args:
            openai_client: OpenAI client (or compatible API)
            auto_learn: If True, automatically triggers learning when coordinator decides
            check_interval: Seconds between coordinator checks (default 60)
            extract_input: Custom function to parse API kwargs into task input
            extract_output: Custom function to parse API response into task output

        Returns:
            Wrapped client - use this for API calls

        Example:
            chef = RuleChef(task)
            client = chef.start_observing(openai_client, auto_learn=True)

            # Use client normally, RuleChef observes
            response = client.chat.completions.create(...)

            # Auto-learns when ready
        """
        # Create observer
        self._observer = OpenAIObserver(
            self.buffer, self.task, extract_input, extract_output
        )

        # Attach to client
        wrapped_client = self._observer.attach(openai_client)

        # Start auto-learning loop if requested
        if auto_learn:
            self._start_learning_loop(check_interval)
            print(
                f"✓ Started observing with auto-learning (check every {check_interval}s)"
            )
        else:
            print("✓ Started observing (manual learning mode)")

        return wrapped_client

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
                    # Ask coordinator if we should learn
                    decision = self.coordinator.should_trigger_learning(
                        self.buffer, self.dataset.rules
                    )

                    if decision.should_learn:
                        print(f"\n{'=' * 60}")
                        print(f"Auto-triggering learning: {decision.reasoning}")
                        print(f"{'=' * 60}")
                        self._auto_learn(decision)

                except Exception as e:
                    print(f"Error in learning loop: {e}")

                # Wait for next check
                self._stop_learning.wait(interval)

        self._learning_thread = threading.Thread(target=loop, daemon=True)
        self._learning_thread.start()

    def _auto_learn(self, decision):
        """Execute learning based on coordinator decision"""
        old_rules = self.dataset.rules.copy() if self.dataset.rules else None

        try:
            # learn_rules() will convert buffer → dataset automatically
            rules, metrics = self.learn_rules(
                sampling_strategy=decision.strategy,
                max_refinement_iterations=decision.max_iterations,
            )

            # Notify coordinator of results
            self.coordinator.on_learning_complete(old_rules, rules, metrics)

        except Exception as e:
            print(f"Error during auto-learning: {e}")

    def _check_and_trigger_learning(self):
        """
        Check coordinator and trigger learning if ready.

        Called after add_example() or add_correction() when auto_trigger=True.
        """
        decision = self.coordinator.should_trigger_learning(
            self.buffer, self.dataset.rules
        )

        if decision.should_learn:
            print(f"\n{'=' * 60}")
            print(f"Auto-triggering learning: {decision.reasoning}")
            print(f"{'=' * 60}")
            self._auto_learn(decision)

    def trigger_manual_learning(self):
        """Manually trigger learning from buffered examples"""
        decision = self.coordinator.should_trigger_learning(
            self.buffer, self.dataset.rules
        )

        if decision.should_learn:
            print(f"✓ Triggering learning: {decision.reasoning}")
            self._auto_learn(decision)
            return True
        else:
            print(f"✗ Not ready to learn: {decision.reasoning}")
            return False

    def get_buffer_stats(self) -> Dict:
        """Get statistics about buffered examples"""
        return {
            **self.buffer.get_stats(),
            "coordinator_analysis": self.coordinator.analyze_buffer(self.buffer),
        }

    # ========================================
    # Utils
    # ========================================

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            "task": self.dataset.task.name,
            "dataset": self.dataset.name,
            "corrections": len(self.dataset.corrections),
            "examples": len(self.dataset.examples),
            "feedback": len(self.dataset.feedback),
            "rules": len(self.dataset.rules),
            "description": self.dataset.description,
        }

    def get_rules_summary(self) -> List[Dict]:
        """Get formatted summary of learned rules"""
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
            with open(filepath, "r") as f:
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
                )
                self.dataset.rules.append(rule)

            print(
                f"✓ Loaded dataset: {len(self.dataset.corrections)} corrections, {len(self.dataset.examples)} examples"
            )

        except Exception as e:
            print(f"Error loading dataset: {e}")
