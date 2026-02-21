"""LLM-based rule learning"""

import json
import random
import re
import time
import uuid
from collections import defaultdict

from openai import OpenAI

from rulechef.core import Correction, Dataset, Rule, RuleFormat, TaskType
from rulechef.evaluation import (
    EvalResult,
    evaluate_dataset,
)
from rulechef.executor import RuleExecutor
from rulechef.prompts import PromptBuilder


class RuleLearner:
    """Learns extraction rules from examples using LLM"""

    def __init__(
        self,
        llm: OpenAI,
        allowed_formats: list[RuleFormat] | None = None,
        sampling_strategy: str = "balanced",
        model: str = "gpt-4o-mini",
        use_spacy_ner: bool = False,
        use_grex: bool = True,
        max_rules: int = 10,
        max_samples: int = 50,
        training_logger=None,
    ):
        """Initialize the rule learner.

        Args:
            llm: OpenAI client instance for LLM calls.
            allowed_formats: Rule formats to generate (e.g. REGEX, CODE, SPACY).
                Defaults to [REGEX, CODE].
            sampling_strategy: How to sample training data for prompts.
                Options: 'balanced', 'recent', 'diversity', 'uncertain', 'varied'.
            model: OpenAI model name for synthesis and patch calls.
            use_spacy_ner: If True, enable spaCy NER during rule execution.
            use_grex: If True, use grex for regex pattern suggestion in prompts.
            max_rules: Maximum number of rules to generate per synthesis call.
            max_samples: Maximum training examples to include in prompts.
            training_logger: Optional TrainingDataLogger for capturing LLM calls.
        """
        self.llm = llm
        self.allowed_formats = allowed_formats or [RuleFormat.REGEX, RuleFormat.CODE]
        self.sampling_strategy = sampling_strategy
        self.model = model
        self.use_spacy_ner = use_spacy_ner
        self.use_grex = use_grex
        self.max_rules = max_rules
        self.max_samples = max_samples
        self.training_logger = training_logger
        self.executor = RuleExecutor(use_spacy_ner=use_spacy_ner)
        self.prompt_builder = PromptBuilder(
            self.allowed_formats,
            use_spacy_ner=use_spacy_ner,
            use_grex=use_grex,
        )

    # ========================================
    # Rule Execution (delegates to executor)
    # ========================================

    def _apply_rules(
        self,
        rules: list[Rule],
        input_data: dict,
        task_type: TaskType | None = None,
        text_field: str | None = None,
    ) -> dict:
        """Apply rules to input data. Delegates to executor."""
        return self.executor.apply_rules(rules, input_data, task_type, text_field)

    # ========================================
    # Rule Synthesis
    # ========================================

    def synthesize_ruleset(
        self,
        dataset: Dataset,
        max_rules: int | None = None,
    ) -> list[Rule]:
        """Generate initial ruleset from dataset.

        Returns:
            List[Rule] of synthesized rules, or empty list on failure.
        """
        max_rules = max_rules or self.max_rules
        prompt = self._build_synthesis_prompt(dataset, max_rules)

        print("📚 Synthesizing rules from dataset...")
        start = time.time()

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                max_completion_tokens=16384,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content
            result = self._parse_json(response_text)
            rules = self._parse_rules_from_response(result, max_rules, dataset)

            if self.training_logger:
                self.training_logger.log(
                    "rule_synthesis",
                    [{"role": "user", "content": prompt}],
                    response_text,
                    {
                        "task_name": dataset.task.name if dataset.task else None,
                        "task_type": dataset.task.type.value if dataset.task else None,
                        "dataset_size": len(dataset.examples),
                        "num_rules_in_response": len(rules),
                        "response_valid": bool(rules),
                        "max_rules": max_rules,
                    },
                )

            elapsed = time.time() - start
            print(f"✓ Synthesized {len(rules)} rules ({elapsed:.1f}s)")
            return rules

        except Exception as e:
            print(f"Error synthesizing rules: {e}")
            return []

    def synthesize_ruleset_per_class(
        self,
        dataset: Dataset,
        max_rules_per_class: int = 5,
        max_counter_examples: int = 10,
    ) -> list[Rule]:
        """Synthesize rules one class at a time for better focus and coverage.

        Args:
            dataset: Dataset with training examples.
            max_rules_per_class: Maximum rules to generate for each class.
            max_counter_examples: Maximum counter-examples (other classes) to
                include per class prompt to prevent false positives.

        Returns:
            List[Rule] combining rules from all classes. Falls back to bulk
            synthesis if no classes are found.
        """
        classes = self._get_classes(dataset)
        if not classes:
            print("⚠ No classes found, falling back to bulk synthesis")
            return self.synthesize_ruleset(dataset)

        print(
            f"📚 Per-class synthesis: {len(classes)} classes, up to {max_rules_per_class} rules each"
        )
        all_rules = []
        total_start = time.time()

        for i, target_class in enumerate(classes):
            # Collect counter-examples from other classes
            counter_examples = []
            task_type = dataset.task.type
            for ex in dataset.examples:
                if task_type == TaskType.CLASSIFICATION:
                    if ex.expected_output.get("label") != target_class:
                        counter_examples.append(ex)
                elif task_type == TaskType.NER:
                    entities = ex.expected_output.get("entities", [])
                    if not any(e.get("type") == target_class for e in entities):
                        counter_examples.append(ex)

            # Sample counter-examples to keep prompt manageable
            if len(counter_examples) > max_counter_examples:
                rng = random.Random(42 + i)
                counter_examples = rng.sample(counter_examples, max_counter_examples)

            prompt = self._build_synthesis_prompt(
                dataset,
                max_rules_per_class,
                target_class=target_class,
                counter_examples=counter_examples,
            )

            start = time.time()
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=16384,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                response_text = response.choices[0].message.content
                result = self._parse_json(response_text)
                rules = self._parse_rules_from_response(
                    result, max_rules_per_class, dataset
                )
                all_rules.extend(rules)

                if self.training_logger:
                    self.training_logger.log(
                        "rule_synthesis_per_class",
                        [{"role": "user", "content": prompt}],
                        response_text,
                        {
                            "task_name": dataset.task.name if dataset.task else None,
                            "task_type": dataset.task.type.value
                            if dataset.task
                            else None,
                            "dataset_size": len(dataset.examples),
                            "target_class": target_class,
                            "num_counter_examples": len(counter_examples),
                            "num_rules_in_response": len(rules),
                            "response_valid": bool(rules),
                        },
                    )

                elapsed = time.time() - start
                print(
                    f"  [{i + 1}/{len(classes)}] {target_class}: {len(rules)} rules ({elapsed:.1f}s)"
                )
            except Exception as e:
                elapsed = time.time() - start
                print(
                    f"  [{i + 1}/{len(classes)}] {target_class}: ERROR ({elapsed:.1f}s) - {e}"
                )

        total_elapsed = time.time() - total_start
        print(
            f"✓ Per-class synthesis complete: {len(all_rules)} rules from {len(classes)} classes ({total_elapsed:.1f}s)"
        )
        return all_rules

    def _parse_rules_from_response(
        self, result: dict, max_rules: int, dataset: Dataset | None = None
    ) -> list[Rule]:
        """Parse rules from LLM response"""
        from rulechef.core import is_pydantic_schema

        primary_key = None
        if dataset and dataset.task.output_schema:
            if is_pydantic_schema(dataset.task.output_schema):
                # For Pydantic, get first field name from model
                fields = list(dataset.task.output_schema.model_fields.keys())
                primary_key = fields[0] if fields else None
            else:
                primary_key = list(dataset.task.output_schema.keys())[0]

        rules = []
        for i, rule_data in enumerate(result.get("rules", [])[:max_rules]):
            raw_format = rule_data.get("format", "regex")
            # Accept common aliases
            if raw_format == "python":
                raw_format = "code"

            try:
                rule_format = RuleFormat(raw_format)
            except ValueError:
                print(
                    f"   ⚠ Skipped rule with unsupported format '{raw_format}': {rule_data.get('name', f'Rule {i + 1}')}"
                )
                continue

            if rule_format not in self.allowed_formats:
                print(
                    f"   ⚠ Skipped {rule_format.value} rule (not allowed): {rule_data.get('name', f'Rule {i + 1}')}"
                )
                continue

            pattern_content = rule_data.get("pattern") or rule_data.get("content", "")
            # Normalize structured content to JSON string
            if isinstance(pattern_content, (list, dict)):
                try:
                    pattern_content = json.dumps(pattern_content)
                except Exception:
                    pattern_content = str(pattern_content)

            output_template = rule_data.get("output_template")
            output_key = rule_data.get("output_key")

            # Ensure output_template is a dict (LLM might return string/list)
            if output_template and not isinstance(output_template, dict):
                print(
                    f"   ⚠ Skipped rule with invalid output_template (not a dict): {rule_data.get('name', f'Rule {i + 1}')}"
                )
                continue

            if dataset and dataset.task.type == TaskType.NER:
                if not output_template:
                    print(
                        f"   ⚠ Skipped rule without output_template: {rule_data.get('name', f'Rule {i + 1}')}"
                    )
                    continue
                if not output_key:
                    output_key = primary_key or "entities"

                # Validate label against schema if Pydantic
                valid_labels = dataset.task.get_labels()
                if valid_labels and output_template:
                    template_type = output_template.get("type")
                    if (
                        template_type
                        and template_type not in valid_labels
                        and not template_type.startswith("$")
                    ):
                        print(
                            f"   ⚠ Warning: Rule '{rule_data.get('name')}' uses unknown label '{template_type}'. Valid: {valid_labels}"
                        )
            elif output_template and not output_key and primary_key:
                output_key = primary_key
            elif dataset and dataset.task.type == TaskType.TRANSFORMATION:
                if rule_format != RuleFormat.CODE:
                    if not output_key and primary_key:
                        output_key = primary_key
                    if not output_key:
                        print(
                            f"   ⚠ Skipped transformation rule without output_key: {rule_data.get('name', f'Rule {i + 1}')}"
                        )
                        continue

            rule = Rule(
                id=self._generate_id(),
                name=rule_data.get("name", f"Rule {i + 1}"),
                description=rule_data.get("description", ""),
                format=rule_format,
                content=pattern_content,
                priority=rule_data.get("priority", 5),
                output_template=output_template,
                output_key=output_key,
            )

            if self._validate_rule(rule):
                # Ensure spaCy content is stored as JSON string for consistency
                if rule.format == RuleFormat.SPACY and not isinstance(
                    rule.content, str
                ):
                    rule.content = json.dumps(rule.content)
                rules.append(rule)
            else:
                print(f"   ⚠ Skipped invalid rule: {rule.name}")

        return rules

    # ========================================
    # Rule Evaluation & Refinement
    # ========================================

    def evaluate_and_refine(
        self,
        rules: list[Rule],
        dataset: Dataset,
        max_iterations: int = 3,
        coordinator=None,
        iteration_callback=None,
    ) -> tuple:
        """Evaluate rules and refine through patch-based loop.

        Each iteration generates patch rules for failures and merges them
        into the existing set, keeping working rules intact. Stops early
        if exact match reaches 90% or the coordinator signals to stop.

        Args:
            rules: Initial set of rules to refine.
            dataset: Dataset to evaluate against.
            max_iterations: Maximum refinement iterations (1-3).
            coordinator: Optional CoordinatorProtocol. If provided, its
                guide_refinement() is called each iteration for LLM-powered
                guidance on which classes to focus and when to stop.
            iteration_callback: Optional callable(iteration: int, rules: List[Rule],
                eval_result: EvalResult) called after each evaluation. Useful for
                logging per-iteration metrics in benchmarks.

        Returns:
            Tuple of (best_rules, best_eval_result) where best_rules is
            the rule set with the highest micro F1 seen across iterations.
        """
        print(f"\n🔄 Refinement loop (max {max_iterations} iterations)")

        best_rules = rules
        best_f1 = 0.0
        best_eval = EvalResult()

        for iteration in range(max_iterations):
            iter_num = iteration + 1
            print(f"[{iter_num}/{max_iterations}] Evaluating rules...")

            eval_result = self._evaluate_rules(rules, dataset)
            exact = eval_result.exact_match
            correct = int(exact * eval_result.total_docs)

            print(
                f"[{iter_num}/{max_iterations}] Exact match: {exact:.1%} "
                f"({correct}/{eval_result.total_docs}), "
                f"micro F1: {eval_result.micro_f1:.1%}"
            )

            if eval_result.micro_f1 > best_f1:
                best_rules = rules
                best_f1 = eval_result.micro_f1
                best_eval = eval_result

            if iteration_callback:
                iteration_callback(iter_num, rules, eval_result)

            if exact >= 0.90:
                print("✓ Achieved 90%+ exact match!")
                break

            # Ask coordinator for guidance (if provided)
            guidance = ""
            if coordinator:
                guidance, should_continue = coordinator.guide_refinement(
                    eval_result, iteration, max_iterations
                )
                if not should_continue:
                    print("🤖 Coordinator: stop refining")
                    break

            if eval_result.failures:
                print(
                    f"[{iter_num}/{max_iterations}] Patching {len(eval_result.failures)} failures..."
                )
                start = time.time()
                patch = self.synthesize_patch_ruleset(
                    rules,
                    eval_result.failures,
                    max_rules=self.max_rules,
                    dataset=dataset,
                    guidance=guidance,
                    class_metrics=eval_result.per_class,
                )
                elapsed = time.time() - start
                if not patch:
                    print("⚠ Patch synthesis returned nothing, keeping best rules")
                else:
                    candidate = self._merge_patch(rules, patch)
                    candidate_eval = self._evaluate_rules(candidate, dataset)
                    candidate_exact = candidate_eval.exact_match
                    if candidate_exact >= exact:
                        rules = candidate
                        print(
                            f"[{iter_num}/{max_iterations}] Patched → {len(rules)} rules, "
                            f"exact match {exact:.1%} → {candidate_exact:.1%} ({elapsed:.1f}s)"
                        )
                        if candidate_eval.micro_f1 > best_f1:
                            best_rules = rules
                            best_f1 = candidate_eval.micro_f1
                            best_eval = candidate_eval
                    else:
                        print(
                            f"[{iter_num}/{max_iterations}] Patch made it worse "
                            f"({exact:.1%} → {candidate_exact:.1%}), keeping previous"
                        )
            else:
                print("✓ No failures to fix!")
                break

        return best_rules, best_eval

    @staticmethod
    def _merge_patch(existing: list[Rule], patches: list[Rule]) -> list[Rule]:
        """Merge patch rules into existing set by name."""
        by_name = {r.name: r for r in existing}
        for pr in patches:
            if pr.name in by_name:
                current = by_name[pr.name]
                pr.times_applied = current.times_applied
                pr.successes = current.successes
                pr.failures = current.failures
                pr.confidence = current.confidence
            by_name[pr.name] = pr
        return list(by_name.values())

    def _evaluate_rules(self, rules: list[Rule], dataset: Dataset) -> EvalResult:
        """Evaluate rules on all training data. Returns EvalResult."""
        return evaluate_dataset(rules, dataset, self._apply_rules, mode="text")

    def _coerce_spacy_content(self, content) -> list | None:
        """Coerce spaCy content into a list of patterns."""
        if isinstance(content, list):
            return content
        if isinstance(content, dict):
            return [content]
        if isinstance(content, str):
            text = content.strip()
            try:
                return json.loads(text)
            except Exception:
                # Try to extract the first JSON array substring
                if "[" in text and "]" in text:
                    try:
                        candidate = text[text.index("[") : text.rindex("]") + 1]
                        return json.loads(candidate)
                    except Exception:
                        return None
        return None

    def synthesize_patch_ruleset(
        self,
        current_rules: list[Rule],
        failures: list[dict],
        max_rules: int | None = None,
        dataset: Dataset | None = None,
        guidance: str = "",
        class_metrics: list | None = None,
    ) -> list[Rule]:
        """Generate incremental rules targeted at specific failures.

        Returns only new/updated rules; the caller is responsible for
        merging them into the existing rule set.

        Returns:
            List[Rule] of patch rules, or empty list on failure.
        """
        max_rules = max_rules or self.max_rules
        sampled_failures = self._sample_failures(
            failures,
            max_samples=self.max_samples,
            class_metrics=class_metrics,
        )
        prompt = self._build_patch_prompt(
            current_rules,
            sampled_failures,
            max_rules,
            dataset=dataset,
            guidance=guidance,
        )

        print("🩹 Synthesizing patch rules...")
        start = time.time()

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                max_completion_tokens=16384,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content
            result = self._parse_json(response_text)
            rules = self._parse_rules_from_response(result, max_rules, dataset=dataset)

            if self.training_logger:
                self.training_logger.log(
                    "rule_patch",
                    [{"role": "user", "content": prompt}],
                    response_text,
                    {
                        "task_name": dataset.task.name
                        if dataset and dataset.task
                        else None,
                        "task_type": dataset.task.type.value
                        if dataset and dataset.task
                        else None,
                        "num_failures": len(sampled_failures),
                        "num_existing_rules": len(current_rules),
                        "num_rules_in_response": len(rules),
                        "response_valid": bool(rules),
                        "guidance": guidance[:200] if guidance else None,
                    },
                )

            elapsed = time.time() - start
            print(f"✓ Patch synthesis returned {len(rules)} rules ({elapsed:.1f}s)")
            return rules
        except Exception as e:
            print(f"Error synthesizing patch rules: {e}")
            return []

    # ========================================
    # Prompt Building
    # ========================================

    def _build_synthesis_prompt(
        self,
        dataset: Dataset,
        max_rules: int,
        target_class: str | None = None,
        counter_examples: list | None = None,
    ) -> str:
        """Build prompt for rule synthesis using PromptBuilder.

        When target_class is set, builds a focused prompt for one class only.
        """
        if target_class:
            return self._build_per_class_prompt(
                dataset, max_rules, target_class, counter_examples or []
            )

        # Original bulk synthesis path
        sampled_data = self._sample_training_data(
            dataset,
            max_samples=self.max_samples,
            strategy=self.sampling_strategy,
        )

        # Separate corrections and examples
        corrections = [d for d in sampled_data if isinstance(d, Correction)]
        examples = [d for d in sampled_data if not isinstance(d, Correction)]

        # Build base prompt from builder
        prompt = self.prompt_builder._build_task_header(dataset)

        # Add training data sections
        if corrections:
            prompt += self.prompt_builder._build_corrections_section(corrections)
        if examples:
            prompt += self.prompt_builder._build_examples_section(examples)

        # Entity evidence helps the model infer labels/patterns when schemas don't encode them.
        prompt += self.prompt_builder._build_data_evidence(dataset)

        # Add other sections
        prompt += self.prompt_builder._build_feedback_section(dataset)
        prompt += self.prompt_builder._build_existing_rules_section(dataset)
        prompt += self.prompt_builder._build_task_instructions(dataset, max_rules)
        prompt += self.prompt_builder._build_format_instructions(dataset.task.type)
        prompt += self.prompt_builder._build_response_schema(dataset)
        prompt += self.prompt_builder._build_format_examples(dataset.task.type)
        prompt += self.prompt_builder._build_closing_instructions()

        return prompt

    def _build_per_class_prompt(
        self,
        dataset: Dataset,
        max_rules: int,
        target_class: str,
        counter_examples: list,
    ) -> str:
        """Build focused synthesis prompt for a single class."""
        task_type = dataset.task.type
        is_transformation = task_type == TaskType.TRANSFORMATION

        # Collect positive examples for this class
        positives = []
        for ex in dataset.examples:
            if is_transformation:
                # All examples have the field — show them all
                positives.append(ex)
            elif task_type == TaskType.CLASSIFICATION:
                if ex.expected_output.get("label") == target_class:
                    positives.append(ex)
            elif task_type == TaskType.NER:
                entities = ex.expected_output.get("entities", [])
                if any(e.get("type") == target_class for e in entities):
                    positives.append(ex)

        # Build prompt
        prompt = self.prompt_builder._build_task_header(dataset)

        if is_transformation:
            prompt += f"\n\nFOCUS: Generate rules to extract the '{target_class}' field from the input text.\n"
        else:
            prompt += f"\n\nFOCUS: Generate rules for class '{target_class}' ONLY.\n"

        # Show positive examples
        prompt += (
            f"\nPOSITIVE EXAMPLES for '{target_class}' ({len(positives)} total):\n"
        )
        for ex in positives:
            prompt += f"\nInput: {json.dumps(ex.input)}"
            prompt += f"\nOutput: {json.dumps(ex.expected_output)}"

        # Show counter-examples (not this class)
        if counter_examples and not is_transformation:
            prompt += f"\n\nCOUNTER-EXAMPLES (these are NOT '{target_class}' — your rules must NOT match these):\n"
            for ex in counter_examples:
                prompt += f"\nInput: {json.dumps(ex.input)}"
                prompt += f"\nLabel: {json.dumps(ex.expected_output)}"

        prompt += self.prompt_builder._build_data_evidence(dataset)
        prompt += self.prompt_builder._build_format_instructions(dataset.task.type)

        # Focused task instructions
        prompt += f"""

INSTRUCTIONS:
- Generate up to {max_rules} rules that match examples of '{target_class}'.
- Rules should generalize to unseen text, not just memorize the examples shown.
- Use keyword-based patterns with word boundaries where possible.
- Specific/literal patterns are acceptable as fallbacks for unusual inputs.
"""
        if not is_transformation:
            prompt += "- Rules must NOT match the counter-examples shown above.\n"

        prompt += self.prompt_builder._build_response_schema(dataset)
        prompt += self.prompt_builder._build_format_examples(dataset.task.type)
        prompt += self.prompt_builder._build_closing_instructions()

        return prompt

    def _build_patch_prompt(
        self,
        current_rules: list[Rule],
        failures: list[dict],
        max_rules: int,
        dataset: Dataset | None = None,
        guidance: str = "",
    ) -> str:
        """Build prompt for targeted patch rules."""
        rules_detail = []
        for r in current_rules:
            entry = {
                "name": r.name,
                "description": r.description,
                "format": r.format.value,
                "content": r.content,
                "priority": r.priority,
            }
            if r.output_template:
                entry["output_template"] = r.output_template
            if r.output_key:
                entry["output_key"] = r.output_key
            # Attach rule-level feedback if available
            if dataset:
                rule_fb = dataset.get_feedback_for("rule", r.id)
                if rule_fb:
                    entry["user_feedback"] = [f.text for f in rule_fb]
            rules_detail.append(entry)

        failure_snippets = []
        for f in failures[:20]:
            failure_snippets.append(
                {
                    "input": f.get("input"),
                    "expected": f.get("expected"),
                    "got": f.get("got"),
                    "is_correction": f.get("is_correction", False),
                }
            )

        # Use the schema-aware response format so patch rules include
        # output_template/output_key when needed (NER, TRANSFORMATION).
        if dataset:
            response_schema = self.prompt_builder._build_response_schema(dataset)
            data_evidence = self.prompt_builder._build_data_evidence(dataset)
        else:
            response_schema = """Return JSON:
{
  "analysis": "short reasoning",
  "rules": [
    {
      "name": "rule name",
      "description": "what this rule fixes",
      "format": "regex|code|spacy",
      "content": "pattern or code",
      "priority": 1-10
    }
  ]
}"""
            data_evidence = ""

        # Collect task-level feedback
        task_feedback_section = ""
        if dataset:
            task_fb = dataset.get_feedback_for("task")
            if task_fb:
                lines = "\n".join(f"- {f.text}" for f in task_fb)
                task_feedback_section = (
                    f"\nUSER GUIDANCE (task-level feedback):\n{lines}\n"
                )

        guidance_section = ""
        if guidance:
            guidance_section = (
                f"\nCOORDINATOR GUIDANCE (prioritize this):\n{guidance}\n"
            )

        prompt = f"""You are updating an existing rule-based extractor. Do NOT rewrite good rules; add or adjust only what is needed.

{self.prompt_builder._build_task_header(dataset) if dataset else ""}
{data_evidence}
{task_feedback_section}
{guidance_section}
CURRENT RULES (full details, note any user_feedback on specific rules):
{json.dumps(rules_detail, indent=2)}

FAILURES TO FIX (sampled, corrections are high priority):
{json.dumps(failure_snippets, indent=2)}

Instructions:
- Add or tweak rules to fix the shown failures.
- Pay close attention to user_feedback on rules AND task-level USER GUIDANCE — these are direct instructions from the user and MUST be addressed even if there are no failures.
- If a rule has user_feedback, modify or replace that rule to address the feedback.
- IMPORTANT: When updating an existing rule, you MUST reuse the EXACT same "name" as the original rule. Do NOT add suffixes like "_fixed", "_v2", "_updated", etc. The merge system uses name-matching to replace the old version — a different name creates a duplicate instead of replacing.
- Prefer keyword-based patterns with word boundaries that generalize to unseen text. Specific/literal patterns are OK as fallbacks, but prioritize patterns that capture the concept, not just one example.
- Keep total new/updated rules <= {max_rules}.
- Use formats: {", ".join([f.value for f in self.allowed_formats])}
- Avoid touching unrelated behaviors.

{self.prompt_builder._build_format_instructions(dataset.task.type) if dataset else ""}

{response_schema}
"""
        return prompt

    # ========================================
    # Smart Sampling
    # ========================================

    def _sample_training_data(
        self,
        dataset: Dataset,
        max_samples: int = 100,
        strategy: str = "balanced",
    ):
        """Intelligently sample training data for prompt inclusion."""
        samples = []

        # Priority 1: ALL corrections
        samples.extend(dataset.corrections)

        if len(samples) >= max_samples:
            return samples[:max_samples]

        remaining_budget = max_samples - len(samples)
        examples = dataset.examples

        if not examples:
            return samples[:max_samples]

        if strategy == "balanced":
            samples.extend(examples[:remaining_budget])
        elif strategy == "corrections_first" or strategy == "recent":
            samples.extend(
                sorted(examples, key=lambda e: e.timestamp, reverse=True)[
                    :remaining_budget
                ]
            )
        elif strategy == "diversity":
            if len(examples) <= remaining_budget:
                samples.extend(examples)
            else:
                step = len(examples) // remaining_budget
                samples.extend([examples[i * step] for i in range(remaining_budget)])
        elif strategy == "uncertain":
            sorted_by_confidence = sorted(
                examples, key=lambda e: e.confidence, reverse=False
            )
            samples.extend(sorted_by_confidence[:remaining_budget])
        elif strategy == "varied":
            thirds = remaining_budget // 3
            recent = sorted(examples, key=lambda e: e.timestamp, reverse=True)[:thirds]
            diverse = [
                examples[i * (len(examples) // thirds)]
                for i in range(1, thirds + 1)
                if i * (len(examples) // thirds) < len(examples)
            ]
            uncertain = sorted(examples, key=lambda e: e.confidence, reverse=False)[
                : remaining_budget - len(recent) - len(diverse)
            ]
            samples.extend(recent + diverse + uncertain)

        return samples[:max_samples]

    def _sample_failures(
        self,
        failures: list[dict],
        max_samples: int = 20,
        class_metrics: list | None = None,
    ):
        """Sample failures for refinement/patch, prioritizing corrections and weak classes."""
        correction_failures = [f for f in failures if f.get("is_correction", False)]
        other_failures = [f for f in failures if not f.get("is_correction", False)]

        # Prioritize corrections
        sampled = list(correction_failures)
        remaining = max_samples - len(sampled)

        if remaining > 0 and other_failures:
            # Group by expected label/class
            by_class = defaultdict(list)
            for f in other_failures:
                expected = f.get("expected", {})
                label = (
                    expected.get("label", "")
                    if isinstance(expected, dict)
                    else str(expected)
                )
                by_class[label].append(f)

            # Weight by inverse recall if metrics available (weak classes get more samples)
            class_weights = {}
            if class_metrics:
                for cm in class_metrics:
                    class_weights[cm.label] = max(0.1, 1.0 - cm.recall)

            # Build weighted class order
            classes = list(by_class.keys())
            if class_weights:
                # Sort by weight descending — weakest classes first
                classes.sort(key=lambda c: class_weights.get(c, 0.5), reverse=True)
            else:
                random.shuffle(classes)

            # Round-robin with weighted class order
            idx = 0
            while remaining > 0 and any(by_class.values()):
                cls = classes[idx % len(classes)]
                if by_class[cls]:
                    sampled.append(by_class[cls].pop(0))
                    remaining -= 1
                idx += 1
                classes = [c for c in classes if by_class[c]]
                if not classes:
                    break

        return sampled[:max_samples]

    def _get_classes(self, dataset: Dataset) -> list[str]:
        """Discover classes from dataset based on task type.

        CLASSIFICATION: unique labels from expected_output
        NER: unique entity types from expected_output entities
        TRANSFORMATION: output schema field names
        EXTRACTION: returns [] (no class dimension)
        """
        from rulechef.core import is_pydantic_schema

        task_type = dataset.task.type
        classes = set()

        if task_type == TaskType.CLASSIFICATION:
            for ex in dataset.examples:
                label = ex.expected_output.get("label", "")
                if label:
                    classes.add(label)

        elif task_type == TaskType.NER:
            for ex in dataset.examples:
                entities = ex.expected_output.get("entities", [])
                for ent in entities:
                    ent_type = ent.get("type", "")
                    if ent_type:
                        classes.add(ent_type)

        elif task_type == TaskType.TRANSFORMATION:
            schema = dataset.task.output_schema
            if is_pydantic_schema(schema):
                classes = set(schema.model_fields.keys())
            elif isinstance(schema, dict):
                classes = set(schema.keys())

        return sorted(classes)

    # ========================================
    # Utilities
    # ========================================

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response"""
        if isinstance(text, dict):
            return text

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"\n⚠️ JSON parsing error: {e}")
            print(f"Failed to parse: {preview}")
            # Try to salvage by taking substring between first { and last }
            if "{" in text and "}" in text:
                try:
                    candidate = text[text.index("{") : text.rindex("}") + 1]
                    return json.loads(candidate)
                except Exception:
                    pass
            raise

    def _validate_rule(self, rule: Rule) -> bool:
        """Validate rule syntax"""
        try:
            if rule.format == RuleFormat.REGEX:
                re.compile(rule.content)
            elif rule.format == RuleFormat.CODE:
                compile(rule.content, "<string>", "exec")
                if "def extract(" not in rule.content:
                    print("      Code rule must define extract() function")
                    return False
            elif rule.format == RuleFormat.SPACY:
                pattern_data = self._coerce_spacy_content(rule.content)
                if not isinstance(pattern_data, list) or not pattern_data:
                    return False
                if not self.use_spacy_ner and self._pattern_uses_ent_type(pattern_data):
                    print(
                        "      spaCy NER is disabled; ENT_TYPE/ENT_ID patterns are not allowed"
                    )
                    return False
                rule.content = json.dumps(pattern_data)
            return True
        except Exception as e:
            print(f"      Validation error: {e}")
            return False

    def _generate_id(self) -> str:
        """Generate unique ID"""
        return str(uuid.uuid4())[:8]

    def _pattern_uses_ent_type(self, pattern_data: list) -> bool:
        """Detect spaCy patterns that rely on NER entity types."""

        def _walk(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in ("ENT_TYPE", "ENT_ID"):
                        return True
                    if _walk(v):
                        return True
            elif isinstance(value, list):
                return any(_walk(item) for item in value)
            return False

        return _walk(pattern_data)

    def generate_synthetic_input(self, task, seed: int = 0) -> dict:
        """Generate a synthetic input example"""
        prompt = self.prompt_builder.build_generation_prompt(task, seed)

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.choices[0].message.content
        try:
            text = response_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            result = json.loads(text.strip())
            valid = True
        except Exception:
            result = {"question": "When?", "context": "In 1995"}
            valid = False

        if self.training_logger:
            self.training_logger.log(
                "synthetic_generation",
                [{"role": "user", "content": prompt}],
                response_text,
                {
                    "task_name": task.name if task else None,
                    "task_type": task.type.value if task else None,
                    "seed": seed,
                    "response_valid": valid,
                },
            )

        return result
