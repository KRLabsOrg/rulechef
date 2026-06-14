"""LLM-based rule learning"""

import json
import random
import re
import time
import uuid
from collections import defaultdict

from openai import OpenAI

from rulechef.core import Correction, Dataset, Feedback, Rule, RuleFormat, TaskType
from rulechef.evaluation import (
    EvalResult,
    evaluate_dataset,
    evaluate_rules_individually,
)
from rulechef.executor import RuleExecutor
from rulechef.llm_calls import LLMCallConfig, LLMCallManager, PromptTooLargeError, PromptVariant
from rulechef.prompts import RULE_QUALITY_GUIDE, PromptBuilder


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
        max_rules_per_class: int = 5,
        max_counter_examples: int = 10,
        training_logger=None,
        temperature: float | None = None,
        llm_config: LLMCallConfig | None = None,
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
                Applied to per-class positive examples and patch failure sampling.
            max_rules_per_class: Maximum rules to generate per class in per-class synthesis.
            max_counter_examples: Maximum counter-examples from other classes in per-class prompts.
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
        self.max_rules_per_class = max_rules_per_class
        self.max_counter_examples = max_counter_examples
        self.training_logger = training_logger
        self.temperature = temperature
        self.llm_config = llm_config or LLMCallConfig()
        self.llm_calls = LLMCallManager(llm, model, self.llm_config)
        self.executor = RuleExecutor(use_spacy_ner=use_spacy_ner)
        self.prompt_builder = PromptBuilder(
            self.allowed_formats,
            use_spacy_ner=use_spacy_ner,
            use_grex=use_grex,
        )

    def _temp_kwargs(self) -> dict:
        """Return temperature kwarg dict if set, empty dict otherwise."""
        if self.temperature is not None:
            return {"temperature": self.temperature}
        return {}

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
                **self._temp_kwargs(),
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
        max_rules_per_class: int | None = None,
        max_counter_examples: int | None = None,
    ) -> list[Rule]:
        """Synthesize rules one class at a time for better focus and coverage.

        Args:
            dataset: Dataset with training examples.
            max_rules_per_class: Maximum rules to generate for each class.
                Defaults to self.max_rules_per_class.
            max_counter_examples: Maximum counter-examples (other classes) to
                include per class prompt to prevent false positives.
                Defaults to self.max_counter_examples.

        Returns:
            List[Rule] combining rules from all classes. Falls back to bulk
            synthesis if no classes are found.
        """
        max_rules_per_class = max_rules_per_class or self.max_rules_per_class
        max_counter_examples = max_counter_examples or self.max_counter_examples
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
            task_type = dataset.task.type

            # Collect positives and counter-examples for this class
            positives = []
            counter_examples = []
            for ex in dataset.examples:
                if task_type == TaskType.CLASSIFICATION:
                    if ex.expected_output.get("label") == target_class:
                        positives.append(ex)
                    else:
                        counter_examples.append(ex)
                elif task_type == TaskType.NER:
                    entities = ex.expected_output.get("entities", [])
                    if any(e.get("type") == target_class for e in entities):
                        positives.append(ex)
                    else:
                        counter_examples.append(ex)
                elif task_type == TaskType.TRANSFORMATION:
                    positives.append(ex)

            # Sample positives using the configured strategy
            if len(positives) > self.max_samples:
                class_dataset = Dataset(name=dataset.name, task=dataset.task)
                class_dataset.examples = positives
                class_dataset.corrections = [
                    c
                    for c in dataset.corrections
                    if (
                        task_type == TaskType.CLASSIFICATION
                        and c.expected_output.get("label") == target_class
                    )
                    or (
                        task_type == TaskType.NER
                        and any(
                            e.get("type") == target_class
                            for e in c.expected_output.get("entities", [])
                        )
                    )
                    or task_type == TaskType.TRANSFORMATION
                ]
                positives = self._sample_training_data(
                    class_dataset, self.max_samples, self.sampling_strategy
                )

            # Sample counter-examples to keep prompt manageable
            if len(counter_examples) > max_counter_examples:
                rng = random.Random(42 + i)
                counter_examples = rng.sample(counter_examples, max_counter_examples)

            prompt = self._build_per_class_prompt(
                dataset,
                max_rules_per_class,
                target_class=target_class,
                positives=positives,
                counter_examples=counter_examples,
            )

            start = time.time()
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=16384,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    **self._temp_kwargs(),
                )
                response_text = response.choices[0].message.content
                result = self._parse_json(response_text)
                rules = self._parse_rules_from_response(result, max_rules_per_class, dataset)
                all_rules.extend(rules)

                if self.training_logger:
                    self.training_logger.log(
                        "rule_synthesis_per_class",
                        [{"role": "user", "content": prompt}],
                        response_text,
                        {
                            "task_name": dataset.task.name if dataset.task else None,
                            "task_type": dataset.task.type.value if dataset.task else None,
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
                print(f"  [{i + 1}/{len(classes)}] {target_class}: ERROR ({elapsed:.1f}s) - {e}")

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
                if rule.format == RuleFormat.SPACY and not isinstance(rule.content, str):
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
        audit_interval: int = 3,
        holdout_fraction: float = 0.0,
        split_seed: int = 42,
    ) -> tuple:
        """Evaluate rules and refine through patch-based loop.

        Each iteration generates patch rules for failures and merges them
        into the existing set, keeping working rules intact. Stops early
        if exact match reaches 90% or the coordinator signals to stop.

        With holdout_fraction > 0, a stratified dev set is held out of the
        dataset before refinement: failures are still collected from the
        train portion (so patches learn from it), but patch acceptance,
        best-rules tracking, and early stopping are decided on dev metrics.
        This prevents accepting rules that memorize the training data.

        Args:
            rules: Initial set of rules to refine.
            dataset: Dataset to evaluate against.
            max_iterations: Maximum refinement iterations (1-3).
            coordinator: Optional CoordinatorProtocol. If provided, its
                guide_refinement() is called each iteration for LLM-powered
                guidance on which classes to focus and when to stop.
            iteration_callback: Optional callable(iteration: int, rules: List[Rule],
                eval_result: EvalResult) called after each evaluation. Useful for
                logging per-iteration metrics in benchmarks. Receives the dev
                eval when a holdout is active, train eval otherwise.
            audit_interval: Run LLM audit every N iterations to prune duplicates
                and weak rules. Only runs if coordinator supports audit. 0 to disable.
                Overridden by coordinator.audit_interval if set.
            holdout_fraction: Fraction of examples to hold out as a dev set
                (0 disables, keeping the historical evaluate-on-train behavior).
                Corrections always stay in train. The split is skipped when the
                resulting dev set would be too small (see splitting.split_dataset).
            split_seed: Random seed for the stratified split.

        Returns:
            Tuple of (best_rules, best_eval_result) where best_rules is the
            rule set with the highest micro F1 seen across iterations. When a
            holdout is active, best_eval_result is measured on the dev set.
        """
        from rulechef.splitting import split_dataset

        # Read intervals from coordinator if available
        if coordinator and hasattr(coordinator, "audit_interval"):
            audit_interval = coordinator.audit_interval
        critic_interval = getattr(coordinator, "critic_interval", 0) if coordinator else 0

        train_ds, dev_ds = split_dataset(dataset, holdout_fraction, seed=split_seed)
        if holdout_fraction > 0 and dev_ds is None:
            print("⚠ Dataset too small for a dev holdout, refining on training data only")
        if dev_ds is not None:
            print(
                f"\n🔄 Refinement loop (max {max_iterations} iterations, "
                f"train={len(train_ds.get_all_training_data())}, dev={len(dev_ds.examples)})"
            )
        else:
            print(f"\n🔄 Refinement loop (max {max_iterations} iterations)")

        best_rules = rules
        best_f1 = 0.0
        best_eval = EvalResult()

        for iteration in range(max_iterations):
            iter_num = iteration + 1
            print(f"[{iter_num}/{max_iterations}] Evaluating rules...")

            eval_result = self._evaluate_rules(rules, train_ds)
            # Selection metrics come from dev when a holdout is active
            select_eval = self._evaluate_rules(rules, dev_ds) if dev_ds is not None else eval_result
            exact = select_eval.exact_match
            correct = int(exact * select_eval.total_docs)

            if dev_ds is not None:
                print(
                    f"[{iter_num}/{max_iterations}] Train F1: {eval_result.micro_f1:.1%} | "
                    f"Dev exact: {exact:.1%} ({correct}/{select_eval.total_docs}), "
                    f"dev F1: {select_eval.micro_f1:.1%}"
                )
            else:
                print(
                    f"[{iter_num}/{max_iterations}] Exact match: {exact:.1%} "
                    f"({correct}/{select_eval.total_docs}), "
                    f"micro F1: {select_eval.micro_f1:.1%}"
                )

            if select_eval.micro_f1 > best_f1:
                best_rules = rules
                best_f1 = select_eval.micro_f1
                best_eval = select_eval

            # Run critic periodically for strategic feedback
            if (
                critic_interval > 0
                and iteration % critic_interval == 0
                and coordinator
                and hasattr(coordinator, "critique_rules")
            ):
                self._run_critic(rules, train_ds, coordinator, eval_result)

            if iteration_callback:
                iteration_callback(iter_num, rules, select_eval)

            if exact >= 0.90:
                print("✓ Achieved 90%+ exact match!")
                break

            # Periodic audit to consolidate rules mid-refinement
            if (
                audit_interval > 0
                and coordinator
                and iter_num > 1
                and iter_num % audit_interval == 0
                and hasattr(coordinator, "prune_after_learn")
                and coordinator.prune_after_learn
            ):
                rules = self._run_mid_refinement_audit(
                    rules, train_ds, coordinator, eval_result, iter_num
                )
                post_audit = (
                    self._evaluate_rules(rules, dev_ds) if dev_ds is not None else eval_result
                )
                if post_audit.micro_f1 > best_f1:
                    best_rules = rules
                    best_f1 = post_audit.micro_f1
                    best_eval = post_audit

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
                    f"[{iter_num}/{max_iterations}] Patching "
                    f"({len(eval_result.failures)} failures total; "
                    f"sampling ≤{self.max_samples}, ≤20 shown in prompt + mode summary)..."
                )
                start = time.time()
                patch_max = min(self.max_rules, 8)
                patch, deleted_names = self.synthesize_patch_ruleset(
                    rules,
                    eval_result.failures,
                    max_rules=patch_max,
                    dataset=train_ds,
                    guidance=guidance,
                    class_metrics=eval_result.per_class,
                    fp_examples=eval_result.fp_examples,
                )
                elapsed = time.time() - start
                if not patch and not deleted_names:
                    print("⚠ Patch synthesis returned nothing, keeping best rules")
                else:
                    candidate = self._merge_patch(rules, patch, deleted_names)
                    if dev_ds is not None:
                        candidate_eval = self._evaluate_rules(candidate, dev_ds)
                    else:
                        candidate_eval = self._evaluate_rules(candidate, train_ds)
                    prev_f1 = select_eval.micro_f1
                    prev_p = select_eval.micro_precision
                    cand_f1 = candidate_eval.micro_f1
                    cand_p = candidate_eval.micro_precision

                    # Accept if:
                    # - F1 doesn't drop more than 0.5%, OR
                    # - Precision improved (higher precision at cost of some
                    #   recall is a net win for rule quality)
                    # Measured on dev when a holdout is active, so patches
                    # that merely memorize train failures get rejected.
                    accepted = cand_f1 >= prev_f1 - 0.005 or cand_p > prev_p
                    self._log_patch_decision(
                        iter_num,
                        patch,
                        deleted_names,
                        select_eval,
                        candidate_eval,
                        accepted,
                        on_dev=dev_ds is not None,
                        dataset=dataset,
                    )
                    metric_label = "dev " if dev_ds is not None else ""
                    if accepted:
                        rules = candidate
                        print(
                            f"[{iter_num}/{max_iterations}] Patched → {len(rules)} rules, "
                            f"{metric_label}F1 {prev_f1:.1%} → {cand_f1:.1%}, "
                            f"P {prev_p:.1%} → {cand_p:.1%} ({elapsed:.1f}s)"
                        )
                        if candidate_eval.micro_f1 > best_f1:
                            best_rules = rules
                            best_f1 = candidate_eval.micro_f1
                            best_eval = candidate_eval
                    else:
                        print(
                            f"[{iter_num}/{max_iterations}] Patch rejected "
                            f"({metric_label}F1 {prev_f1:.1%} → {cand_f1:.1%}, "
                            f"P {prev_p:.1%} → {cand_p:.1%}), keeping previous"
                        )
            else:
                print("✓ No failures to fix!")
                break

        # Stamp validated per-rule stats so the executor can resolve
        # conflicts by measured precision (dev when available, else train).
        self._stamp_validated_stats(best_rules, dev_ds or train_ds)

        return best_rules, best_eval

    def _stamp_validated_stats(self, rules: list[Rule], dataset: Dataset) -> None:
        """Measure each rule's solo precision and store it on the rule."""
        if not rules:
            return
        try:
            metrics = evaluate_rules_individually(rules, dataset, self._apply_rules, mode="text")
        except Exception as e:
            print(f"⚠ Skipping validated-stat stamping: {e}")
            return
        by_id = {m.rule_id: m for m in metrics}
        for rule in rules:
            m = by_id.get(rule.id)
            if m:
                rule.validated_precision = m.precision
                rule.validated_support = m.true_positives + m.false_positives

    def _log_patch_decision(
        self,
        iteration: int,
        patch: list[Rule],
        deleted_names: set[str],
        prev_eval: EvalResult,
        candidate_eval: EvalResult,
        accepted: bool,
        on_dev: bool,
        dataset: Dataset | None,
    ) -> None:
        """Log an accept/reject decision as a trajectory record.

        These records pair candidate rules with their measured quality delta —
        training data for a future small rule-ranker/rule-writer model.
        """
        if not self.training_logger:
            return
        self.training_logger.log(
            "patch_decision",
            [],
            json.dumps(
                {
                    "patch_rules": [r.to_dict() for r in patch],
                    "deleted_rules": sorted(deleted_names),
                }
            ),
            {
                "task_name": dataset.task.name if dataset and dataset.task else None,
                "task_type": dataset.task.type.value if dataset and dataset.task else None,
                "iteration": iteration,
                "accepted": accepted,
                "decided_on_dev": on_dev,
                "prev_micro_f1": round(prev_eval.micro_f1, 4),
                "prev_micro_precision": round(prev_eval.micro_precision, 4),
                "candidate_micro_f1": round(candidate_eval.micro_f1, 4),
                "candidate_micro_precision": round(candidate_eval.micro_precision, 4),
                "f1_delta": round(candidate_eval.micro_f1 - prev_eval.micro_f1, 4),
            },
        )

    def _run_mid_refinement_audit(
        self,
        rules: list[Rule],
        dataset: Dataset,
        coordinator,
        eval_result: EvalResult,
        iter_num: int,
    ) -> list[Rule]:
        """Run LLM audit mid-refinement to consolidate rules."""
        print(f"[{iter_num}] Running mid-refinement audit ({len(rules)} rules)...")
        rule_metrics = evaluate_rules_individually(rules, dataset, self._apply_rules, mode="text")
        audit = coordinator.audit_rules(rules, rule_metrics)
        if not audit.actions:
            return rules

        # Apply audit actions inline
        pre_f1 = eval_result.micro_f1
        pre_rules = list(rules)
        rules_by_id = {r.id: r for r in rules}

        for action in audit.actions:
            if action.action == "merge" and len(action.rule_ids) >= 2:
                sources = [rules_by_id[rid] for rid in action.rule_ids if rid in rules_by_id]
                if len(sources) < 2:
                    continue
                if action.merged_pattern:
                    try:
                        re.compile(action.merged_pattern)
                    except re.error:
                        continue
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
                rules = [r for r in rules if r.id not in action.rule_ids]
                rules.append(merged)
                rules_by_id = {r.id: r for r in rules}

            elif action.action == "remove":
                for rid in action.rule_ids:
                    if rid in rules_by_id:
                        rules = [r for r in rules if r.id != rid]
                        del rules_by_id[rid]

        # Safety net
        post_eval = self._evaluate_rules(rules, dataset)
        if post_eval.micro_f1 < pre_f1 - 0.01:
            print(f"⚠ Mid-audit dropped F1 ({pre_f1:.2f} → {post_eval.micro_f1:.2f}), reverting")
            return pre_rules

        print(f"🧹 Mid-audit: {len(pre_rules)} → {len(rules)} rules")
        return rules

    def _run_critic(self, rules, dataset, coordinator, eval_result):
        """Run critic agent — writes feedback to dataset like a human domain expert."""
        rule_metrics = evaluate_rules_individually(rules, dataset, self._apply_rules, mode="text")
        critique = coordinator.critique_rules(rules, rule_metrics, eval_result, dataset)
        if not critique:
            return

        # Clear previous critic feedback (refresh each learning cycle)
        dataset.structured_feedback = [
            f for f in dataset.structured_feedback if f.source != "critic"
        ]
        added_rule = 0
        added_task = 0

        # Write rule-level feedback (same as chef.add_feedback would)
        for rule_id, text in critique.get("rule_feedback", {}).items():
            fb = Feedback(
                id=self._generate_id(),
                text=text,
                level="rule",
                target_id=rule_id,
                source="critic",
            )
            dataset.structured_feedback.append(fb)
            added_rule += 1

        # Write task-level feedback
        if critique.get("task_guidance"):
            fb = Feedback(
                id=self._generate_id(),
                text=critique["task_guidance"],
                level="task",
                source="critic",
            )
            dataset.structured_feedback.append(fb)
            dataset.feedback.append(critique["task_guidance"])
            added_task += 1

        analysis = critique.get("analysis", "")
        print(f"📝 Critic: {analysis}")
        if added_rule or added_task:
            print(f"   Added {added_rule} rule-level + {added_task} task-level feedback")

    @staticmethod
    def _merge_patch(
        existing: list[Rule], patches: list[Rule], deleted_names: set[str] | None = None
    ) -> list[Rule]:
        """Merge patch rules into existing set by name, with optional deletions."""
        by_name = {r.name: r for r in existing}

        # Remove deleted rules first
        if deleted_names:
            actually_deleted = [n for n in deleted_names if n in by_name]
            for name in actually_deleted:
                del by_name[name]
            if actually_deleted:
                print(
                    f"🗑️ Patch deleted {len(actually_deleted)} rules: {', '.join(actually_deleted)}"
                )

        for pr in patches:
            if pr.name in by_name:
                current = by_name[pr.name]
                pr.times_applied = current.times_applied
                pr.successes = current.successes
                pr.failures = current.failures
                pr.confidence = current.confidence
            by_name[pr.name] = pr
        merged = list(by_name.values())
        return RuleLearner._dedup_rules(merged)

    @staticmethod
    def _dedup_rules(rules: list[Rule]) -> list[Rule]:
        """Remove duplicate regex rules with identical content and output type."""
        seen: dict[tuple, Rule] = {}  # key: (content, output_type, output_key)
        result = []
        for r in rules:
            if r.format != RuleFormat.REGEX:
                result.append(r)
                continue
            # Normalize: strip whitespace
            content = r.content.strip()
            output_type = (r.output_template or {}).get("type", "")
            key = (content, output_type, r.output_key or "")
            if key in seen:
                # Keep higher priority, or the one with more successes
                existing = seen[key]
                if r.priority > existing.priority or (
                    r.priority == existing.priority and r.successes > existing.successes
                ):
                    # Replace
                    result = [x for x in result if x.id != existing.id]
                    result.append(r)
                    seen[key] = r
                # else skip this duplicate
            else:
                seen[key] = r
                result.append(r)
        deduped = len(rules) - len(result)
        if deduped:
            print(f"🧹 Deduped {deduped} identical rules")
        return result

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
        fp_examples: list[dict] | None = None,
    ) -> tuple[list[Rule], set[str]]:
        """Generate incremental rules targeted at specific failures.

        Returns only new/updated rules; the caller is responsible for
        merging them into the existing rule set.

        Returns:
            Tuple of (patch_rules, deleted_names) where deleted_names is
            a set of rule names the LLM wants removed.
        """
        max_rules = max_rules or self.max_rules
        task_type = dataset.task.type if dataset and dataset.task else None
        sampled_failures = self._sample_failures(
            failures,
            max_samples=self.max_samples,
            class_metrics=class_metrics,
            task_type=task_type,
        )
        failure_summary = self._failure_mode_summary(failures, task_type)
        variants = self._build_patch_prompt_variants(
            current_rules,
            sampled_failures,
            max_rules,
            dataset=dataset,
            guidance=guidance,
            class_metrics=class_metrics,
            fp_examples=fp_examples,
            failure_summary=failure_summary,
        )

        print("🩹 Synthesizing patch rules...")
        start = time.time()

        try:
            call_result = self.llm_calls.complete_with_variants(
                variants,
                output_tokens=self.llm_config.patch_output_tokens,
                extra_kwargs=self._temp_kwargs(),
            )
            response_text = call_result.response_text
            result = self._parse_json(response_text)
            rules = self._parse_rules_from_response(result, max_rules, dataset=dataset)
            deleted_names = set(result.get("deleted_rules", [])) if result else set()

            if self.training_logger:
                self.training_logger.log(
                    "rule_patch",
                    call_result.messages,
                    response_text,
                    {
                        "task_name": dataset.task.name if dataset and dataset.task else None,
                        "task_type": dataset.task.type.value if dataset and dataset.task else None,
                        "num_failures": len(sampled_failures),
                        "num_existing_rules": len(current_rules),
                        "num_rules_in_response": len(rules),
                        "num_deleted": len(deleted_names),
                        "response_valid": bool(rules),
                        "guidance": guidance[:200] if guidance else None,
                        "prompt_variant": call_result.variant.name,
                        **call_result.variant.metadata,
                        **call_result.metadata,
                    },
                )

            elapsed = time.time() - start
            msg = f"✓ Patch synthesis returned {len(rules)} rules"
            if deleted_names:
                msg += f", {len(deleted_names)} deletions"
            print(f"{msg} ({elapsed:.1f}s)")
            return rules, deleted_names
        except PromptTooLargeError as e:
            print(f"⚠ Patch prompt too large: {e}")
            return [], set()
        except Exception as e:
            print(f"Error synthesizing patch rules: {e}")
            return [], set()

    # ========================================
    # Prompt Building
    # ========================================

    def _build_synthesis_prompt(
        self,
        dataset: Dataset,
        max_rules: int,
    ) -> str:
        """Build prompt for bulk rule synthesis using PromptBuilder."""
        # Bulk synthesis path
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
        positives: list,
        counter_examples: list,
    ) -> str:
        """Build focused synthesis prompt for a single class."""
        task_type = dataset.task.type
        is_transformation = task_type == TaskType.TRANSFORMATION

        # Build prompt
        prompt = self.prompt_builder._build_task_header(dataset)

        if is_transformation:
            prompt += f"\n\nFOCUS: Generate rules to extract the '{target_class}' field from the input text.\n"
        else:
            prompt += f"\n\nFOCUS: Generate rules for class '{target_class}' ONLY.\n"

        # Show positive examples
        prompt += f"\nPOSITIVE EXAMPLES for '{target_class}' ({len(positives)} total):\n"
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

{RULE_QUALITY_GUIDE}

INSTRUCTIONS:
- Generate up to {max_rules} rules that match examples of '{target_class}'.
- Rules should generalize to unseen text, not just memorize the examples shown.
- Use structural patterns with word boundaries that generalize to unseen text.
"""
        if not is_transformation:
            prompt += "- Rules must NOT match the counter-examples shown above.\n"

        prompt += self.prompt_builder._build_response_schema(dataset)
        prompt += self.prompt_builder._build_format_examples(dataset.task.type)
        prompt += self.prompt_builder._build_closing_instructions()

        return prompt

    def _build_patch_prompt_variants(
        self,
        current_rules: list[Rule],
        failures: list[dict],
        max_rules: int,
        dataset: Dataset | None = None,
        guidance: str = "",
        class_metrics: list | None = None,
        fp_examples: list[dict] | None = None,
        failure_summary: str = "",
    ) -> list[PromptVariant]:
        """Build ordered patch prompt variants from richest to most compact."""
        specs = [
            ("full", 20, True, "full"),
            ("fewer_failures", 10, True, "full"),
            ("minimal_failures", 5, True, "full"),
            ("no_data_evidence", 5, False, "full"),
            ("relevant_rules", 5, False, "relevant"),
            ("compact_rules", 5, False, "compact"),
        ]

        variants = []
        for name, failure_limit, include_data_evidence, rules_mode in specs:
            prompt = self._build_patch_prompt(
                current_rules,
                failures,
                max_rules,
                dataset=dataset,
                guidance=guidance,
                class_metrics=class_metrics,
                fp_examples=fp_examples,
                failure_limit=failure_limit,
                include_data_evidence=include_data_evidence,
                rules_mode=rules_mode,
                failure_summary=failure_summary,
            )
            variants.append(
                PromptVariant(
                    name=name,
                    prompt=prompt,
                    metadata={
                        "failure_limit": failure_limit,
                        "include_data_evidence": include_data_evidence,
                        "rules_mode": rules_mode,
                    },
                )
            )
        return variants

    def _failure_labels(
        self, failures: list[dict], fp_examples: list[dict] | None = None
    ) -> set[str]:
        labels: set[str] = set()

        def add_from_output(output):
            if isinstance(output, dict):
                label = output.get("label") or output.get("type")
                if isinstance(label, str):
                    labels.add(label)
                for key in ("entities", "spans"):
                    items = output.get(key)
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                item_label = item.get("type") or item.get("label")
                                if isinstance(item_label, str):
                                    labels.add(item_label)

        for failure in failures:
            add_from_output(failure.get("expected"))
            add_from_output(failure.get("got"))

        for fp in fp_examples or []:
            for key in ("predicted_type", "correct_type"):
                value = fp.get(key)
                if isinstance(value, str):
                    labels.add(value)
        return labels

    def _rule_labels(self, rule: Rule) -> set[str]:
        labels: set[str] = set()
        if isinstance(rule.output_template, dict):
            for key in ("label", "type"):
                value = rule.output_template.get(key)
                if isinstance(value, str) and not value.startswith("$"):
                    labels.add(value)
        return labels

    def _rule_prompt_entry(
        self,
        rule: Rule,
        dataset: Dataset | None = None,
        *,
        compact: bool = False,
        truncate: bool = False,
    ) -> dict:
        entry = {
            "name": rule.name,
            "format": rule.format.value,
            "priority": rule.priority,
        }
        if rule.output_template:
            entry["output_template"] = rule.output_template
        if rule.output_key:
            entry["output_key"] = rule.output_key

        if compact:
            labels = sorted(self._rule_labels(rule))
            if labels:
                entry["labels"] = labels
            return entry

        entry["description"] = rule.description[:160] if truncate else rule.description
        entry["content"] = rule.content[:300] if truncate else rule.content

        if dataset:
            rule_fb = dataset.get_feedback_for("rule", rule.id)
            if rule_fb:
                entry["user_feedback"] = [f.text for f in rule_fb]
        return entry

    def _build_patch_prompt(
        self,
        current_rules: list[Rule],
        failures: list[dict],
        max_rules: int,
        dataset: Dataset | None = None,
        guidance: str = "",
        class_metrics: list | None = None,
        fp_examples: list[dict] | None = None,
        failure_limit: int = 20,
        include_data_evidence: bool = True,
        rules_mode: str = "full",
        failure_summary: str = "",
    ) -> str:
        """Build prompt for targeted patch rules."""
        rules_detail = []
        compact_rules = []
        failure_labels = self._failure_labels(failures, fp_examples)

        for r in current_rules:
            has_feedback = bool(dataset and dataset.get_feedback_for("rule", r.id))
            labels = self._rule_labels(r)
            is_relevant = not failure_labels or bool(labels & failure_labels) or has_feedback

            if rules_mode == "relevant" and not is_relevant:
                compact_rules.append(self._rule_prompt_entry(r, dataset, compact=True))
                continue

            rules_detail.append(
                self._rule_prompt_entry(
                    r,
                    dataset,
                    compact=False,
                    truncate=rules_mode == "compact",
                )
            )

        failure_snippets = []
        for f in failures[:failure_limit]:
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
            data_evidence = (
                self.prompt_builder._build_data_evidence(dataset) if include_data_evidence else ""
            )
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
  ],
  "deleted_rules": ["name_of_rule_to_remove"]
}"""
            data_evidence = ""

        # Collect task-level feedback
        task_feedback_section = ""
        if dataset:
            task_fb = dataset.get_feedback_for("task")
            if task_fb:
                lines = "\n".join(f"- {f.text}" for f in task_fb)
                task_feedback_section = f"\nUSER GUIDANCE (task-level feedback):\n{lines}\n"

        guidance_section = ""
        if guidance:
            guidance_section = f"\nCOORDINATOR GUIDANCE (prioritize this):\n{guidance}\n"

        # Build per-class metrics summary so the LLM knows which classes have problems
        class_metrics_section = ""
        if class_metrics:
            lines = ["\nPER-CLASS METRICS (current performance):"]
            for cm in class_metrics:
                lines.append(
                    f"  {cm.label}: P={cm.precision:.0%} R={cm.recall:.0%} F1={cm.f1:.0%} "
                    f"(TP={cm.tp} FP={cm.fp} FN={cm.fn})"
                )
            class_metrics_section = "\n".join(lines) + "\n"

        # Build FP examples section
        fp_section = ""
        if fp_examples:
            fp_lines = [
                "\nFALSE POSITIVES (rules are incorrectly matching these — tighten the responsible rules):"
            ]
            for fp in fp_examples:
                line = f'  Predicted "{fp["predicted_text"]}" as {fp["predicted_type"]}'
                if fp.get("correct_type"):
                    line += f" — should be {fp['correct_type']}"
                else:
                    line += " — not an entity"
                fp_lines.append(line)
            fp_section = "\n".join(fp_lines) + "\n"

        compact_rules_section = ""
        if compact_rules:
            compact_rules_section = (
                "\nOTHER CURRENT RULES (compact index; avoid duplicating these names/labels):\n"
                f"{json.dumps(compact_rules, indent=2)}\n"
            )

        prompt = f"""You are updating an existing rule-based extractor. Do NOT rewrite good rules; add or adjust only what is needed.

{self.prompt_builder._build_task_header(dataset) if dataset else ""}
{data_evidence}
{task_feedback_section}
{guidance_section}
{class_metrics_section}
CURRENT RULES (full details, note any user_feedback on specific rules):
{json.dumps(rules_detail, indent=2)}
{compact_rules_section}

{failure_summary}
FAILURES TO FIX (sampled, corrections are high priority):
{json.dumps(failure_snippets, indent=2)}
{fp_section}
Instructions:
- Add, tweak, or DELETE rules to fix the shown failures and reduce false positives.
- Pay close attention to user_feedback on rules AND task-level USER GUIDANCE — these are direct instructions from the user and MUST be addressed even if there are no failures.
- If a rule has user_feedback, modify or replace that rule to address the feedback.
- IMPORTANT: When updating an existing rule, you MUST reuse the EXACT same "name" as the original rule. Do NOT add suffixes like "_fixed", "_v2", "_updated", etc. The merge system uses name-matching to replace the old version — a different name creates a duplicate instead of replacing.
- If a rule is fundamentally too broad (FP >> TP) and you're providing better, narrower replacements, list the old rule's exact name in "deleted_rules". Only delete if you're providing replacements in "rules".
- If a rule has high false positives (FP >> TP), TIGHTEN its pattern or DELETE it and add narrower replacements. Adding context or narrowing the match is better than piling on new rules.
- Use structural patterns with word boundaries that generalize to unseen text.
- Keep total new/updated rules <= {max_rules}.
- Use formats: {", ".join([f.value for f in self.allowed_formats])}
- Avoid touching unrelated behaviors.

{RULE_QUALITY_GUIDE}

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
                sorted(examples, key=lambda e: e.timestamp, reverse=True)[:remaining_budget]
            )
        elif strategy == "diversity":
            if len(examples) <= remaining_budget:
                samples.extend(examples)
            else:
                step = len(examples) // remaining_budget
                samples.extend([examples[i * step] for i in range(remaining_budget)])
        elif strategy == "uncertain":
            sorted_by_confidence = sorted(examples, key=lambda e: e.confidence, reverse=False)
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

    def _failure_signature(self, failure: dict, task_type: TaskType | None) -> tuple[str, str]:
        """Classify a failure into a (expected_class, error_kind) failure mode.

        Failure modes separate "rule missing" from "wrong rule fired" so
        sampling and prompts can target each mode instead of treating all
        failures for a class as interchangeable.
        """
        expected = failure.get("expected") or {}
        got = failure.get("got") or {}

        if task_type == TaskType.CLASSIFICATION:
            exp_label = str(expected.get("label", "")) if isinstance(expected, dict) else ""
            got_label = str(got.get("label", "")) if isinstance(got, dict) else ""
            if not got_label:
                return exp_label, "no_prediction"
            return exp_label, f"predicted_as:{got_label}"

        def type_counts(output) -> dict[str, int]:
            counts: dict[str, int] = defaultdict(int)
            if isinstance(output, dict):
                for key in ("entities", "spans"):
                    for item in output.get(key) or []:
                        if isinstance(item, dict):
                            counts[item.get("type") or item.get("label") or "_untyped"] += 1
            return counts

        exp_types = type_counts(expected)
        got_types = type_counts(got)
        missed = sorted(t for t in exp_types if exp_types[t] > got_types.get(t, 0))
        spurious = sorted(t for t in got_types if got_types[t] > exp_types.get(t, 0))

        cls = missed[0] if missed else (spurious[0] if spurious else "_none")
        if missed and spurious:
            return cls, f"missed:{'+'.join(missed)}|spurious:{'+'.join(spurious)}"
        if missed:
            return cls, f"missed:{'+'.join(missed)}"
        if spurious:
            return cls, f"spurious:{'+'.join(spurious)}"
        return cls, "mismatch"

    def _cluster_failures(
        self, failures: list[dict], task_type: TaskType | None
    ) -> dict[tuple[str, str], list[dict]]:
        """Group failures by (expected_class, error_kind) failure mode."""
        clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for f in failures:
            clusters[self._failure_signature(f, task_type)].append(f)
        return clusters

    def _failure_mode_summary(
        self,
        failures: list[dict],
        task_type: TaskType | None,
        max_modes: int = 20,
    ) -> str:
        """Aggregate failure-mode counts over ALL failures.

        Only a handful of concrete failures fit in a patch prompt; this
        summary tells the LLM how the full failure distribution looks so it
        targets the dominant modes instead of the sampled tail.
        """
        clusters = self._cluster_failures(failures, task_type)
        if not clusters:
            return ""
        ordered = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
        lines = [f"\nFAILURE MODE SUMMARY (all {len(failures)} failures, not just the sample):"]
        for (cls, kind), items in ordered[:max_modes]:
            lines.append(f"  {cls} — {kind}: {len(items)}")
        if len(ordered) > max_modes:
            remaining = sum(len(items) for _, items in ordered[max_modes:])
            lines.append(f"  ... and {len(ordered) - max_modes} more modes ({remaining} failures)")
        return "\n".join(lines) + "\n"

    def _sample_failures(
        self,
        failures: list[dict],
        max_samples: int = 20,
        class_metrics: list | None = None,
        task_type: TaskType | None = None,
    ):
        """Sample failures for refinement/patch, prioritizing corrections and weak modes.

        Failures are clustered by (expected_class, error_kind) and sampled
        round-robin across clusters, so every failure mode present in the
        data gets representation in the patch prompt — not just the most
        frequent class.
        """
        correction_failures = [f for f in failures if f.get("is_correction", False)]
        other_failures = [f for f in failures if not f.get("is_correction", False)]

        # Prioritize corrections
        sampled = list(correction_failures)
        remaining = max_samples - len(sampled)

        if remaining > 0 and other_failures:
            clusters = self._cluster_failures(other_failures, task_type)

            # Weight by inverse recall if metrics available (weak classes get more samples)
            class_weights = {}
            if class_metrics:
                for cm in class_metrics:
                    class_weights[cm.label] = max(0.1, 1.0 - cm.recall)

            def cluster_weight(key: tuple[str, str]) -> float:
                cls, _kind = key
                return len(clusters[key]) * class_weights.get(cls, 0.5)

            keys = sorted(clusters.keys(), key=cluster_weight, reverse=True)
            if not class_weights:
                random.Random(42).shuffle(keys)

            # Round-robin across failure-mode clusters
            idx = 0
            while remaining > 0 and keys:
                key = keys[idx % len(keys)]
                if clusters[key]:
                    sampled.append(clusters[key].pop(0))
                    remaining -= 1
                idx += 1
                keys = [k for k in keys if clusters[k]]

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
            # Response truncated mid-stream (hit output token limit): recover
            # as many complete rule objects as possible instead of losing all.
            recovered = self._recover_truncated_rules(text)
            if recovered is not None:
                print(f"   ↻ Recovered {len(recovered.get('rules', []))} rules from truncated JSON")
                return recovered
            raise

    @staticmethod
    def _recover_truncated_rules(text: str) -> dict | None:
        """Extract complete rule objects from a truncated ``"rules": [...]`` array.

        When the model hits its output token budget, the final rule object and
        closing brackets are cut off, so json.loads fails on the whole payload.
        This walks the rules array and keeps every fully-balanced object,
        discarding the incomplete tail.

        Returns a dict like {"rules": [...], "deleted_rules": [...]} or None
        if no rules array is present.
        """
        marker = '"rules"'
        idx = text.find(marker)
        if idx == -1:
            return None
        bracket = text.find("[", idx)
        if bracket == -1:
            return None

        objects = []
        depth = 0
        in_string = False
        escaped = False
        start = None
        for i in range(bracket + 1, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    fragment = text[start : i + 1]
                    try:
                        objects.append(json.loads(fragment))
                    except json.JSONDecodeError:
                        pass
                    start = None
            elif ch == "]" and depth == 0:
                break

        if not objects:
            return None

        # Recover an explicit deleted_rules array if it parsed before truncation
        deleted: list[str] = []
        del_match = re.search(r'"deleted_rules"\s*:\s*(\[[^\]]*\])', text)
        if del_match:
            try:
                deleted = json.loads(del_match.group(1))
            except json.JSONDecodeError:
                deleted = []

        return {"rules": objects, "deleted_rules": deleted}

    # Diverse non-empty probe strings used to detect catch-all regex patterns.
    # Empty string is excluded on purpose: a rule like ``^.{1,}$`` matches every
    # real (non-empty) input and is just as degenerate as ``.*``.
    _CATCH_ALL_PROBES = (
        "hello world this is a benign sentence",
        "1234567890",
        "!@#$%^&*()",
        "the quick brown fox jumps over the lazy dog",
        "élève naïve façade",
        "a",
    )

    def _is_catch_all_regex(self, pattern: str) -> bool:
        """Detect regex patterns that match essentially any input.

        A rule like ``.*`` or ``^[\\s\\S]{0,1000}$`` matches every document,
        so it degenerates into a constant base-rate predictor — it passes
        i.i.d. dev checks (it just predicts the majority class) and collapses
        under distribution shift. Such rules are rejected at validation.

        Pure negative-lookahead rules (``^(?!.*\\bkill\\b).*$``) are also
        flagged: they match every input except ones containing a niche
        blocked term, which is the same degenerate near-constant behavior.
        A lookahead paired with a *positive* anchor (``(?!.*kill).*\\bweather\\b``)
        only matches relevant inputs, so it escapes the probes and is kept.
        """
        try:
            compiled = re.compile(pattern)
        except re.error:
            return False
        return all(compiled.search(probe) is not None for probe in self._CATCH_ALL_PROBES)

    def _validate_rule(self, rule: Rule) -> bool:
        """Validate rule syntax"""
        try:
            if rule.format == RuleFormat.REGEX:
                re.compile(rule.content)
                if self._is_catch_all_regex(rule.content):
                    print(
                        f"      Rejected catch-all regex (matches any input): {rule.content[:60]!r}"
                    )
                    return False
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
                    print("      spaCy NER is disabled; ENT_TYPE/ENT_ID patterns are not allowed")
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
            **self._temp_kwargs(),
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
