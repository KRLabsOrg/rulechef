"""Learning pipeline: rule synthesis, refinement, and audit."""

from __future__ import annotations

import re as re_mod
import time
from typing import TYPE_CHECKING

from rulechef.core import Correction, Example, Rule

if TYPE_CHECKING:
    from rulechef.engine import RuleChef


class LearningPipeline:
    """Orchestrates the learn_rules() pipeline: synthesis, refinement, audit."""

    def __init__(self, chef: RuleChef):
        self._chef = chef

    def run(
        self,
        run_evaluation: bool | None = None,
        min_examples: int = 1,
        max_refinement_iterations: int = 3,
        sampling_strategy: str | None = None,
        incremental_only: bool = False,
    ):
        """Run the full learning pipeline.

        Returns:
            Optional[Tuple[List[Rule], Optional[EvalResult]]]: A tuple of
                (learned_rules, eval_result) on success. eval_result is None
                when refinement is disabled. Returns None if not enough data.
        """
        start_time = time.time()
        chef = self._chef

        # Step 1: Merge pending raw observations into observer
        chef._observations._merge_pending_into_observer()

        # Step 2: Integrate observer (discover task, map pending)
        self._integrate_observer()

        # Step 3: Ensure task is defined
        if chef.task is None:
            raise RuntimeError(
                "Task not defined. Either pass task= to RuleChef(), "
                "call discover_task(), or use start_observing() to enable "
                "auto-discovery."
            )

        # Step 4: Commit buffer to dataset
        self._commit_buffer()

        total_data = len(chef.dataset.get_all_training_data())

        if total_data < min_examples:
            print(f"Need at least {min_examples} examples/corrections")
            print(
                f"Currently have: {len(chef.dataset.corrections)} corrections, "
                f"{len(chef.dataset.examples)} examples"
            )
            return

        # Smart default: disable evaluation for tiny datasets
        if run_evaluation is None:
            run_evaluation = total_data >= 3

        self._print_header(total_data, run_evaluation, max_refinement_iterations, sampling_strategy)

        # Override sampling strategy temporarily
        original_strategy = chef.learner.sampling_strategy
        if sampling_strategy:
            chef.learner.sampling_strategy = sampling_strategy

        # Prevent self-observation during synthesis/refinement
        observer = chef._observations._observer
        if observer:
            observer._skip = True

        try:
            # Step 5: Synthesize rules
            rules = self._synthesize(incremental_only)
            if rules is None:
                return None

            # Step 6: Evaluate and refine
            if run_evaluation:
                rules, eval_result = chef.learner.evaluate_and_refine(
                    rules,
                    chef.dataset,
                    max_iterations=max_refinement_iterations,
                    coordinator=chef.coordinator,
                )
            else:
                eval_result = None

            # Step 7: Save learned rules
            chef.dataset.rules = rules
            chef._store.save(chef.dataset)

            # Step 8: Audit rules
            rules, eval_result = self._audit_rules(rules, eval_result)

            elapsed = time.time() - start_time
            self._print_summary(rules, eval_result, elapsed)

            return rules, eval_result
        finally:
            # Re-enable observation
            if observer:
                observer._skip = False
            # Restore original sampling strategy
            if sampling_strategy:
                chef.learner.sampling_strategy = original_strategy

    def _integrate_observer(self):
        """Discover task and/or map raw observations from observer."""
        chef = self._chef

        # GLiNER auto-task creation
        gliner_obs = chef._observations._gliner_observer
        if gliner_obs and chef.task is None and gliner_obs._observed_count > 0:
            print("\n🔍 Auto-creating task from GLiNER observations...")
            task = gliner_obs.build_task()
            chef._initialize_with_task(task)
            gliner_obs.task = task
            labels = (
                ", ".join(gliner_obs._observed_labels) if gliner_obs._observed_labels else "none"
            )
            print(f"✓ Created task: {task.name} (labels: {labels})")

        # OpenAI observer integration
        observer = chef._observations._observer

        if observer is None:
            return

        observer._skip = True
        try:
            # Auto-discover task if not yet defined
            if chef.task is None:
                obs_stats = observer.get_stats()
                if obs_stats["observed"] == 0:
                    raise RuntimeError(
                        "No observations captured yet. Make some LLM calls "
                        "through the observed client before calling learn_rules()."
                    )
                print("\n🔍 Auto-discovering task from observed LLM calls...")
                task = observer.discover_task(chef.llm, chef.model)
                chef._initialize_with_task(task)
                observer.task = task
                print(f"✓ Discovered task: {task.name} ({task.type.value})")

            # Map raw observations to task schema (auto/mapped modes)
            if not observer._custom_extractors:
                pending = observer.get_stats()["pending"]
                if pending > 0:
                    print(f"\n📋 Mapping {pending} raw observations to task schema...")
                    added = observer.map_pending(chef.task, chef.llm, chef.model)
                    print(f"✓ Mapped {added} observations to buffer")
        finally:
            observer._skip = False

    def _commit_buffer(self):
        """Convert buffered examples to dataset entries."""
        chef = self._chef
        buffer_stats = chef.buffer.get_stats()

        if buffer_stats["new_examples"] == 0:
            return

        print(f"\n📥 Converting {buffer_stats['new_examples']} buffered examples to dataset...")
        print(
            f"   ({buffer_stats['new_corrections']} corrections, "
            f"{buffer_stats['llm_observations']} LLM, "
            f"{buffer_stats['human_examples']} human)"
        )

        for example in chef.buffer.get_new_examples():
            if example.is_correction:
                metadata = getattr(example, "metadata", {}) or {}
                correction_feedback = metadata.get("feedback")
                if isinstance(correction_feedback, str):
                    correction_feedback = correction_feedback.strip() or None

                # Add as Correction to dataset
                correction = Correction(
                    id=chef._store.generate_id(),
                    input=example.input,
                    model_output=example.output.get("actual", {}),
                    expected_output=example.output.get("expected", example.output),
                    feedback=correction_feedback,
                )
                chef.dataset.corrections.append(correction)
            else:
                # Add as Example to dataset
                ex = Example(
                    id=chef._store.generate_id(),
                    input=example.input,
                    expected_output=example.output,
                    source=example.source,
                )
                chef.dataset.examples.append(ex)

        # Mark buffer as processed
        chef.buffer.mark_learned()

        # Save dataset with new examples
        chef._store.save(chef.dataset)

        print(
            f"✓ Converted to dataset: {len(chef.dataset.corrections)} corrections, "
            f"{len(chef.dataset.examples)} examples"
        )

    def _synthesize(self, incremental_only: bool) -> list[Rule] | None:
        """Run rule synthesis (full or incremental)."""
        chef = self._chef

        if incremental_only and chef.dataset.rules:
            print("Incremental mode: patching existing rules")
            pre_eval = chef.learner._evaluate_rules(chef.dataset.rules, chef.dataset)

            # If there's feedback but no failures, create a synthetic
            # failure entry so the patch prompt still fires with feedback
            failures = pre_eval.failures
            has_feedback = len(chef.dataset.structured_feedback) > 0
            if not failures and has_feedback:
                print("  No failures found but feedback exists — forcing refinement with feedback")
                # Use a small sample of training data as context
                sample = chef.dataset.get_all_training_data()[:3]
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

            patch_rules = chef.learner.synthesize_patch_ruleset(
                chef.dataset.rules,
                failures,
                dataset=chef.dataset,
            )
            return self._merge_rules(chef.dataset.rules, patch_rules)
        else:
            # Choose synthesis method based on strategy
            use_per_class = False
            if chef.synthesis_strategy == "per_class":
                use_per_class = True
            elif chef.synthesis_strategy == "auto":
                classes = chef.learner._get_classes(chef.dataset)
                use_per_class = len(classes) > 1

            if use_per_class:
                rules = chef.learner.synthesize_ruleset_per_class(chef.dataset)
            else:
                rules = chef.learner.synthesize_ruleset(chef.dataset)

            if not rules:
                print("Failed to synthesize rules")
                return None

            print(f"✓ Generated {len(rules)} rules")
            return rules

    def _audit_rules(self, rules: list[Rule], eval_result) -> tuple:
        """Run coordinator audit and apply actions with F1 safety net."""
        chef = self._chef
        audit = chef.coordinator.audit_rules(rules, chef.get_rule_metrics(verbose=False))
        if audit.actions:
            rules = self._apply_audit(audit, eval_result)
            eval_result = chef.evaluate(verbose=False) if eval_result else None
        return rules, eval_result

    def _apply_audit(self, audit, eval_result) -> list[Rule]:
        """Apply audit actions (merge/remove) with F1 safety net."""
        chef = self._chef
        pre_audit_rules = list(chef.dataset.rules)
        pre_f1 = eval_result.micro_f1 if eval_result else None
        rules_by_id = {r.id: r for r in chef.dataset.rules}
        changed = False

        for action in audit.actions:
            if action.action == "merge" and len(action.rule_ids) >= 2:
                # Find the source rules
                sources = [rules_by_id[rid] for rid in action.rule_ids if rid in rules_by_id]
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
                    id=chef._store.generate_id(),
                    name=action.merged_name or base.name,
                    description=f"Merged: {action.reason}",
                    format=base.format,
                    content=action.merged_pattern or base.content,
                    priority=base.priority,
                    output_template=base.output_template,
                    output_key=base.output_key,
                )

                # Remove sources, add merged
                chef.dataset.rules = [r for r in chef.dataset.rules if r.id not in action.rule_ids]
                chef.dataset.rules.append(merged)
                rules_by_id = {r.id: r for r in chef.dataset.rules}
                changed = True

            elif action.action == "remove":
                for rid in action.rule_ids:
                    if rid in rules_by_id:
                        chef.dataset.rules = [r for r in chef.dataset.rules if r.id != rid]
                        del rules_by_id[rid]
                        changed = True

        if not changed:
            return chef.dataset.rules

        # Safety net: revert if F1 dropped
        if pre_f1 is not None:
            post_eval = chef.evaluate(verbose=False)
            if post_eval.micro_f1 < pre_f1 - 0.01:
                print(f"⚠ Audit dropped F1 ({pre_f1:.2f} → {post_eval.micro_f1:.2f}), reverting")
                chef.dataset.rules = pre_audit_rules
                chef._store.save(chef.dataset)
                return chef.dataset.rules

        before = len(pre_audit_rules)
        after = len(chef.dataset.rules)
        if after != before:
            print(f"🧹 Audit: {before} → {after} rules")
        chef._store.save(chef.dataset)
        return chef.dataset.rules

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
            r for r in merged if not (r.times_applied >= 3 and r.successes == 0 and r.failures >= 3)
        ]
        if len(pruned) < len(merged):
            print(f"🧹 Pruned {len(merged) - len(pruned)} weak rules")
        return pruned

    def _print_header(
        self, total_data, run_evaluation, max_refinement_iterations, sampling_strategy
    ):
        """Print learning header."""
        chef = self._chef
        print(f"\n{'=' * 60}")
        print(f"Learning rules from {total_data} training items")
        print(f"  Corrections: {len(chef.dataset.corrections)} (high value)")
        print(f"  Examples: {len(chef.dataset.examples)}")
        if run_evaluation:
            print(f"  Mode: Synthesis + Refinement (max {max_refinement_iterations} iterations)")
        else:
            print("  Mode: Synthesis only (no refinement)")

        # Show sampling strategy
        strategy = sampling_strategy or chef.sampling_strategy
        if strategy != "balanced":
            print(f"  Sampling: {strategy}")
        print(f"{'=' * 60}\n")

    def _print_summary(self, rules, eval_result, elapsed):
        """Print learning summary."""
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
                print(f"    {cm.label}: F1={cm.f1:.0%} (P={cm.precision:.0%} R={cm.recall:.0%})")
        print(f"{'=' * 60}\n")
