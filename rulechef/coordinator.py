"""Coordination layer for learning decisions - swappable simple/agentic implementations"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rulechef.buffer import ExampleBuffer
    from rulechef.core import Rule


@dataclass
class AuditAction:
    """A single audit action: remove or merge."""

    action: str  # "remove" or "merge"
    rule_ids: List[str]  # IDs involved (1 for remove, 2+ for merge)
    reason: str
    merged_pattern: Optional[str] = None  # New pattern for merges
    merged_name: Optional[str] = None


@dataclass
class AuditResult:
    """Result of a rule audit."""

    actions: List[AuditAction] = field(default_factory=list)
    analysis: str = ""


@dataclass
class CoordinationDecision:
    """Result of coordinator analysis - explains what/why/how to learn"""

    should_learn: bool
    strategy: str  # Sampling strategy to use
    reasoning: str  # Human-readable explanation
    max_iterations: int = 3
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CoordinatorProtocol(ABC):
    """
    Abstract interface for learning coordination.

    Implementations can be simple (heuristics) or agentic (LLM-powered).
    RuleChef uses this interface, making coordinators swappable.
    """

    @abstractmethod
    def should_trigger_learning(
        self, buffer: "ExampleBuffer", current_rules: Optional[List["Rule"]]
    ) -> CoordinationDecision:
        """
        Decide if learning should be triggered now.

        Args:
            buffer: Current example buffer
            current_rules: Currently learned rules (None if first learn)

        Returns:
            CoordinationDecision with should_learn, strategy, reasoning
        """
        pass

    @abstractmethod
    def analyze_buffer(self, buffer: "ExampleBuffer") -> Dict[str, Any]:
        """
        Analyze current buffer state.

        Returns:
            Dict with buffer statistics and insights
        """
        pass

    @abstractmethod
    def on_learning_complete(
        self,
        old_rules: Optional[List["Rule"]],
        new_rules: List["Rule"],
        metrics: Dict[str, Any],
    ):
        """
        Callback after learning completes.

        Args:
            old_rules: Rules before learning (None if first learn)
            new_rules: Newly learned rules
            metrics: Learning metrics (accuracy, etc.)
        """
        pass

    def guide_refinement(
        self, eval_result: Any, iteration: int, max_iterations: int
    ) -> tuple:
        """Analyze per-class metrics and return (guidance_text, should_continue).

        Called after each refinement iteration. The guidance string is injected
        into the patch prompt. should_continue=False stops the loop early.

        Default: no guidance, always continue.
        """
        return "", True

    def audit_rules(self, rules: List["Rule"], rule_metrics: List[Any]) -> AuditResult:
        """Audit rules for redundancy, dead rules, and conflicts.

        Called after learning completes when pruning is enabled.
        Returns an AuditResult with actions (remove/merge).
        The engine applies actions and reverts if performance drops.

        Args:
            rules: Current learned rules.
            rule_metrics: Per-rule RuleMetrics from evaluate_rules_individually.

        Returns:
            AuditResult with actions to take.
        """
        return AuditResult()


class SimpleCoordinator(CoordinatorProtocol):
    """
    Deterministic heuristic-based coordinator.

    Uses simple rules to make decisions:
    - First learn: trigger after N examples
    - Subsequent: trigger after N examples OR M corrections
    - Strategy selection: corrections_first if corrections, else balanced/diversity
    """

    def __init__(
        self,
        trigger_threshold: int = 50,
        correction_threshold: int = 10,
        verbose: bool = True,
    ):
        """
        Args:
            trigger_threshold: Number of examples needed to trigger learning
            correction_threshold: Number of corrections to trigger early learning
            verbose: Print coordination decisions
        """
        self.trigger_threshold = trigger_threshold
        self.correction_threshold = correction_threshold
        self.verbose = verbose

    def should_trigger_learning(
        self, buffer: "ExampleBuffer", current_rules: Optional[List["Rule"]]
    ) -> CoordinationDecision:
        """Simple heuristic decision"""
        stats = buffer.get_stats()
        new_examples_count = stats["new_examples"]
        corrections_count = stats["new_corrections"]

        # First learn: need enough examples
        if current_rules is None:
            should_learn = new_examples_count >= self.trigger_threshold
            reasoning = (
                f"First learn: {new_examples_count}/{self.trigger_threshold} examples"
            )
            strategy = "balanced"  # Start with balanced sampling
            max_iterations = 3

        # Subsequent learns
        else:
            # Trigger if enough examples OR enough corrections (high-value signal)
            should_learn = (
                new_examples_count >= self.trigger_threshold
                or corrections_count >= self.correction_threshold
            )

            if corrections_count >= self.correction_threshold:
                reasoning = f"Corrections accumulated: {corrections_count}/{self.correction_threshold}"
                strategy = "corrections_first"  # Focus on fixing mistakes
                max_iterations = 2  # Faster refinement for corrections
            elif new_examples_count >= self.trigger_threshold:
                reasoning = f"Examples accumulated: {new_examples_count}/{self.trigger_threshold}"
                strategy = "diversity"  # Explore new patterns
                max_iterations = 3
            else:
                reasoning = f"Not ready: {new_examples_count}/{self.trigger_threshold} examples, {corrections_count}/{self.correction_threshold} corrections"
                strategy = "balanced"
                max_iterations = 3

        if self.verbose and should_learn:
            print(f"\n🔄 Coordinator decision: {reasoning}")
            print(f"   Strategy: {strategy}, max iterations: {max_iterations}")

        return CoordinationDecision(
            should_learn=should_learn,
            strategy=strategy,
            reasoning=reasoning,
            max_iterations=max_iterations,
            metadata={
                "buffer_stats": stats,
                "trigger_threshold": self.trigger_threshold,
                "correction_threshold": self.correction_threshold,
            },
        )

    def analyze_buffer(self, buffer: "ExampleBuffer") -> Dict[str, Any]:
        """Basic buffer statistics"""
        stats = buffer.get_stats()
        return {
            **stats,
            "ready_for_first_learn": stats["new_examples"] >= self.trigger_threshold,
            "ready_for_refinement": (
                stats["new_examples"] >= self.trigger_threshold
                or stats["new_corrections"] >= self.correction_threshold
            ),
        }

    def on_learning_complete(
        self,
        old_rules: Optional[List["Rule"]],
        new_rules: List["Rule"],
        metrics,
    ):
        """Log learning results. metrics is an EvalResult or None."""
        if self.verbose:
            if old_rules is None:
                print("✓ Initial learning complete:")
            else:
                print("✓ Refinement complete:")

            print(f"  {len(new_rules)} rules")
            if metrics and hasattr(metrics, "exact_match"):
                print(
                    f"  Exact match: {metrics.exact_match:.1%}, F1: {metrics.micro_f1:.1%}"
                )


# Placeholder for future agentic implementation
class AgenticCoordinator(CoordinatorProtocol):
    """
    LLM-based intelligent coordinator.

    Uses LLM to make adaptive decisions:
    - Analyze buffer patterns to detect when learning would be beneficial
    - Choose optimal sampling strategy based on data characteristics
    - Decide iteration count based on learning progress
    - Provide detailed reasoning for decisions
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "gpt-4o-mini",
        min_batch_size: int = 5,
        min_correction_batch: int = 1,
        verbose: bool = True,
        prune_after_learn: bool = False,
    ):
        """
        Args:
            llm_client: OpenAI client
            model: Model to use for coordination
            min_batch_size: Minimum new examples before asking LLM
            min_correction_batch: Minimum corrections before asking LLM
            verbose: Print coordination decisions
            prune_after_learn: If True, audit and prune/merge rules after learning
        """
        self.llm = llm_client
        self.model = model
        self.min_batch_size = min_batch_size
        self.min_correction_batch = min_correction_batch
        self.verbose = verbose
        self.prune_after_learn = prune_after_learn

    def should_trigger_learning(
        self, buffer: "ExampleBuffer", current_rules: Optional[List["Rule"]]
    ) -> CoordinationDecision:
        """Agentic decision based on buffer content"""
        stats = buffer.get_stats()
        new_examples_count = stats["new_examples"]
        corrections_count = stats["new_corrections"]

        # 1. Fast path: Don't bother LLM if not enough data
        # Unless we have corrections (high value) or it's the very first learn
        if current_rules is None or not current_rules:
            if new_examples_count < self.min_batch_size:
                return CoordinationDecision(
                    should_learn=False,
                    strategy="balanced",
                    reasoning=f"Waiting for initial batch (have {new_examples_count}/{self.min_batch_size})",
                )
            # If we have any data at all, allow an initial learn without LLM gating
            return CoordinationDecision(
                should_learn=True,
                strategy="balanced",
                reasoning=f"Initial learn with {new_examples_count} examples",
            )
        else:
            if (
                new_examples_count < self.min_batch_size
                and corrections_count < self.min_correction_batch
            ):
                return CoordinationDecision(
                    should_learn=False,
                    strategy="balanced",
                    reasoning=f"Batch too small (examples: {new_examples_count}/{self.min_batch_size}, corrections: {corrections_count}/{self.min_correction_batch})",
                )

        # 2. Agentic path: Ask LLM
        try:
            decision = self._ask_llm(buffer, current_rules)
            if self.verbose and decision.should_learn:
                print(f"\n🤖 Agentic decision: {decision.reasoning}")
                print(
                    f"   Strategy: {decision.strategy}, max iterations: {decision.max_iterations}"
                )
            return decision
        except Exception as e:
            print(f"Error in agentic coordinator: {e}")
            # Fallback to simple heuristic
            return CoordinationDecision(
                should_learn=True,
                strategy="balanced",
                reasoning="Fallback due to agent error",
            )

    def analyze_buffer(self, buffer: "ExampleBuffer") -> Dict[str, Any]:
        """Analyze buffer stats"""
        return buffer.get_stats()

    def on_learning_complete(
        self,
        old_rules: Optional[List["Rule"]],
        new_rules: List["Rule"],
        metrics,
    ):
        """Log learning results. metrics is an EvalResult or None."""
        if self.verbose:
            if metrics and hasattr(metrics, "exact_match"):
                print(
                    f"✓ Learning complete. Exact match: {metrics.exact_match:.1%}, F1: {metrics.micro_f1:.1%}"
                )
            else:
                print("✓ Learning complete.")

    def guide_refinement(
        self, eval_result: Any, iteration: int, max_iterations: int
    ) -> tuple:
        """LLM-powered refinement guidance based on per-class metrics."""
        import json

        if not hasattr(eval_result, "per_class") or not eval_result.per_class:
            return "", True

        # Build per-class metrics table
        class_lines = []
        for cm in sorted(eval_result.per_class, key=lambda c: c.f1):
            class_lines.append(
                f"  {cm.label}: F1={cm.f1:.0%} P={cm.precision:.0%} R={cm.recall:.0%} "
                f"(TP={cm.tp} FP={cm.fp} FN={cm.fn})"
            )

        prompt = f"""You are the Refinement Coordinator for a rule learning system.
After each refinement iteration, you analyze per-class performance and guide the next patch.

ITERATION: {iteration + 1}/{max_iterations}
OVERALL: accuracy={eval_result.exact_match:.1%}, micro_F1={eval_result.micro_f1:.1%}, macro_F1={eval_result.macro_f1:.1%}

PER-CLASS PERFORMANCE (sorted worst to best):
{chr(10).join(class_lines)}

Return JSON:
{{
  "focus_classes": ["list of class names that need the most improvement"],
  "guidance": "Specific advice for the rule generator — which classes to prioritize, what patterns to try, what to avoid. 2-3 sentences max.",
  "should_continue": boolean (false if performance is good enough or unlikely to improve further)
}}
"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)

            guidance = result.get("guidance", "")
            should_continue = result.get("should_continue", True)
            focus = result.get("focus_classes", [])

            if self.verbose and guidance:
                print(f"🤖 Coordinator: {guidance}")
                if focus:
                    print(f"   Focus: {', '.join(focus)}")

            return guidance, should_continue

        except Exception as e:
            if self.verbose:
                print(f"⚠ Coordinator error: {e}")
            return "", True

    def audit_rules(self, rules: List["Rule"], rule_metrics: List[Any]) -> AuditResult:
        """LLM-powered rule audit: merge redundant rules, remove pure noise."""
        import json

        if not self.prune_after_learn or len(rules) <= 1:
            return AuditResult()

        # Build a compact summary of each rule + its metrics
        rule_entries = []
        metrics_by_id = {m.rule_id: m for m in rule_metrics}

        for rule in rules:
            m = metrics_by_id.get(rule.id)
            entry = {
                "id": rule.id,
                "name": rule.name,
                "format": rule.format.value,
                "pattern": rule.content[:300],
                "priority": rule.priority,
                "output_key": rule.output_key,
            }
            if rule.output_template:
                entry["output_template"] = rule.output_template
            if m:
                entry["metrics"] = {
                    "matches": m.matches,
                    "precision": round(m.precision, 2),
                    "recall": round(m.recall, 2),
                    "f1": round(m.f1, 2),
                    "true_positives": m.true_positives,
                    "false_positives": m.false_positives,
                }
            rule_entries.append(entry)

        prompt = f"""You are a Rule Auditor for a rule-based extraction/classification system.

Analyze these {len(rules)} rules and their per-rule metrics.

RULES:
{json.dumps(rule_entries, indent=2)}

Your job is to CONSOLIDATE the ruleset. Prefer MERGING over removing.

ACTIONS:
1. MERGE: Two+ regex rules with similar patterns targeting the same output/label.
   Combine their patterns into one rule (e.g. merge `(?:bad|awful)` and `(?:terrible|worst)` into `(?:bad|awful|terrible|worst)`).
   Only merge rules of the same format and same output_template/output_key.
2. REMOVE: Only for rules that are pure noise — precision=0 AND matches>0 (every match is wrong).

IMPORTANT — do NOT remove:
- Rules with low F1/recall — they may catch rare but important cases
- Rules with 0 matches — the training set may be small, they could help on unseen data
- The only rule for a class/label — even if it looks weak

Return JSON:
{{
  "analysis": "Brief summary (1-2 sentences)",
  "actions": [
    {{"action": "merge", "rule_ids": ["id1", "id2"], "merged_pattern": "new regex pattern", "merged_name": "Combined rule name", "reason": "why"}},
    {{"action": "remove", "rule_ids": ["id"], "reason": "why"}}
  ]
}}

Return {{"analysis": "All rules are useful", "actions": []}} if no changes needed.
"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)

            actions = []
            for a in result.get("actions", []):
                action_type = a.get("action", "")
                rule_ids = a.get("rule_ids", [])
                if not rule_ids:
                    continue

                actions.append(
                    AuditAction(
                        action=action_type,
                        rule_ids=rule_ids,
                        reason=a.get("reason", ""),
                        merged_pattern=a.get("merged_pattern"),
                        merged_name=a.get("merged_name"),
                    )
                )

            audit = AuditResult(
                actions=actions,
                analysis=result.get("analysis", ""),
            )

            if self.verbose:
                if not actions:
                    print(f"🔍 Audit: {audit.analysis}")
                else:
                    print(f"\n🔍 Rule audit: {audit.analysis}")
                    for a in actions:
                        if a.action == "merge":
                            print(
                                f"   Merge {a.rule_ids} → {a.merged_name}: {a.reason}"
                            )
                        elif a.action == "remove":
                            print(f"   Remove {a.rule_ids[0]}: {a.reason}")

            return audit

        except Exception as e:
            if self.verbose:
                print(f"⚠ Audit error: {e}")
            return AuditResult()

    def _ask_llm(
        self, buffer: "ExampleBuffer", current_rules: Optional[List["Rule"]]
    ) -> CoordinationDecision:
        """Construct prompt and get decision from LLM"""
        import json

        # Get sample of new data
        new_data = buffer.get_new_examples()
        # Limit to 10 samples for prompt context
        samples = new_data[:10]

        prompt = f"""You are the Coordinator for a rule learning system.
Decide if we should trigger a retraining loop NOW based on new data.

STATUS:
- New examples: {len(new_data)}
- New corrections (high priority): {len([e for e in new_data if e.is_correction])}
- Current rules: {len(current_rules) if current_rules else 0}

NEW DATA SAMPLES (up to 10):
"""
        for ex in samples:
            type_str = "CORRECTION" if ex.is_correction else "EXAMPLE"
            prompt += f"- [{type_str}] Input: {json.dumps(ex.input)} -> Output: {json.dumps(ex.output)}\n"

        prompt += """
DECISION CRITERIA:
1. TRIGGER if we have corrections (users fixing mistakes).
2. TRIGGER if we have a significant batch of new examples (5+).
3. WAIT if data looks sparse or redundant.

STRATEGIES:
- 'balanced': Standard mix (default)
- 'corrections_first': If we have corrections
- 'diversity': If we have many similar examples
- 'uncertain': If examples look ambiguous

Return JSON:
{
  "should_learn": boolean,
  "strategy": "balanced" | "corrections_first" | "diversity" | "uncertain",
  "max_iterations": integer (1-3, use 3 for hard changes, 1 for simple),
  "reasoning": "Short explanation"
}
"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        return CoordinationDecision(
            should_learn=result.get("should_learn", False),
            strategy=result.get("strategy", "balanced"),
            reasoning=result.get("reasoning", ""),
            max_iterations=result.get("max_iterations", 3),
            metadata={"source": "llm_agent"},
        )
