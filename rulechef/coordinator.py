"""Coordination layer for learning decisions - swappable simple/agentic implementations"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rulechef.buffer import ExampleBuffer
    from rulechef.core import Rule


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
        metrics: Dict[str, Any],
    ):
        """Log learning results"""
        if self.verbose:
            accuracy = metrics.get("accuracy", 0)
            total = metrics.get("total", 0)
            correct = metrics.get("correct", 0)

            if old_rules is None:
                print("✓ Initial learning complete:")
            else:
                print("✓ Refinement complete:")

            print(f"  {len(new_rules)} rules")
            if total > 0:
                print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")


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
    ):
        """
        Args:
            llm_client: OpenAI client
            model: Model to use for coordination
            min_batch_size: Minimum new examples before asking LLM
            min_correction_batch: Minimum corrections before asking LLM
            verbose: Print coordination decisions
        """
        self.llm = llm_client
        self.model = model
        self.min_batch_size = min_batch_size
        self.min_correction_batch = min_correction_batch
        self.verbose = verbose

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
        metrics: Dict[str, Any],
    ):
        """Log learning results"""
        if self.verbose:
            accuracy = metrics.get("accuracy", 0) if metrics else 0
            print(f"✓ Learning complete. Accuracy: {accuracy:.1%}")

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
