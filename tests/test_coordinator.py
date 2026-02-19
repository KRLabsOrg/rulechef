"""Tests for rulechef.coordinator — SimpleCoordinator decision logic."""


from rulechef.buffer import ExampleBuffer
from rulechef.core import Rule, RuleFormat
from rulechef.coordinator import CoordinationDecision, SimpleCoordinator


# =========================================================================
# CoordinationDecision
# =========================================================================


class TestCoordinationDecision:
    def test_metadata_defaults_to_empty_dict(self):
        d = CoordinationDecision(
            should_learn=True, strategy="balanced", reasoning="test"
        )
        assert d.metadata == {}

    def test_metadata_provided(self):
        d = CoordinationDecision(
            should_learn=False,
            strategy="balanced",
            reasoning="test",
            metadata={"key": "value"},
        )
        assert d.metadata == {"key": "value"}

    def test_max_iterations_default(self):
        d = CoordinationDecision(
            should_learn=True, strategy="balanced", reasoning="test"
        )
        assert d.max_iterations == 3


# =========================================================================
# SimpleCoordinator
# =========================================================================


def _make_dummy_rules(n=1):
    """Create n dummy rules."""
    return [
        Rule(
            id=f"r{i}",
            name=f"rule_{i}",
            description="dummy",
            format=RuleFormat.REGEX,
            content=r"\d+",
        )
        for i in range(n)
    ]


def _fill_buffer(buffer, n_examples=0, n_corrections=0):
    """Add examples and corrections to a buffer."""
    for i in range(n_examples):
        buffer.add_human_example({"text": f"example-{i}"}, {"spans": []})
    for i in range(n_corrections):
        buffer.add_human_correction(
            {"text": f"correction-{i}"}, {"spans": []}, {"spans": []}
        )


class TestSimpleCoordinatorFirstLearn:
    def test_not_ready_below_threshold(self):
        """First learn: not ready when buffer < trigger_threshold."""
        coordinator = SimpleCoordinator(trigger_threshold=10, verbose=False)
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=5)

        decision = coordinator.should_trigger_learning(buffer, current_rules=None)
        assert decision.should_learn is False
        assert "5/10" in decision.reasoning

    def test_ready_at_threshold(self):
        """First learn: ready when buffer >= trigger_threshold."""
        coordinator = SimpleCoordinator(trigger_threshold=10, verbose=False)
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=10)

        decision = coordinator.should_trigger_learning(buffer, current_rules=None)
        assert decision.should_learn is True
        assert decision.strategy == "balanced"
        assert decision.max_iterations == 3

    def test_ready_above_threshold(self):
        coordinator = SimpleCoordinator(trigger_threshold=5, verbose=False)
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=20)

        decision = coordinator.should_trigger_learning(buffer, current_rules=None)
        assert decision.should_learn is True


class TestSimpleCoordinatorSubsequentLearn:
    def test_enough_examples_triggers_diversity(self):
        """Subsequent learn with enough examples uses diversity strategy."""
        coordinator = SimpleCoordinator(
            trigger_threshold=10, correction_threshold=5, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=15)

        decision = coordinator.should_trigger_learning(
            buffer, current_rules=_make_dummy_rules(3)
        )
        assert decision.should_learn is True
        assert decision.strategy == "diversity"
        assert decision.max_iterations == 3

    def test_enough_corrections_triggers_corrections_first(self):
        """Subsequent learn with enough corrections uses corrections_first strategy."""
        coordinator = SimpleCoordinator(
            trigger_threshold=100, correction_threshold=3, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=2, n_corrections=3)

        decision = coordinator.should_trigger_learning(
            buffer, current_rules=_make_dummy_rules(2)
        )
        assert decision.should_learn is True
        assert decision.strategy == "corrections_first"
        assert decision.max_iterations == 2

    def test_not_ready_below_both_thresholds(self):
        """Subsequent learn: not ready when below both thresholds."""
        coordinator = SimpleCoordinator(
            trigger_threshold=50, correction_threshold=10, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=5, n_corrections=2)

        decision = coordinator.should_trigger_learning(
            buffer, current_rules=_make_dummy_rules(1)
        )
        assert decision.should_learn is False
        assert "Not ready" in decision.reasoning

    def test_corrections_priority_over_examples(self):
        """When both thresholds are met, corrections take priority."""
        coordinator = SimpleCoordinator(
            trigger_threshold=5, correction_threshold=3, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=10, n_corrections=5)

        decision = coordinator.should_trigger_learning(
            buffer, current_rules=_make_dummy_rules(1)
        )
        assert decision.should_learn is True
        # corrections_first takes priority because corrections >= correction_threshold
        assert decision.strategy == "corrections_first"


class TestSimpleCoordinatorCustomThresholds:
    def test_custom_thresholds(self):
        coordinator = SimpleCoordinator(
            trigger_threshold=2, correction_threshold=1, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=2)

        # First learn with custom threshold of 2
        decision = coordinator.should_trigger_learning(buffer, current_rules=None)
        assert decision.should_learn is True

    def test_very_high_threshold(self):
        coordinator = SimpleCoordinator(
            trigger_threshold=1000, correction_threshold=500, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=100, n_corrections=50)

        decision = coordinator.should_trigger_learning(
            buffer, current_rules=_make_dummy_rules()
        )
        assert decision.should_learn is False


class TestSimpleCoordinatorAnalyzeBuffer:
    def test_analyze_buffer_stats(self):
        coordinator = SimpleCoordinator(
            trigger_threshold=5, correction_threshold=2, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=3, n_corrections=1)

        analysis = coordinator.analyze_buffer(buffer)
        assert analysis["total_examples"] == 4
        assert analysis["new_examples"] == 4
        assert analysis["new_corrections"] == 1
        assert analysis["ready_for_first_learn"] is False  # 4 < 5
        assert analysis["ready_for_refinement"] is False  # 4 < 5 and 1 < 2

    def test_analyze_buffer_ready(self):
        coordinator = SimpleCoordinator(
            trigger_threshold=3, correction_threshold=2, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=5, n_corrections=3)

        analysis = coordinator.analyze_buffer(buffer)
        assert analysis["ready_for_first_learn"] is True
        assert analysis["ready_for_refinement"] is True


class TestSimpleCoordinatorGuideRefinement:
    def test_default_returns_empty_and_true(self):
        """Default guide_refinement returns ('', True)."""
        coordinator = SimpleCoordinator(verbose=False)
        guidance, should_continue = coordinator.guide_refinement(
            eval_result=None, iteration=0, max_iterations=3
        )
        assert guidance == ""
        assert should_continue is True


class TestSimpleCoordinatorMetadata:
    def test_decision_includes_metadata(self):
        coordinator = SimpleCoordinator(
            trigger_threshold=5, correction_threshold=2, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=10)

        decision = coordinator.should_trigger_learning(buffer, current_rules=None)
        assert "buffer_stats" in decision.metadata
        assert decision.metadata["trigger_threshold"] == 5
        assert decision.metadata["correction_threshold"] == 2

    def test_mark_learned_affects_subsequent_decisions(self):
        """After mark_learned, only new examples count toward threshold."""
        coordinator = SimpleCoordinator(
            trigger_threshold=5, correction_threshold=2, verbose=False
        )
        buffer = ExampleBuffer()
        _fill_buffer(buffer, n_examples=10)
        buffer.mark_learned()

        # After mark_learned, new_examples = 0
        decision = coordinator.should_trigger_learning(buffer, current_rules=None)
        assert decision.should_learn is False

        # Add more examples
        _fill_buffer(buffer, n_examples=5)
        decision = coordinator.should_trigger_learning(buffer, current_rules=None)
        assert decision.should_learn is True
