"""Tests for rulechef.buffer — example buffering logic."""

from rulechef.buffer import ExampleBuffer, ObservedExample


class TestExampleBuffer:
    def setup_method(self):
        self.buffer = ExampleBuffer()

    # -----------------------------------------------------------------
    # add_human_example + get_all_examples
    # -----------------------------------------------------------------

    def test_add_human_example_and_retrieve(self):
        self.buffer.add_human_example({"text": "hello"}, {"spans": [{"text": "hello"}]})
        all_ex = self.buffer.get_all_examples()
        assert len(all_ex) == 1
        assert all_ex[0].source == "human"
        assert all_ex[0].is_correction is False
        assert all_ex[0].input == {"text": "hello"}
        assert all_ex[0].output == {"spans": [{"text": "hello"}]}

    # -----------------------------------------------------------------
    # add_llm_observation
    # -----------------------------------------------------------------

    def test_add_llm_observation(self):
        self.buffer.add_llm_observation(
            {"text": "test"}, {"spans": []}, metadata={"model": "gpt-4"}
        )
        all_ex = self.buffer.get_all_examples()
        assert len(all_ex) == 1
        assert all_ex[0].source == "llm"
        assert all_ex[0].is_correction is False
        assert all_ex[0].metadata == {"model": "gpt-4"}

    def test_add_llm_observation_default_metadata(self):
        self.buffer.add_llm_observation({"text": "x"}, {"spans": []})
        assert self.buffer.get_all_examples()[0].metadata == {}

    # -----------------------------------------------------------------
    # add_human_correction
    # -----------------------------------------------------------------

    def test_add_human_correction(self):
        self.buffer.add_human_correction(
            input_data={"text": "Alice"},
            expected_output={"entities": [{"text": "Alice", "type": "PERSON"}]},
            actual_output={"entities": []},
            feedback="Missed entity",
        )
        all_ex = self.buffer.get_all_examples()
        assert len(all_ex) == 1
        ex = all_ex[0]
        assert ex.is_correction is True
        assert ex.source == "human"
        assert ex.output["expected"] == {
            "entities": [{"text": "Alice", "type": "PERSON"}]
        }
        assert ex.output["actual"] == {"entities": []}
        assert ex.metadata["feedback"] == "Missed entity"

    def test_add_human_correction_no_feedback(self):
        self.buffer.add_human_correction(
            input_data={"text": "x"},
            expected_output={"label": "A"},
            actual_output={"label": "B"},
        )
        ex = self.buffer.get_all_examples()[0]
        assert ex.is_correction is True
        assert "feedback" not in ex.metadata

    # -----------------------------------------------------------------
    # get_new_examples / get_all_examples / mark_learned
    # -----------------------------------------------------------------

    def test_get_new_examples_vs_all_after_mark_learned(self):
        self.buffer.add_human_example({"text": "a"}, {"spans": []})
        self.buffer.add_human_example({"text": "b"}, {"spans": []})
        self.buffer.mark_learned()

        self.buffer.add_human_example({"text": "c"}, {"spans": []})

        assert len(self.buffer.get_all_examples()) == 3
        new = self.buffer.get_new_examples()
        assert len(new) == 1
        assert new[0].input == {"text": "c"}

    def test_get_new_corrections_filters(self):
        self.buffer.add_human_example({"text": "a"}, {"spans": []})
        self.buffer.add_human_correction(
            {"text": "b"}, {"spans": [{"text": "b"}]}, {"spans": []}
        )
        self.buffer.add_human_example({"text": "c"}, {"spans": []})

        new_corrections = self.buffer.get_new_corrections()
        assert len(new_corrections) == 1
        assert new_corrections[0].is_correction is True

    def test_get_new_corrections_respects_mark_learned(self):
        self.buffer.add_human_correction({"text": "a"}, {"spans": []}, {"spans": []})
        self.buffer.mark_learned()

        # This correction is after mark_learned
        self.buffer.add_human_correction({"text": "b"}, {"spans": []}, {"spans": []})

        new_corrections = self.buffer.get_new_corrections()
        assert len(new_corrections) == 1
        assert new_corrections[0].input == {"text": "b"}

    # -----------------------------------------------------------------
    # mark_learned
    # -----------------------------------------------------------------

    def test_mark_learned_resets_window(self):
        for i in range(5):
            self.buffer.add_human_example({"text": f"ex-{i}"}, {"spans": []})

        assert len(self.buffer.get_new_examples()) == 5
        self.buffer.mark_learned()
        assert len(self.buffer.get_new_examples()) == 0
        assert len(self.buffer.get_all_examples()) == 5

    def test_mark_learned_twice(self):
        self.buffer.add_human_example({"text": "a"}, {})
        self.buffer.mark_learned()
        self.buffer.add_human_example({"text": "b"}, {})
        self.buffer.add_human_example({"text": "c"}, {})
        self.buffer.mark_learned()
        assert len(self.buffer.get_new_examples()) == 0
        assert len(self.buffer.get_all_examples()) == 3

    # -----------------------------------------------------------------
    # get_stats
    # -----------------------------------------------------------------

    def test_get_stats(self):
        self.buffer.add_human_example({"text": "a"}, {})
        self.buffer.add_llm_observation({"text": "b"}, {})
        self.buffer.add_human_correction({"text": "c"}, {}, {})
        self.buffer.add_llm_observation({"text": "d"}, {})

        stats = self.buffer.get_stats()
        assert stats["total_examples"] == 4
        assert stats["new_examples"] == 4
        assert stats["new_corrections"] == 1
        assert stats["llm_observations"] == 2
        assert stats["human_examples"] == 1

    def test_get_stats_after_mark_learned(self):
        self.buffer.add_human_example({"text": "a"}, {})
        self.buffer.add_human_example({"text": "b"}, {})
        self.buffer.mark_learned()
        self.buffer.add_human_example({"text": "c"}, {})

        stats = self.buffer.get_stats()
        assert stats["total_examples"] == 3
        assert stats["new_examples"] == 1
        assert stats["new_corrections"] == 0

    # -----------------------------------------------------------------
    # clear
    # -----------------------------------------------------------------

    def test_clear_empties_everything(self):
        self.buffer.add_human_example({"text": "a"}, {})
        self.buffer.add_human_example({"text": "b"}, {})
        self.buffer.mark_learned()
        self.buffer.add_human_example({"text": "c"}, {})

        self.buffer.clear()

        assert len(self.buffer.get_all_examples()) == 0
        assert len(self.buffer.get_new_examples()) == 0
        stats = self.buffer.get_stats()
        assert stats["total_examples"] == 0
        assert stats["new_examples"] == 0

    # -----------------------------------------------------------------
    # Thread safety: basic functional test
    # -----------------------------------------------------------------

    def test_returns_copies_not_references(self):
        """get_all_examples and get_new_examples should return copies."""
        self.buffer.add_human_example({"text": "a"}, {})
        all1 = self.buffer.get_all_examples()
        all1.append(ObservedExample(input={}, output={}, source="test"))
        assert len(self.buffer.get_all_examples()) == 1  # Original unchanged
