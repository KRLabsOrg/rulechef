"""Example buffering for observed LLM and human interactions"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObservedExample:
    """Example from LLM observation or human input"""

    input: dict[str, Any]
    output: dict[str, Any]
    source: str  # "llm" | "human"
    is_correction: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExampleBuffer:
    """Thread-safe buffer for incoming examples from multiple sources."""

    def __init__(self):
        """Initialize an empty example buffer.

        Examples accumulate until mark_learned() is called, which advances
        the cursor so get_new_examples() only returns unprocessed items.
        """
        self.examples: list[ObservedExample] = []
        self.last_learn_index = 0
        self.lock = threading.Lock()

    def add_llm_observation(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        metadata: dict = None,
    ):
        """Add example observed from LLM interaction.

        Args:
            input_data: Input dict that was sent to the LLM.
            output_data: Output dict parsed from the LLM response.
            metadata: Optional metadata dict (e.g. model name, seed).
        """
        with self.lock:
            self.examples.append(
                ObservedExample(
                    input=input_data,
                    output=output_data,
                    source="llm",
                    is_correction=False,
                    metadata=metadata or {},
                )
            )

    def add_human_example(
        self, input_data: dict[str, Any], output_data: dict[str, Any]
    ):
        """Add human-labeled example.

        Args:
            input_data: Input dict matching the task's input_schema.
            output_data: Expected output dict matching the task's output_schema.
        """
        with self.lock:
            self.examples.append(
                ObservedExample(
                    input=input_data,
                    output=output_data,
                    source="human",
                    is_correction=False,
                )
            )

    def add_human_correction(
        self,
        input_data: dict[str, Any],
        expected_output: dict[str, Any],
        actual_output: dict[str, Any],
        feedback: str | None = None,
    ):
        """Add human correction of model output.

        Args:
            input_data: Input dict that was processed.
            expected_output: The correct output the model should have produced.
            actual_output: The incorrect output that was produced.
            feedback: Optional free-text explanation of what went wrong.
        """
        metadata = {}
        if feedback is not None:
            metadata["feedback"] = feedback

        with self.lock:
            self.examples.append(
                ObservedExample(
                    input=input_data,
                    output={
                        "expected": expected_output,
                        "actual": actual_output,
                    },
                    source="human",
                    is_correction=True,
                    metadata=metadata,
                )
            )

    def get_all_examples(self) -> list[ObservedExample]:
        """Get all examples"""
        with self.lock:
            return self.examples.copy()

    def get_new_examples(self) -> list[ObservedExample]:
        """Get examples added since last learn"""
        with self.lock:
            return self.examples[self.last_learn_index :].copy()

    def get_new_corrections(self) -> list[ObservedExample]:
        """Get corrections added since last learn"""
        return [e for e in self.get_new_examples() if e.is_correction]

    def mark_learned(self):
        """Mark current state as learned from"""
        with self.lock:
            self.last_learn_index = len(self.examples)

    def get_stats(self) -> dict[str, int]:
        """Get buffer statistics.

        Returns:
            Dict with keys: 'total_examples' (int), 'new_examples' (int),
            'new_corrections' (int), 'llm_observations' (int),
            'human_examples' (int).
        """
        with self.lock:
            new_examples = self.examples[self.last_learn_index :]
            return {
                "total_examples": len(self.examples),
                "new_examples": len(new_examples),
                "new_corrections": len([e for e in new_examples if e.is_correction]),
                "llm_observations": len([e for e in new_examples if e.source == "llm"]),
                "human_examples": len(
                    [
                        e
                        for e in new_examples
                        if e.source == "human" and not e.is_correction
                    ]
                ),
            }

    def clear(self):
        """Clear all examples (use with caution)"""
        with self.lock:
            self.examples.clear()
            self.last_learn_index = 0
