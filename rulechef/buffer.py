"""Example buffering for observed LLM and human interactions"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ObservedExample:
    """Example from LLM observation or human input"""

    input: Dict[str, Any]
    output: Dict[str, Any]
    source: str  # "llm" | "human"
    is_correction: bool = False
    is_negative: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExampleBuffer:
    """Thread-safe buffer for incoming examples from multiple sources"""

    def __init__(self):
        self.examples: List[ObservedExample] = []
        self.last_learn_index = 0
        self.lock = threading.Lock()

    def add_llm_observation(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Dict = None,
    ):
        """Add example observed from LLM interaction"""
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
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        is_negative: bool = False,
    ):
        """Add human-labeled example"""
        with self.lock:
            self.examples.append(
                ObservedExample(
                    input=input_data,
                    output=output_data,
                    source="human",
                    is_negative=is_negative,
                    is_correction=False,
                )
            )

    def add_human_correction(
        self,
        input_data: Dict[str, Any],
        expected_output: Dict[str, Any],
        actual_output: Dict[str, Any],
    ):
        """Add human correction of model output"""
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
                )
            )

    def get_all_examples(self) -> List[ObservedExample]:
        """Get all examples"""
        with self.lock:
            return self.examples.copy()

    def get_new_examples(self) -> List[ObservedExample]:
        """Get examples added since last learn"""
        with self.lock:
            return self.examples[self.last_learn_index :].copy()

    def get_new_corrections(self) -> List[ObservedExample]:
        """Get corrections added since last learn"""
        return [e for e in self.get_new_examples() if e.is_correction]

    def mark_learned(self):
        """Mark current state as learned from"""
        with self.lock:
            self.last_learn_index = len(self.examples)

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics"""
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
