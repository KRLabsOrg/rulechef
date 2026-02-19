"""Training data logger for distillation.

Captures all LLM calls made during rule synthesis, patching, and coordination
as structured (prompt, response) pairs suitable for fine-tuning a smaller model.

Usage:
    from rulechef import RuleChef
    from rulechef.training_logger import TrainingDataLogger

    logger = TrainingDataLogger("training_data/run_001.jsonl")
    chef = RuleChef(task, client, training_logger=logger)

    # ... add examples, learn rules ...
    # All LLM calls are logged to the JSONL file.

    print(logger.stats)

Output format (one JSON object per line):
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "call_type": "rule_synthesis|rule_patch|guide_refinement|audit_rules|trigger_decision|...",
        "metadata": {
            "task_name": "Banking Intent Classification",
            "task_type": "classification",
            "dataset_size": 25,
            "num_classes": 5,
            "iteration": 3,
            "eval_before": {"micro_f1": 0.55, "macro_f1": 0.48, "exact_match": 0.42},
            "num_rules_in_response": 8,
            "response_valid": true,
            ...
        },
        "timestamp": "2026-02-19T14:30:00"
    }

The metadata is not part of the training input — it's for filtering and quality control.
During fine-tuning, use only the "messages" field. Filter on metadata to select
high-quality examples (e.g. only keep runs where final F1 > 60%).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingDataLogger:
    """Logs LLM calls as training data for model distillation.

    Each logged call is a (messages, response) pair with metadata.
    Written as JSONL, one entry per LLM call.

    Attributes:
        path: Path to the output JSONL file.
        stats: Dict of call counts by type.
        run_id: Identifier for this logging session.
    """

    def __init__(
        self,
        path: str,
        run_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the logger.

        Args:
            path: Path to the output JSONL file. Created if it doesn't exist.
                Appends if it already exists (safe for multiple runs).
            run_metadata: Optional dict of metadata to attach to every entry
                in this session (e.g. dataset name, model, configuration).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_metadata = run_metadata or {}
        self.stats: Dict[str, int] = {}
        self._count = 0

    def log(
        self,
        call_type: str,
        messages: List[Dict[str, str]],
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single LLM call.

        Args:
            call_type: Type of call. One of:
                - "rule_synthesis": Initial rule generation from examples
                - "rule_synthesis_per_class": Per-class rule generation
                - "rule_patch": Patch rules for failures
                - "guide_refinement": Coordinator refinement guidance
                - "audit_rules": Rule pruning/merging audit
                - "trigger_decision": Should-learn decision
                - "synthetic_generation": Synthetic example generation
            messages: The messages sent to the LLM (system + user).
            response: The raw response text from the LLM.
            metadata: Additional context for this specific call. Useful fields:
                - task_name, task_type: What task this is for
                - dataset_size, num_classes: Data stats
                - iteration, max_iterations: Where in the refinement loop
                - eval_before: Metrics before this call's output is applied
                - num_rules_in_response: How many rules were parsed
                - response_valid: Whether the response parsed successfully
                - target_class: For per-class synthesis
                - num_failures: For patch calls
                - guidance: For coordinator calls
        """
        # Build the messages array in ChatML format
        # Include assistant response as the final message
        full_messages = list(messages) + [{"role": "assistant", "content": response}]

        entry = {
            "messages": full_messages,
            "call_type": call_type,
            "metadata": {**self.run_metadata, **(metadata or {})},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self.stats[call_type] = self.stats.get(call_type, 0) + 1
        self._count += 1

    @property
    def count(self) -> int:
        """Total number of entries logged."""
        return self._count

    def __repr__(self) -> str:
        return f"TrainingDataLogger(path={self.path!r}, entries={self._count})"
