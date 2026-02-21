"""Dataset persistence and queries."""

import json
import uuid
from pathlib import Path

from rulechef.core import (
    Correction,
    Dataset,
    Example,
    Feedback,
    Rule,
    RuleFormat,
)


class DatasetStore:
    """Handles saving, loading, and querying datasets on disk."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, dataset: Dataset) -> None:
        """Save dataset to disk."""
        filepath = self.storage_path / f"{dataset.name}.json"
        with open(filepath, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2, default=str)

    def load(self, dataset: Dataset) -> None:
        """Load dataset from disk if it exists."""
        filepath = self.storage_path / f"{dataset.name}.json"

        if not filepath.exists():
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Restore examples
            for ex in data.get("examples", []):
                example = Example(
                    id=ex["id"],
                    input=ex["input"],
                    expected_output=ex["expected_output"],
                    source=ex["source"],
                    confidence=ex.get("confidence", 0.8),
                )
                dataset.examples.append(example)

            # Restore corrections
            for corr in data.get("corrections", []):
                correction = Correction(
                    id=corr["id"],
                    input=corr["input"],
                    model_output=corr["model_output"],
                    expected_output=corr["expected_output"],
                    feedback=corr.get("feedback"),
                )
                dataset.corrections.append(correction)

            # Restore feedback
            dataset.feedback = data.get("feedback", [])

            # Restore structured feedback
            for fb_data in data.get("structured_feedback", []):
                fb = Feedback(
                    id=fb_data["id"],
                    text=fb_data["text"],
                    level=fb_data["level"],
                    target_id=fb_data.get("target_id", ""),
                )
                dataset.structured_feedback.append(fb)

            # Restore rules
            for rule_data in data.get("rules", []):
                rule = Rule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    description=rule_data["description"],
                    format=RuleFormat(rule_data["format"]),
                    content=rule_data["content"],
                    priority=rule_data.get("priority", 5),
                    confidence=rule_data.get("confidence", 0.5),
                    times_applied=rule_data.get("times_applied", 0),
                    successes=rule_data.get("successes", 0),
                    failures=rule_data.get("failures", 0),
                    output_template=rule_data.get("output_template"),
                    output_key=rule_data.get("output_key"),
                )
                dataset.rules.append(rule)

            print(
                f"✓ Loaded dataset: {len(dataset.corrections)} corrections, "
                f"{len(dataset.examples)} examples"
            )

        except Exception as e:
            print(f"Error loading dataset: {e}")

    def generate_id(self) -> str:
        """Generate unique ID."""
        return str(uuid.uuid4())[:8]

    def get_stats(self, dataset: Dataset) -> dict:
        """Get dataset statistics."""
        return {
            "task": dataset.task.name,
            "dataset": dataset.name,
            "corrections": len(dataset.corrections),
            "examples": len(dataset.examples),
            "feedback": len(dataset.feedback),
            "rules": len(dataset.rules),
            "description": dataset.description,
        }

    def get_rules_summary(self, dataset: Dataset) -> list[dict]:
        """Get formatted summary of learned rules."""
        summaries = []
        for rule in sorted(dataset.rules, key=lambda r: r.priority, reverse=True):
            success_rate = (
                rule.successes / rule.times_applied * 100 if rule.times_applied > 0 else 0
            )
            summaries.append(
                {
                    "name": rule.name,
                    "description": rule.description,
                    "format": rule.format.value,
                    "priority": rule.priority,
                    "confidence": f"{rule.confidence:.2f}",
                    "times_applied": rule.times_applied,
                    "success_rate": f"{success_rate:.1f}%" if rule.times_applied > 0 else "N/A",
                }
            )
        return summaries
