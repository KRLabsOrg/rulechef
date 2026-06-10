import json
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

from benchmarks.data import BenchmarkRun
from rulechef.core import Rule, RuleFormat
from rulechef.training_logger import TrainingDataLogger


def save_checkpoint(path: Path, data: dict):
    tmp = path.with_suffix(".cp_tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text())


def load_rules_from_json(path: str | Path) -> list:
    saved = json.loads(Path(path).read_text())
    return [Rule.from_dict(r) for r in saved["rules"]]


def save_results(output_path: Path, run: BenchmarkRun):
    args = run.args
    results = {
        "config": {
            "shots": args.shots,
            "model": args.model,
            "format": args.format,
            "max_rules": args.max_rules,
            "max_samples": args.max_samples,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "train_size": run.train_size,
            "eval_size": run.eval_size,
            "test_size": run.test_size,
            "train_annotations": run.train_annotations,
            "eval_annotations": run.eval_annotations,
            "test_annotations": run.test_annotations,
            "use_grex": not args.no_grex,
            "agentic": args.agentic,
            "enable_critic": args.enable_critic,
            "enable_prune": args.enable_prune,
            "critic_interval": args.critic_interval,
            "audit_interval": args.audit_interval,
            "windows": args.windows,
            "sampling_strategy": args.sampling_strategy,
            "train_ratio": args.train_ratio,
            "pool_size": args.pool_size,
            "batch_size": args.batch_size,
            "refine_per_batch": args.refine_per_batch,
            "synthesis_strategy": args.synthesis_strategy,
            "selected_classes": run.selected_classes,
            "train_name": args.train_name,
            "test_name": args.test_name,
            "transfer_name": args.transfer_name,
            "transfer_test_name": args.transfer_test_name,
            "max_counter_examples": args.max_counter_examples,
            "refine_every": args.refine_every,
        },
        "metadata": run.metadata,
        "results": {
            "accuracy": run.eval_results.exact_match,
            "micro_precision": run.eval_results.micro_precision,
            "micro_recall": run.eval_results.micro_recall,
            "micro_f1": run.eval_results.micro_f1,
            "macro_f1": run.eval_results.macro_f1,
            "num_rules": len(run.rules),
            "learning_time_s": round(run.t_learn, 1),
            "eval_time_s": round(run.t_eval, 3),
            "per_query_ms": round(run.t_eval / len(run.test_data) * 1000, 2),
        },
        "per_class": [
            {
                "label": cm.label,
                "precision": cm.precision,
                "recall": cm.recall,
                "f1": cm.f1,
                "tp": cm.tp,
                "fp": cm.fp,
                "fn": cm.fn,
            }
            for cm in (run.eval_results.per_class or [])
        ],
        "iteration_metrics": run.iteration_metrics,
        "batch_test_metrics": run.batch_test_metrics,
        "rules": [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "format": r.format.value,
                "content": r.content,
                "priority": r.priority,
                "output_template": r.output_template,
                "output_key": r.output_key,
                "created_at": r.created_at.isoformat(),
            }
            for r in run.rules
        ],
        "best_rules": {
            "batch": run.metadata.get("best_batch_idx"),
            "micro_f1": run.metadata.get("best_batch_f1"),
            "rules": run.metadata.get("best_rules_serialized", []),
        },
    }
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")
    return results


def setup_output_paths(args, selected_classes):
    storage_dir = tempfile.mkdtemp(prefix="rulechef_bench_")

    model_name = args.model.replace("/", "_")
    date_str = datetime.now().strftime("%Y-%m-%d")
    selected_classes_str = "_".join(c.replace(" ", "") for c in selected_classes).strip()
    num_classes = len(selected_classes)

    if args.rules_json:
        output_dir = Path(args.rules_json).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = Path(f"reports/{args.dataset_name}/{model_name}") / selected_classes_str
        base_name = date_str

        output_dir = base_dir / base_name
        if output_dir.exists():
            version = 1
            while (base_dir / f"{base_name}_v{version}").exists():
                version += 1
            output_dir = base_dir / f"{base_name}_v{version}"

        output_dir.mkdir(parents=True, exist_ok=True)

    out_name = Path(args.output)
    if args.rules_json or getattr(args, "feedback", None):
        out_name = out_name.with_stem(out_name.stem + "_refined")

    log_path = (output_dir / out_name).with_suffix(".training.jsonl")
    logger = TrainingDataLogger(
        str(log_path),
        run_metadata={
            "benchmark": args.dataset_name,
            "model": args.model,
            "format": args.format,
            "num_classes": num_classes,
        },
    )
    print(f"Training log: {log_path}")

    config_name = (
        "config_refined.yaml"
        if (args.rules_json or getattr(args, "feedback", None))
        else "config.yaml"
    )
    config_path = output_dir / config_name
    config_dict = {k.replace("_", "-"): v for k, v in vars(args).items() if k not in ("config",)}
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    print(f"Config saved to {config_path}")

    return storage_dir, output_dir, out_name, logger
