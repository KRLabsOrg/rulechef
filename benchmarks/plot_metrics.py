import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {
    "f1": "#e6a817",
    "precision": "steelblue",
    "recall": "red",
    "phase1": "#1f77b4",  # blue
    "transfer": "#ff7f0e",  # orange
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _plot_ax(ax, metrics, final_results=None, title="Batch Metrics", xlabel="Batch"):
    if not metrics or len(metrics) <= 1:
        ax.text(
            0.5,
            0.5,
            f"No {title.lower()}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return
    steps = [m["batch"] for m in metrics] if xlabel == "Batch" else list(range(len(metrics)))
    ax.plot(
        steps,
        [m["micro_f1"] for m in metrics],
        label="F1",
        color=COLORS["f1"],
        linewidth=1.5,
    )
    ax.plot(
        steps,
        [m["micro_precision"] for m in metrics],
        label="Precision",
        color=COLORS["precision"],
        linewidth=1.5,
    )
    ax.plot(
        steps,
        [m["micro_recall"] for m in metrics],
        label="Recall",
        color=COLORS["recall"],
        linewidth=1.5,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_single(data: dict, label: str, output_path: Path | None = None):
    config = data.get("config", {})
    model = config.get("model", "")
    dataset = label

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{dataset} — {model}", fontsize=11)

    _plot_ax(
        axes[0],
        data.get("batch_test_metrics", []),
        final_results=data.get("results"),
        title="Batch Metrics (dev set)",
        xlabel="Batch",
    )
    _plot_ax(
        axes[1],
        data.get("iteration_metrics", []),
        title="Refinement Iterations",
        xlabel="Iteration",
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_combined(phase1_data: dict, transfer_data: dict, output_path: Path | None = None):
    p1_batch = phase1_data.get("batch_test_metrics", [])
    tr_batch = transfer_data.get("batch_test_metrics", [])
    p1_config = phase1_data.get("config", {})
    tr_config = transfer_data.get("config", {})
    model = p1_config.get("model", tr_config.get("model", ""))
    p1_name = p1_config.get("train_name", "Phase 1")
    tr_name = tr_config.get("transfer_name", "manual_findok")

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Transfer Learning Progression — {model}", fontsize=11)

    offset = p1_batch[-1]["batch"] + 1 if p1_batch else 0
    separator_x = offset - 0.5 if p1_batch and tr_batch else None

    p1_x = [m["batch"] for m in p1_batch]
    tr_x = [m["batch"] + offset for m in tr_batch]
    all_x = p1_x + tr_x

    for key, color, label, ls in [
        ("micro_f1", COLORS["f1"], "F1", "-"),
        ("micro_precision", COLORS["precision"], "Precision", "--"),
        ("micro_recall", COLORS["recall"], "Recall", ":"),
    ]:
        all_y = [m[key] for m in p1_batch] + [m[key] for m in tr_batch]
        ax.plot(all_x, all_y, label=label, color=color, linewidth=1.5, linestyle=ls)

    if p1_x:
        ax.axvspan(
            p1_x[0] - 0.5,
            separator_x if separator_x else p1_x[-1] + 0.5,
            alpha=0.06,
            color=COLORS["phase1"],
        )
        ax.text(
            (p1_x[0] + p1_x[-1]) / 2,
            0.97,
            p1_name,
            ha="center",
            va="top",
            fontsize=8,
            color=COLORS["phase1"],
            transform=ax.get_xaxis_transform(),
        )
    if tr_x:
        ax.axvspan(
            separator_x if separator_x else tr_x[0] - 0.5,
            tr_x[-1] + 0.5,
            alpha=0.06,
            color=COLORS["transfer"],
        )
        ax.text(
            (tr_x[0] + tr_x[-1]) / 2,
            0.97,
            tr_name,
            ha="center",
            va="top",
            fontsize=8,
            color=COLORS["transfer"],
            transform=ax.get_xaxis_transform(),
        )

    if separator_x is not None:
        ax.axvline(x=separator_x, color="purple", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to phase 1 results JSON")
    parser.add_argument("--transfer", default=None, help="Path to transfer results JSON")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path prefix (e.g. 'plots/run1'). "
        "Saves <prefix>_phase1.png, _transfer.png, _combined.png",
    )
    args = parser.parse_args()

    phase1_data = _load(Path(args.results))
    prefix = Path(args.output) if args.output else None
    parent = Path(args.results).parent
    plot_single(
        phase1_data,
        label=f"phase1 ({phase1_data.get('config', {}).get('dataset_name', '')})",
        output_path=Path(f"{parent}/phase1.png"),
    )

    if args.transfer:
        transfer_data = _load(Path(args.transfer))

        plot_single(
            transfer_data,
            label=f"transfer ({transfer_data.get('config', {}).get('dataset_name', '')})",
            output_path=Path(f"{parent}/transfer.png"),
        )

        plot_combined(phase1_data, transfer_data, output_path=Path(f"{parent}/combined.png"))


if __name__ == "__main__":
    main()
