#!/usr/bin/env python3
"""Regenerate the paper figures from the committed result JSONs.

Produces:
  - observation_curves.pdf : Banking77 coverage/replacement curve (Fig. 2a)
                             + DBpedia trust-thresholded delegation (Fig. 2b)

All inputs are the result JSONs in this directory, so the figures are fully
traceable to the experiments. Output directory defaults to docs/internal/figures
(the paper tree) and is created if missing.

Usage:
    python benchmarks/make_paper_figures.py [--outdir docs/internal/figures]
"""

import argparse
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="docs/internal/figures")
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    obs = json.loads((here / "results" / "results_observation_banking77.json").read_text())
    cps = obs["checkpoint_metrics"]
    n_obs = [c["observations"] for c in cps]
    cov = [c["coverage"] for c in cps]
    repl = [c["replacement_rate"] for c in cps]
    pgold = [c["precision_vs_gold"] for c in cps]

    deleg = json.loads((here / "results" / "results_delegation_dbpedia.json").read_text())
    dc = deleg["curve"]
    d_obs = [c["observed_calls"] for c in dc]
    d_rate = [c["by_trust"]["0.3"]["delegation_rate"] for c in dc]
    d_fid = [c["by_trust"]["0.3"]["fidelity_to_target"] for c in dc]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.0))
    ax1.plot(n_obs, cov, "o-", color="#2166ac", label="rule coverage")
    ax1.plot(n_obs, repl, "s--", color="#67a9cf", label="LLM calls replaced")
    ax1.plot(n_obs, pgold, "^-", color="#b2182b", label="precision vs. gold")
    ax1.set_xlabel("observed LLM calls")
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(n_obs)
    ax1.set_title("(a) Banking77, 5 intents", fontsize=9)
    ax1.legend(frameon=False, fontsize=7, loc="center right")

    ax2.plot(d_obs, d_rate, "o-", color="#2166ac", label="delegation rate")
    ax2.plot(d_obs, d_fid, "^-", color="#b2182b", label="fidelity to LLM")
    ax2.set_xlabel("observed LLM calls")
    ax2.set_ylim(0, 1.02)
    ax2.set_xscale("log")
    ax2.set_xticks(d_obs)
    ax2.set_xticklabels([str(v) for v in d_obs])
    ax2.minorticks_off()
    ax2.set_title("(b) DBpedia, 14 classes (trust $\\geq$ 0.3)", fontsize=9)
    ax2.legend(frameon=False, fontsize=7, loc="center right")

    fig.tight_layout()
    out = outdir / "observation_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
