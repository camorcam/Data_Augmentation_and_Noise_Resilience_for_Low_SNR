"""Plot final curves, all seed histories and -4/+4/+14 dB matrices."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/v2")
    p.add_argument("--runs", default="runs/v2")
    a = p.parse_args()
    root = Path(a.results)
    r = json.loads((root / "benchmark_results.json").read_text())
    out = root / "plots"
    out.mkdir(exist_ok=True)
    names = ["baseline", r["augmented_config"]]
    colors = ["#0F172A", "#0284C7"]
    grid = r["snr_grid"]
    for metric in ("accuracy", "macro_f1"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, color in zip(names, colors):
            values = r["summary"][name][metric + "_per_snr"]
            mean = np.array([v["mean"] for v in values])
            sd = np.array([v["std_across_seeds"] or 0 for v in values])
            ax.plot(grid, mean, "o-", label=name, color=color)
            if len(r["seeds"]) > 1:
                ax.fill_between(
                    grid,
                    np.maximum(mean - sd, 0),
                    np.minimum(mean + sd, 1),
                    alpha=0.14,
                    color=color,
                )
        ax.axvspan(-10, 2, color="#E2E8F0", alpha=0.3)
        ax.set(xlabel="SNR (dB)", ylabel=metric, ylim=(0, 1), xticks=grid)
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_title(
            "Mean ± SD across training seeds"
            if len(r["seeds"]) > 1
            else "Single training seed"
        )
        fig.tight_layout()
        fig.savefig(out / f"{metric}_vs_snr.png", dpi=200)
        plt.close(fig)
    # Show matrices from every seed, rather than silently selecting a favorable run.
    for rec in r["records"]:
        for snr in (-4, 4, 14):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ax, name in zip(axes, names):
                cm = np.array(
                    rec["models"][name]["per_snr"][str(snr)]["confusion_normalized"]
                )
                im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
                for row in range(8):
                    for col in range(8):
                        ax.text(
                            col,
                            row,
                            f"{cm[row,col]*100:.0f}",
                            ha="center",
                            va="center",
                            color="white" if cm[row, col] > 0.6 else "black",
                            fontsize=9,
                        )
                ax.set(
                    xticks=range(8),
                    yticks=range(8),
                    xticklabels=r["classes"],
                    yticklabels=r["classes"],
                    xlabel="Predicted",
                    ylabel="True",
                    title=f"{name}, {snr:+d} dB",
                )
                ax.tick_params(axis="x", labelrotation=45)
            fig.suptitle(f'Seed {rec["seed"]}: row-normalized percentages')
            fig.tight_layout()
            fig.savefig(out / f'cm_seed{rec["seed"]}_{snr:+d}dB.png', dpi=200)
            plt.close(fig)
    for seed in r["seeds"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, color in zip(names, colors):
            h = json.loads(
                (Path(a.runs) / f"seed_{seed}" / name / "history.json").read_text()
            )
            epochs = [v["epoch"] for v in h]
            axes[0].plot(
                epochs,
                [v["train_clean_eval_loss"] for v in h],
                color=color,
                marker="o",
                linestyle="--",
                label=name + " clean train",
            )
            axes[0].plot(
                epochs,
                [v["val_loss"] for v in h],
                color=color,
                marker="o",
                label=name + " val",
            )
            axes[1].plot(
                epochs,
                [v["train_aug_loss"] for v in h],
                color=color,
                marker="o",
                label=name + " training inputs",
            )
        for ax in axes:
            ax.set(xlabel="Epoch", ylabel="Cross-entropy")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        axes[0].set_title("Same clean evaluation protocol")
        axes[1].set_title("Training losses: input distributions differ")
        fig.tight_layout()
        fig.savefig(out / f"loss_seed{seed}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
