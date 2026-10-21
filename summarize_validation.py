"""Inspect validation ablations without opening test predictions."""

import argparse, json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs/ablation")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    a = p.parse_args()
    root = Path(a.runs)
    configs = sorted(
        {
            d.name
            for seed in a.seeds
            for d in (root / f"seed_{seed}").glob("*")
            if d.is_dir()
        }
    )
    result = {}
    for name in configs:
        vals = []
        global_vals = []
        for seed in a.seeds:
            r = json.loads(
                (root / f"seed_{seed}" / name / "validation.json").read_text()
            )
            vals.append(r["low_snr_mean_accuracy"])
            global_vals.append(r["global"]["accuracy"])
        result[name] = {
            "seeds": a.seeds,
            "low_snr_mean": float(np.mean(vals)),
            "low_snr_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "global_mean": float(np.mean(global_vals)),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
