"""Final evaluation of a validation-frozen pair on the persisted test set."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from dataset.storage import load_partitions
from models.network import ModulationClassifier1DCNN
from models.transforms import CONFIGS
from train import loader, predict
from metrics import report


def stats(values):
    values = [float(x) for x in values if x is not None]
    return {
        "mean": float(np.mean(values)) if values else None,
        "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def evaluate(a):
    if a.augmented_config == "baseline":
        raise ValueError("Choose an augmented configuration")
    if len(set(a.seeds)) != len(a.seeds):
        raise ValueError("Duplicate seeds")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    parts, manifest = load_partitions(a.data)
    fingerprint = hashlib.sha256(
        (Path(a.data) / "manifest.json").read_bytes()
    ).hexdigest()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = []
    preds = []
    for seed in a.seeds:
        pair = {}
        cfgs = {}
        for name in ("baseline", a.augmented_config):
            cp = Path(a.runs) / f"seed_{seed}" / name / "best.pth"
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            cfg = ck["config"]
            cfgs[name] = cfg
            if cfg["manifest_sha256"] != fingerprint:
                raise ValueError(f"Dataset mismatch: {cp}")
            if cfg["seed"] != seed or cfg["config_name"] != name:
                raise ValueError(f"Run identity mismatch: {cp}")
            model = ModulationClassifier1DCNN()
            model.load_state_dict(ck["model_state_dict"])
            model.to(device)
            loss, y, p, s = predict(model, loader(parts["test"], a.batch_size), device)
            pair[name] = report(y, p, s, manifest["snr_grid"])
            pair[name]["loss"] = loss
            preds.append((f"{seed}_{name}", p))
        base, aug = cfgs["baseline"], cfgs[a.augmented_config]
        for k in (
            "initial_state_sha256",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "selection_metric",
        ):
            if base[k] != aug[k]:
                raise ValueError(f"Unmatched pair: {k}")
        b = pair["baseline"]["snr80"]["snr_db"]
        v = pair[a.augmented_config]["snr80"]["snr_db"]
        records.append(
            {
                "seed": seed,
                "models": pair,
                "sensitivity_gain_db": None if b is None or v is None else b - v,
            }
        )
    summary = {}
    for name in ("baseline", a.augmented_config):
        summary[name] = {
            "global_accuracy": stats(
                [r["models"][name]["global"]["accuracy"] for r in records]
            ),
            "global_macro_f1": stats(
                [r["models"][name]["global"]["macro_f1"] for r in records]
            ),
            "low_snr_mean_accuracy": stats(
                [r["models"][name]["low_snr_mean_accuracy"] for r in records]
            ),
        }
        for metric in ("accuracy", "macro_f1"):
            summary[name][metric + "_per_snr"] = [
                stats([r["models"][name]["per_snr"][str(s)][metric] for r in records])
                for s in manifest["snr_grid"]
            ]
    paired = {}
    for metric in ("accuracy", "macro_f1"):
        paired[metric + "_difference"] = stats(
            [
                r["models"][a.augmented_config]["global"][metric]
                - r["models"]["baseline"]["global"][metric]
                for r in records
            ]
        )
    paired["low_snr_accuracy_difference"] = stats(
        [
            r["models"][a.augmented_config]["low_snr_mean_accuracy"]
            - r["models"]["baseline"]["low_snr_mean_accuracy"]
            for r in records
        ]
    )
    paired["sensitivity_gain_db"] = stats([r["sensitivity_gain_db"] for r in records])
    result = {
        "manifest_sha256": fingerprint,
        "classes": manifest["classes"],
        "snr_grid": manifest["snr_grid"],
        "augmented_config": a.augmented_config,
        "seeds": a.seeds,
        "records": records,
        "summary": summary,
        "paired": paired,
        "sensitivity_convention": "first upward crossing, linear interpolation, no extrapolation or smoothing",
        "uncertainty": "Sample SD across training seeds on one shared test set, not a confidence interval or dataset uncertainty.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_results.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        out / "predictions.npz",
        y=parts["test"]["y"],
        snr=parts["test"]["snr"],
        sample_id=parts["test"]["sample_id"],
        **dict(preds),
    )
    print(json.dumps({"summary": summary, "paired": paired}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/frozen_v2")
    p.add_argument("--runs", default="runs/v2")
    p.add_argument("--out", default="results/v2")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument(
        "--augmented-config", choices=list(CONFIGS), default="augmented_full"
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=["cpu", "cuda"])
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
