"""Train paired seed/config runs; validation only, never inspect test predictions."""

import argparse
import copy
import hashlib
import json
import os
import platform
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.storage import load_partitions
from dataset.pytorch_dataset import WirelessIQDataset
from models.network import ModulationClassifier1DCNN
from models.transforms import RFAugmentationPipeline, CONFIGS
from metrics import report


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def loader(part, batch, shuffle=False, seed=0, transform=None):
    ds = WirelessIQDataset(part["x"], part["y"], part["snr"], transform)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    ys = []
    ps = []
    ss = []
    loss = 0.0
    n = 0
    for x, y, s in batches:
        logits = model(x.to(device))
        yt = y.to(device)
        loss += torch.nn.functional.cross_entropy(logits, yt, reduction="sum").item()
        n += len(y)
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
        ss.append(s.numpy())
    return loss / n, np.concatenate(ys), np.concatenate(ps), np.concatenate(ss)


def run(a):
    if a.epochs < 1 or a.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    parts, manifest = load_partitions(a.data)
    manifest_hash = hashlib.sha256(
        (Path(a.data) / "manifest.json").read_bytes()
    ).hexdigest()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        seed_everything(seed)
        initial = copy.deepcopy(ModulationClassifier1DCNN().state_dict())
        initial_hash = state_hash(initial)
        for config_name in a.configs:
            path = out / f"seed_{seed}" / config_name
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"Refusing to overwrite {path}")
            path.mkdir(parents=True, exist_ok=True)
            cfg = {
                "seed": seed,
                "augmentation": CONFIGS[config_name],
                "max_drop_db": a.max_drop_db,
                "epochs": a.epochs,
                "batch_size": a.batch_size,
                "lr": a.lr,
                "weight_decay": 1e-4,
                "selection_metric": a.selection_metric,
                "initial_state_sha256": initial_hash,
                "manifest_sha256": manifest_hash,
                "dataset_manifest": manifest,
                "device": str(device),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "config_name": config_name,
                "num_workers": 0,
            }
            (path / "config.json").write_text(json.dumps(cfg, indent=2))
            # Isolate dropout RNG from initialization, shuffle and augmentation RNGs.
            model = ModulationClassifier1DCNN()
            model.load_state_dict(initial)
            model.to(device)
            seed_everything(seed + 10000)
            transform = (
                None
                if config_name == "baseline"
                else RFAugmentationPipeline(
                    **CONFIGS[config_name], seed=seed + 20000, max_drop_db=a.max_drop_db
                )
            )
            train = loader(parts["train"], a.batch_size, True, seed + 30000, transform)
            clean = loader(parts["train"], a.batch_size)
            val = loader(parts["val"], a.batch_size)
            opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
            best = -float("inf")
            history = []
            for epoch in range(1, a.epochs + 1):
                model.train()
                loss_sum = 0.0
                correct = 0
                n = 0
                lr = opt.param_groups[0]["lr"]
                for x, y, _ in train:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits, y)
                    loss.backward()
                    opt.step()
                    loss_sum += loss.item() * len(y)
                    correct += (logits.argmax(1) == y).sum().item()
                    n += len(y)
                vl, vy, vp, vs = predict(model, val, device)
                vr = report(vy, vp, vs, manifest["snr_grid"])
                cl, cy, cp, cs = predict(model, clean, device)
                row = {
                    "epoch": epoch,
                    "lr": lr,
                    "train_aug_loss": loss_sum / n,
                    "train_aug_accuracy": correct / n,
                    "train_clean_eval_loss": cl,
                    "train_clean_eval_accuracy": float(np.mean(cy == cp)),
                    "val_loss": vl,
                    "val_accuracy": vr["global"]["accuracy"],
                    "val_low_snr_accuracy": vr["low_snr_mean_accuracy"],
                }
                history.append(row)
                sched.step()
                metric = row[a.selection_metric]
                if metric > best:
                    best = metric
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch,
                            "config": cfg,
                            "selection_value": best,
                            "optimizer_state_dict": opt.state_dict(),
                            "scheduler_state_dict": sched.state_dict(),
                        },
                        path / "best.pth",
                    )
                    (path / "validation.json").write_text(json.dumps(vr, indent=2))
                (path / "history.json").write_text(json.dumps(history, indent=2))
                print(
                    f'{seed=} {config_name} {epoch=}: val={row["val_accuracy"]:.4f} low={row["val_low_snr_accuracy"]:.4f}',
                    flush=True,
                )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/frozen_v2")
    p.add_argument("--out", default="runs/v2")
    p.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=["baseline", "augmented_full"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max-drop-db", type=float, default=6.0)
    p.add_argument(
        "--selection-metric",
        choices=["val_accuracy", "val_low_snr_accuracy"],
        default="val_low_snr_accuracy",
    )
    p.add_argument("--device", choices=["cpu", "cuda"])
    a = p.parse_args()
    if len(a.seeds) != len(set(a.seeds)) or len(a.configs) != len(set(a.configs)):
        raise ValueError("Duplicate seeds/configs")
    run(a)


if __name__ == "__main__":
    main()
