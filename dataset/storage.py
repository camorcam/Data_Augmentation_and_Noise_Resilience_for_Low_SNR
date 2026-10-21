"""Persist independent, balanced partitions and audit exact burst overlap."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .signal_generator import generate_synthetic_dataset, MODULATION_CLASSES, SNR_GRID

GENERATOR_VERSION = "rf-v2-gaussian-gfsk"


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def audit_partitions(parts):
    seen = {}
    counts = {}
    for name, part in parts.items():
        if part["x"].shape[1:] != (2, 128):
            raise ValueError(f"Invalid I/Q shape: {name}")
        if not np.isfinite(part["x"]).all():
            raise ValueError(f"Nonfinite data: {name}")
        if not np.allclose(
            np.mean(np.sum(part["x"] ** 2, axis=1), axis=1), 1, atol=1e-5
        ):
            raise ValueError(f"Invalid RMS: {name}")
        for x in part["x"]:
            h = array_hash(x)
            if h in seen:
                raise ValueError(
                    f"Duplicate burst within/across partitions: {seen[h]} and {name}"
                )
            seen[h] = name
        counts[name] = len(part["x"])
    return {"counts": counts, "duplicate_bursts": 0, "unique_bursts": len(seen)}


def load_partitions(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["generator_version"] != GENERATOR_VERSION:
        raise ValueError("Unsupported generator version. Prepare a fresh dataset.")
    parts = {}
    for name in ("train", "val", "test"):
        path = directory / f"{name}.npz"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["files"][name]["sha256"]
        ):
            raise ValueError(f"Dataset checksum mismatch: {name}")
        with np.load(path, allow_pickle=False) as z:
            parts[name] = {k: z[k] for k in z.files}
    audit_partitions(parts)
    return parts, manifest


def prepare(directory, counts=(105, 24, 50), seed=12345):
    directory = Path(directory)
    if any(v <= 0 for v in counts):
        raise ValueError("Partition counts must be positive")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "Dataset directory must be empty. Reuse it or choose a new path."
        )
    directory.mkdir(parents=True, exist_ok=True)
    children = np.random.SeedSequence(seed).spawn(3)
    parts = {}
    for (name, count), child in zip(zip(("train", "val", "test"), counts), children):
        stream_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        data = generate_synthetic_dataset(
            samples_per_class_per_snr=count, seed=stream_seed
        )
        parts[name] = {k: data[k] for k in ("x", "y", "snr")}
        parts[name]["sample_id"] = np.array(
            [f"{name}:{i}" for i in range(len(data["x"]))]
        )
    audit = audit_partitions(parts)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "classes": MODULATION_CLASSES,
        "snr_grid": SNR_GRID,
        "counts_per_class_per_snr": dict(zip(("train", "val", "test"), counts)),
        "partition_streams": [c.spawn_key for c in children],
        "audit": audit,
        "files": {},
    }
    for name, part in parts.items():
        path = directory / f"{name}.npz"
        np.savez_compressed(path, **part)
        manifest["files"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/frozen_v2")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--train-per-cell", type=int, default=105)
    p.add_argument("--val-per-cell", type=int, default=24)
    p.add_argument("--test-per-cell", type=int, default=50)
    a = p.parse_args()
    print(
        json.dumps(
            prepare(a.out, (a.train_per_cell, a.val_per_cell, a.test_per_cell), a.seed),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
