"""Metrics include every class and document the 80% crossing convention."""

import numpy as np


def confusion(y, p, nclasses=8):
    cm = np.zeros((nclasses, nclasses), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def scores(y, p, nclasses=8):
    cm = confusion(y, p, nclasses)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(nclasses, dtype=float), where=denom > 0)
    norm = np.divide(
        cm,
        cm.sum(1, keepdims=True),
        out=np.zeros_like(cm, dtype=float),
        where=cm.sum(1, keepdims=True) > 0,
    )
    return {
        "accuracy": float(np.mean(y == p)),
        "macro_f1": float(f1.mean()),
        "support": cm.sum(1).tolist(),
        "confusion_counts": cm.tolist(),
        "confusion_normalized": norm.tolist(),
    }


def snr80(grid, acc):
    if acc[0] >= 0.8:
        return {
            "snr_db": None,
            "status": "at_or_below_grid_min",
            "bound_db": float(grid[0]),
        }
    for i in range(1, len(grid)):
        if acc[i - 1] < 0.8 <= acc[i]:
            val = grid[i - 1] + (0.8 - acc[i - 1]) * (grid[i] - grid[i - 1]) / (
                acc[i] - acc[i - 1]
            )
            return {"snr_db": float(val), "status": "first_upward_linear_crossing"}
    return {"snr_db": None, "status": "not_reached"}


def report(y, p, snr, grid):
    per = {str(s): scores(y[snr == s], p[snr == s]) for s in grid}
    acc = [per[str(s)]["accuracy"] for s in grid]
    low = [per[str(s)]["accuracy"] for s in grid if -10 <= s <= 2]
    return {
        "global": scores(y, p),
        "per_snr": per,
        "low_snr_mean_accuracy": float(np.mean(low)),
        "snr80": snr80(grid, acc),
    }
