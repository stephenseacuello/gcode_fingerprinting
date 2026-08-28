#!/usr/bin/env python3
"""Shrink a decoder's results/metrics.json to just the fields the aggregator
and per-class analyses actually consume. In place; idempotent. Used by the
LOMO drivers' RETURN trap to keep per-cell residual under control.

Keeps: test_metrics, val_metrics, best_epoch, epoch, config.
Drops: per-sample arrays / token-level histories / anything else.

Exits silently on any error (best-effort cleanup; never blocks the trap)."""
from __future__ import annotations
import json
import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    p = sys.argv[1]
    try:
        if not os.path.isfile(p) or os.path.getsize(p) <= 1_000_000:
            return 0
        d = json.loads(open(p).read())
        if "test_metrics" not in d:
            return 0
        small: dict = {"test_metrics": d["test_metrics"]}
        for k in ("val_metrics", "best_epoch", "epoch", "config"):
            if k in d:
                small[k] = d[k]
        open(p, "w").write(json.dumps(small))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
