#!/usr/bin/env python3
"""Bayesian weekly review of outreach results.

Reads tracking.csv (date,lead_id,signal_family,angle,cta,stage_reached,replied,...)
Groups by signal_family, angle, and cta. For each bucket with n >= 1, computes
a Beta(1,1) posterior over reply probability and issues an action
(amplify / hold / drop / need-more-data).

Usage: python3 weekly_review.py hot-list/tracking.csv
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict


MIN_N = 10
AMPLIFY_DELTA = 0.03
DROP_DELTA = 0.03
DROP_MIN_N = 20


def beta_mean(replies: int, attempts: int) -> float:
    # Beta(1+replies, 1+attempts-replies) mean
    return (replies + 1) / (attempts + 2)


def beta_quantile(a: float, b: float, q: float, iters: int = 80) -> float:
    """Bisection on the regularized incomplete beta (approximated via cdf from
    a cumulative sum of pdf on a fine grid). Good enough for reporting CIs
    without pulling scipy. Returns a value in [0,1]."""
    # Use a coarse grid; fine enough for reporting.
    N = 1000
    pdf = [0.0] * (N + 1)
    # logB = log Beta(a,b)
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    for i in range(N + 1):
        x = i / N
        if x == 0 or x == 1:
            pdf[i] = 0.0
            continue
        logp = (a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - logB
        pdf[i] = math.exp(logp)
    cdf = 0.0
    step = 1.0 / N
    cum = 0.0
    for i in range(N + 1):
        cum += pdf[i] * step
        if cum >= q:
            return i / N
    return 1.0


def ci90(replies: int, attempts: int) -> tuple[float, float]:
    a = 1 + replies
    b = 1 + attempts - replies
    return beta_quantile(a, b, 0.05), beta_quantile(a, b, 0.95)


def aggregate(rows: list[dict]) -> dict[str, dict[str, tuple[int, int]]]:
    out: dict[str, dict[str, tuple[int, int]]] = {
        "signal_family": defaultdict(lambda: (0, 0)),
        "angle": defaultdict(lambda: (0, 0)),
        "cta": defaultdict(lambda: (0, 0)),
    }
    for r in rows:
        try:
            replied = 1 if int(r.get("replied") or 0) == 1 else 0
        except ValueError:
            replied = 0
        for attr in out.keys():
            bucket = (r.get(attr) or "").strip()
            if not bucket:
                continue
            n, k = out[attr][bucket]
            out[attr][bucket] = (n + 1, k + replied)
    return out


def decide(mean: float, overall: float, n: int) -> str:
    if n < MIN_N:
        return "need-more-data"
    if mean > overall + AMPLIFY_DELTA:
        return "amplify"
    if mean < overall - DROP_DELTA and n >= DROP_MIN_N:
        return "drop"
    return "hold"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: weekly_review.py path/to/tracking.csv", file=sys.stderr)
        return 2

    with open(sys.argv[1], encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        print("no rows in tracking.csv", file=sys.stderr)
        return 1

    total_n = len(rows)
    total_k = sum(1 for r in rows if (r.get("replied") or "0").strip() == "1")
    overall_mean = beta_mean(total_k, total_n)

    print(f"# Weekly review — {total_n} attempts, {total_k} replies, "
          f"overall posterior mean {overall_mean:.3f}\n")
    print(f"{'attribute':<16}{'bucket':<30}{'n':>4}{'k':>4}{'mean':>8}"
          f"{'ci_low':>8}{'ci_high':>8}  action")

    buckets = aggregate(rows)
    for attr, groups in buckets.items():
        for bucket, (n, k) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
            m = beta_mean(k, n)
            lo, hi = ci90(k, n)
            action = decide(m, overall_mean, n)
            print(f"{attr:<16}{bucket[:28]:<30}{n:>4}{k:>4}"
                  f"{m:>8.3f}{lo:>8.3f}{hi:>8.3f}  {action}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
