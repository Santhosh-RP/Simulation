#!/usr/bin/env python3
"""Turn scored.jsonl into a daily hot-list.csv.

- Dedupes by company domain (keeps highest score).
- Sorts by score desc.
- Caps at 30.

Usage: python3 update_hot_list.py hot-list/scored.jsonl > hot-list/hot-list.csv
"""

from __future__ import annotations

import csv
import json
import sys


CAP = 30
FIELDS = [
    "lead_id",
    "company",
    "domain",
    "decision_maker_name",
    "decision_maker_title",
    "signal_family",
    "signal_summary",
    "observed_date",
    "score",
    "tier",
    "top_reasons",
    "public_channel",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: update_hot_list.py path/to/scored.jsonl", file=sys.stderr)
        return 2

    by_domain: dict[str, dict] = {}
    with open(sys.argv[1], encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("domain") or rec.get("company") or rec.get("lead_id") or "").lower()
            if not key:
                continue
            prior = by_domain.get(key)
            if prior is None or rec.get("score", 0) > prior.get("score", 0):
                by_domain[key] = rec

    ranked = sorted(by_domain.values(), key=lambda r: r.get("score", 0), reverse=True)[:CAP]

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    for rec in ranked:
        fam = rec.get("signal_family")
        if not fam:
            fams = rec.get("signal_families") or []
            fam = fams[0] if fams else ""
        row = {
            "lead_id": rec.get("lead_id", ""),
            "company": rec.get("company", ""),
            "domain": rec.get("domain", ""),
            "decision_maker_name": rec.get("decision_maker_name", "") or "",
            "decision_maker_title": rec.get("decision_maker_title", "") or "",
            "signal_family": fam,
            "signal_summary": rec.get("signal_summary", "") or "",
            "observed_date": rec.get("observed_date", "") or "",
            "score": rec.get("score", 0),
            "tier": rec.get("tier", ""),
            "top_reasons": " | ".join((rec.get("reasons") or [])[:3]),
            "public_channel": rec.get("decision_maker_public_profile_url")
                or rec.get("public_email_or_form")
                or "",
        }
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
