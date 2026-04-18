#!/usr/bin/env python3
"""Score enriched leads. Reads JSONL of enriched records from stdin or a file,
writes scored JSONL to stdout.

Score = 0.5 * signal_strength + 0.3 * icp_fit + 0.2 * accessibility

Tier:
  >=80 A   60-79 B   40-59 C   <40 D

Usage:
  python3 score_lead.py hot-list/enriched.jsonl > hot-list/scored.jsonl
  python3 score_lead.py --icp hot-list/icp.yaml hot-list/enriched.jsonl > ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path


AMPLIFY_FAMILIES = {"failed-attempt", "tool-churn"}


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def decay(observed: date | None, today: date) -> float:
    if observed is None:
        return 0.5
    age = (today - observed).days
    if age <= 7:
        return 1.0
    if age <= 30:
        return 0.75
    if age <= 60:
        return 0.4
    return 0.1


def signal_strength(record: dict, today: date) -> tuple[int, list[str]]:
    reasons: list[str] = []
    families = record.get("signal_families") or []
    if not families and record.get("signal_family"):
        families = [record["signal_family"]]

    observed = parse_date(record.get("observed_date"))
    age_factor = decay(observed, today)

    if not families:
        return 0, reasons

    base = 0
    if len(families) >= 2 and age_factor >= 0.75:
        base = 100
        reasons.append(f"multiple fresh signals: {', '.join(families)}")
    else:
        # single-signal scoring based on freshness
        if age_factor == 1.0:
            base = 80
        elif age_factor == 0.75:
            base = 60
        elif age_factor == 0.4:
            base = 40
        else:
            base = 20
        if observed:
            reasons.append(f"{families[0]} signal on {observed.isoformat()}")

    bonus = 0
    for f in families:
        if f in AMPLIFY_FAMILIES:
            bonus += 10
    bonus = min(bonus, 20)
    if bonus:
        reasons.append(f"+{bonus} high-leverage family bonus")

    score = min(100, base + bonus)
    return score, reasons


def icp_fit(record: dict, icp: dict | None) -> tuple[int, list[str]]:
    reasons: list[str] = []
    fit = 100
    if not icp:
        return fit, reasons

    firmo = icp.get("firmographics") or {}
    persona = icp.get("persona") or {}

    size_ok = True
    target_sizes = _listify(firmo.get("size_range"))
    if target_sizes and record.get("size_range") and record["size_range"] not in target_sizes:
        fit -= 30
        size_ok = False
        reasons.append(f"size mismatch: {record.get('size_range')} not in {target_sizes}")

    industries = _listify(firmo.get("industries"))
    if industries and record.get("industry"):
        if not any(ind.lower() in record["industry"].lower() for ind in industries):
            fit -= 30
            reasons.append("industry mismatch")

    geos = _listify(firmo.get("geographies"))
    if geos and record.get("country"):
        if record["country"] not in geos:
            fit -= 20
            reasons.append("geo mismatch")

    disq = _listify(icp.get("disqualifiers"))
    rec_text = json.dumps(record).lower()
    for d in disq:
        if d and d.lower() in rec_text:
            fit -= 20
            reasons.append(f"disqualifier matched: {d}")
            break

    persona_titles = _listify(persona.get("titles"))
    if persona_titles and record.get("decision_maker_title"):
        if not any(t.lower() in record["decision_maker_title"].lower() for t in persona_titles):
            fit -= 10
            reasons.append("title not in persona list")

    fit = max(0, fit)
    _ = size_ok
    return fit, reasons


def accessibility(record: dict) -> tuple[int, list[str]]:
    dm = record.get("decision_maker_name")
    channel = record.get("decision_maker_public_profile_url") or record.get("public_email_or_form")
    activity = record.get("recent_public_activity") or []

    if dm and channel and activity:
        return 100, ["decision-maker + public channel + recent activity"]
    if dm and channel:
        return 70, ["decision-maker + one channel"]
    if channel:
        return 40, ["company-level contact only"]
    return 10, ["no decision-maker, no channel"]


def tier_of(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


def _listify(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x]
    return [str(v)]


def score_record(record: dict, icp: dict | None, today: date) -> dict:
    sig, sig_reasons = signal_strength(record, today)
    fit, fit_reasons = icp_fit(record, icp)
    acc, acc_reasons = accessibility(record)
    total = round(0.5 * sig + 0.3 * fit + 0.2 * acc, 1)

    out = dict(record)
    out["signal_strength"] = sig
    out["icp_fit"] = fit
    out["accessibility"] = acc
    out["score"] = total
    out["tier"] = tier_of(total)
    out["reasons"] = (sig_reasons + fit_reasons + acc_reasons)[:6]
    return out


def load_icp(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        from search_signals import parse_minimal_yaml  # type: ignore
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from search_signals import parse_minimal_yaml  # type: ignore
    text = Path(path).read_text(encoding="utf-8")
    return parse_minimal_yaml(text)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default="-")
    p.add_argument("--icp", default=None, help="path to icp.yaml for fit scoring")
    p.add_argument("--today", default=None, help="override today (YYYY-MM-DD) for deterministic tests")
    args = p.parse_args()

    today = datetime.strptime(args.today, "%Y-%m-%d").date() if args.today else date.today()
    icp = load_icp(args.icp)

    stream = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            print(json.dumps(score_record(rec, icp, today), ensure_ascii=False))
    finally:
        if stream is not sys.stdin:
            stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
