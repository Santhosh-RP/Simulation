# Phase 4 — Scoring

Scoring is not a fit score. It is a *now-ness* score. A perfect-fit prospect with no trigger scores lower than a decent-fit prospect whose pain is screaming today.

## The 3-axis model

Total score = `0.5 × signal_strength + 0.3 × icp_fit + 0.2 × accessibility`, scaled to 0–100.

### Axis 1 — Signal strength (weight 50%)

| Points | Meaning                                                                     |
|--------|-----------------------------------------------------------------------------|
| 100    | Multiple fresh signals (< 14 days) across families — e.g. hiring + frustration |
| 80     | One fresh signal < 14 days old, high specificity to the pain                |
| 60     | One fresh signal 14–30 days old                                             |
| 40     | One signal 30–60 days old                                                   |
| 20     | Signal is a generic/evergreen indicator (always-open role, stale rant)      |
| 0      | No signal, only ICP fit                                                     |

Signal family bonus (+10 each, cap +20):
- `failed-attempt` (they've already admitted the pain)
- `tool-churn` (displacement window)

### Axis 2 — ICP fit (weight 30%)

Start at 100. Subtract:
- −30 if outside the firmographic size range
- −30 if outside the target industry
- −20 if outside the target geography
- −20 if any `disqualifier` from the ICP matches
- −10 per missing "persona" attribute (max −30)

Floor at 0.

### Axis 3 — Accessibility (weight 20%)

| Points | Condition                                                                     |
|--------|-------------------------------------------------------------------------------|
| 100    | Decision-maker identified + reachable public channel (LinkedIn post, email, form) + recent public activity |
| 70     | Decision-maker identified + one reachable channel                             |
| 40     | Only a company-level contact (info@, contact form)                            |
| 10     | No decision-maker, no public channel                                          |

## Tiering

| Score range | Tier | Action                                          |
|-------------|------|-------------------------------------------------|
| 80–100      | A    | Send today. Personalized 1:1 message.           |
| 60–79       | B    | Queue for this week. Lightly templated.         |
| 40–59       | C    | Nurture (follow on LinkedIn, comment on a post). |
| 0–39        | D    | Drop. Not worth the message.                    |

## The 30-lead cap

After scoring, `update_hot_list.py` keeps only the top 30 by score. If fewer than 10 leads score ≥ 60, the fix is NOT to lower the threshold — the fix is to go back to Phase 2 and widen or change signals.

## Signal decay

A signal loses value over time. Applied automatically:
- Day 0–7: full value
- Day 8–30: × 0.75
- Day 31–60: × 0.4
- Day 61+: × 0.1

Day 0 = `observed_date` in the enriched record.

## Output (`hot-list/scored.jsonl`)

Each enriched record plus:

```json
{
  "signal_strength": 80,
  "icp_fit": 90,
  "accessibility": 70,
  "score": 80.0,
  "tier": "A",
  "reasons": ["fresh hiring signal 2026-04-10", "Series B 6 weeks ago", "VP of Sales identified"]
}
```

`reasons` is a 2–4 item list Claude will reuse when drafting the outreach (Phase 5 needs them for the sharp observation).
