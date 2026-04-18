# Phase 6 — Bayesian Feedback Loop

Every interaction is data. Treating outbound as a feelings-driven activity is the single biggest reason systems don't improve. Treat it like a Bayesian experiment.

## The core idea

Each attribute of a message (signal family, angle, CTA, segment) has a prior probability of getting a reply. Every outcome updates that prior. Over weeks, the system converges on what actually works *for you*, not what worked in a blog post.

## What to track

Log every outbound attempt in `hot-list/tracking.csv`:

```
date,lead_id,signal_family,angle,cta,stage_reached,replied,reply_sentiment,notes
```

- `stage_reached`: `sent | opened | clicked | replied | booked | closed-won | closed-lost`
- `replied`: `0 | 1`
- `reply_sentiment`: `positive | neutral | objection | unsubscribe`
- `angle`: a short tag for the pain angle used (e.g. `forecast-drift`, `ramp-time`, `tool-churn`)
- `cta`: the exact micro-yes variant used (e.g. `3-line-breakdown`, `loom-90s`, `one-pager`)

## The weekly review (Bayesian update)

Run weekly:

```bash
python3 .claude/skills/lead-generator/scripts/weekly_review.py hot-list/tracking.csv
```

It groups the data by each attribute and, for each bucket with n ≥ 10 attempts, computes a Beta posterior over reply-rate using a Beta(1,1) uniform prior:

- Posterior mean = `(replies + 1) / (attempts + 2)`
- 90% credible interval via Beta quantiles

Output columns:

```
attribute, bucket, attempts, replies, posterior_mean, ci_low, ci_high, action
```

`action` is one of:
- `amplify` — posterior_mean > overall mean + 0.03, CI is tight → do more of this
- `hold` — within noise of overall mean → keep testing
- `drop` — posterior_mean < overall mean − 0.03 with n ≥ 20 → stop this variant
- `need-more-data` — n < 10

## How to act on the review

Each week, use the `action` column to update:

- **ICP** — if one `segment` bucket dominates → tighten ICP to that segment.
- **Signals** — if one `signal_family` dominates → reallocate Phase 2 queries toward it.
- **Angles** — drop losing angles from Phase 5 message variants.
- **CTAs** — keep the top 2 micro-yes variants; retire the rest.

Rule of thumb: change ONE variable per week. Changing 4 at once means you learn nothing.

## Guardrails

- Don't act on buckets with n < 10. That's noise.
- Don't abandon a signal after one bad week — confirm across two reviews.
- Reply rate is the leading indicator. Booked / closed-won are lagging. Don't over-index on lagging until you have months of data.

## Sample output

```
attribute        bucket             n   replies  mean   ci       action
signal_family    hiring             34  5        0.167  .07-.30  amplify
signal_family    funding            22  1        0.083  .02-.24  drop
angle            forecast-drift     18  4        0.250  .09-.47  amplify
angle            ramp-time          16  0        0.056  .01-.24  drop
cta              3-line-breakdown   28  6        0.233  .10-.40  amplify
cta              book-a-call        19  0        0.048  .01-.20  drop
```

The interpretation writes itself: drop funding signals and ramp-time angle, double down on hiring + forecast-drift + 3-line-breakdown.

## The meta-rule

If the system doesn't have a feedback loop, it cannot improve. Even a weekly 10-minute review beats no review. If the user resists tracking, tell them plainly: *without this, everything upstream is guessing.*
