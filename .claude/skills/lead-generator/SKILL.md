---
name: lead-generator
description: Signal-based lead generation engine. Use when the user wants to generate leads, build a prospect list, find customers, design outreach, research an ICP, enrich prospects, score leads, or diagnose why outbound is not getting replies. Runs a 6-phase pipeline (diagnose, ICP+JTBD, in-motion signals, enrichment, scoring, messaging) powered by WebSearch. Includes Bayesian weekly-review for continuous improvement.
when_to_use: Trigger on "generate leads", "find prospects", "build a lead list", "cold outreach", "no replies to my outbound", "who should I target", "build an ICP", "score my leads", "enrich leads", "hot list", "signal-based prospecting", "why is my outbound failing".
allowed-tools: Bash WebSearch WebFetch Read Write Edit Glob Grep
argument-hint: [describe what you sell and to whom]
---

# Lead Generator — Signal-Based Outbound Engine

You are running a signal-based lead-generation system. The thesis: **no clients is not a motivation problem — it is a system mismatch**. Your job is to diagnose which part of the system (ICP, timing, signal, offer, message, CTA, feedback loop) is broken and fix it, not to send more messages.

## Operating principles (apply at every step)

1. **Demand timing over demographics.** People buy when pain crosses a threshold, not because they fit an ICP. Prioritize *in-motion* prospects over static ones.
2. **Signal > noise.** Generic outreach is ignored. A sharp, specific observation about the prospect breaks the pattern.
3. **Cognitive load kills.** If the prospect has to think to understand the offer, you lose. One sentence: *I help [specific person] get [specific result] without [specific pain].*
4. **Loss aversion over gain.** Make inaction feel expensive. Quantify what they lose per week/month by not fixing it.
5. **Micro-yeses, not big asks.** First CTA is never a 30-min call. It is "want me to send a 3-line breakdown?"
6. **Every interaction is data.** Track opens, replies, drops, angles that land. Update priors weekly (Bayesian).

## The 6-phase pipeline

Run these phases in order. Do NOT skip phases. If the user is frustrated ("no replies", "no leads"), start at Phase 0.

### Phase 0 — Diagnose (only if user reports outbound failure)
Before doing anything, validate the feeling, then reframe:
> "You're not lazy and you're not imagining it. No replies is painful. But the fix is almost never 'try harder' — it's finding which part of the system is mismatched."

Ask the user to answer 7 short diagnostic questions (see `references/01-reframe-diagnosis.md`). Map each "no" answer to the phase that fixes it. Skip to that phase.

### Phase 1 — ICP + JTBD
Build a written ICP with a **Job To Be Done** and a **demand-timing trigger**. The ICP is not a job title — it is a person whose pain just crossed the threshold.

- Input: user's description of what they sell.
- Output: `hot-list/icp.yaml` populated from `templates/icp.yaml`.
- Read `references/02-icp-jtbd.md` for the JTBD interview script and the "pain threshold" checklist.

### Phase 2 — In-motion signals
Translate the ICP into a list of **buying signals** visible on the public web (hiring, recent funding, public frustration, failed attempts, new launches, tool churn, leadership changes, reviews, RFPs).

- Run: `python3 .claude/skills/lead-generator/scripts/search_signals.py hot-list/icp.yaml > hot-list/queries.txt`
- For each query, call `WebSearch`. Save hits to `hot-list/raw.jsonl` (one JSON per line with: url, title, signal, date, prospect, snippet).
- Stop when you have 40–60 raw candidates. You want *in-motion* prospects, not a database dump.
- Depth: `references/03-in-motion-signals.md` has the full signal taxonomy and query templates.

### Phase 3 — Enrichment
For each raw candidate, enrich with public data: company size, stack, decision-maker, recent activity, competitors, funding.

- Use `WebSearch` and `WebFetch` only on public pages (company site, LinkedIn public profile, news, job boards, review sites).
- Write enriched records to `hot-list/enriched.jsonl`.
- Never scrape gated content or invent data. If a field is unknown, leave it null.
- Playbook: `references/04-enrichment.md`.

### Phase 4 — Scoring
Score every enriched lead 0–100 on three weighted axes:
- **Signal strength (50%)** — how fresh and specific is the in-motion signal
- **ICP fit (30%)** — match to ICP firmographics and JTBD
- **Accessibility (20%)** — is a decision-maker reachable on a public channel

Run: `python3 .claude/skills/lead-generator/scripts/score_lead.py hot-list/enriched.jsonl > hot-list/scored.jsonl`

Then: `python3 .claude/skills/lead-generator/scripts/update_hot_list.py hot-list/scored.jsonl > hot-list/hot-list.csv`

This keeps only the top **20–30** leads for the day. More is worse — cognitive load for you, spray-and-pray for them. Rubric: `references/05-scoring.md`.

### Phase 5 — Messaging
For each of the top 20–30, draft one first-touch message with four required elements:

1. **Sharp observation** — something true about *this* prospect, pulled from the signal that surfaced them. Not "I saw you're in X industry."
2. **One-line offer** — `I help [person] get [result] without [pain]`.
3. **Loss-aversion frame** — quantified cost of inaction ("this likely costs you ~$X/month in Y").
4. **Micro-yes CTA** — never a call. "Want me to send a 3-line breakdown of what's leaking?"

Drafts go to `hot-list/outreach.md`, grouped by lead. Read `references/06-messaging.md` for the full frame, anti-patterns, and examples.

### Phase 6 — Bayesian feedback loop
After the user runs the outreach, they log outcomes in `hot-list/tracking.csv`:
`date, lead_id, signal, angle, cta, stage_reached, replied, notes`

Run weekly: `python3 .claude/skills/lead-generator/scripts/weekly_review.py hot-list/tracking.csv`

It outputs reply-rate by signal, by angle, by segment, and suggests new priors (which signals to amplify, which to drop). Update the ICP and signal list accordingly. Depth: `references/07-bayesian-feedback.md`.

## Output layout (create under repo root or CWD)

```
hot-list/
├── icp.yaml          # Phase 1
├── queries.txt       # Phase 2 (search queries)
├── raw.jsonl         # Phase 2 (search hits)
├── enriched.jsonl    # Phase 3
├── scored.jsonl      # Phase 4
├── hot-list.csv      # Phase 4 (top 20–30 for today)
├── outreach.md       # Phase 5 (personalized drafts)
└── tracking.csv      # Phase 6 (outcomes)
```

## Hard rules

- **Never fabricate** a prospect, a quote, a signal, or a data point. If WebSearch did not return it, it does not exist.
- **Never auto-send.** This skill drafts; the user sends. Mass-send with no human review is spam.
- **Never skip Phase 0** if the user sounds frustrated. Validate first, diagnose second, fix third.
- **Respect public-only.** No gated scraping, no purchased PII, no guessed emails presented as fact.
- **Cap the hot list at 30.** A sharper list of 20 outperforms a list of 200 every time.

## Quick-start

If the user just says "generate leads for {X}":

1. Ask two clarifiers (who it's for, what pain it kills). Stop asking after two.
2. Phase 1 → write `icp.yaml`.
3. Phase 2 → generate queries → run WebSearch → save raw.
4. Phase 3 → enrich.
5. Phase 4 → score → produce hot-list.csv.
6. Phase 5 → draft outreach.
7. Show the user the hot-list + one sample message, ask for approval before drafting the rest.

Progress updates between phases should be one sentence. Do not narrate.
