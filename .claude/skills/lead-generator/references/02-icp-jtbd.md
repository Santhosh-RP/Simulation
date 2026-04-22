# Phase 1 — ICP + JTBD

An ICP built from demographics alone is a dead ICP. A real ICP is a person + a Job To Be Done + a demand-timing trigger.

## The three layers of an ICP

1. **Who (firmographics)** — industry, size, geography, stack, revenue.
2. **Job (JTBD)** — what outcome they are trying to hire something to produce, and why *now*.
3. **Trigger (demand timing)** — the event that pushed the pain over the threshold.

A prospect without a trigger is a name, not a lead.

## JTBD interview script (5 questions for the user)

Ask the user who sells the product:

1. Think of your 3 best customers. What were they trying to achieve when they bought from you?
2. What were they doing *before* your product that wasn't working?
3. What event in their week/month made them finally say "I need to fix this now"?
4. What were they afraid would happen if they didn't fix it?
5. How did they describe the solution in their own words (not your marketing)?

The answers produce a JTBD statement:

> When **[situation]**, I want to **[motivation]** so I can **[desired outcome]**.

Example:

> When our best SDR quit mid-quarter and pipeline dried up, I want to replace their output without hiring, so I can hit quota without a 3-month ramp.

## Pain threshold checklist

The prospect is over the threshold only if **at least two** are true:

- [ ] They have recently tried and failed to solve it themselves
- [ ] They are hiring for the problem
- [ ] They are publicly complaining about it
- [ ] They just got funded / grew / lost a customer — i.e. the cost of inaction just jumped
- [ ] A peer just solved it visibly (social proof pressure)
- [ ] Their tool/vendor for this just broke, sunset, or got acquired
- [ ] A regulatory/market change made inaction risky

If zero are true → they're a future prospect, not a now prospect. Do not pursue.

## ICP output — what to write into `hot-list/icp.yaml`

Use the template at `templates/icp.yaml`. Required fields:

- `product` (what you sell)
- `pain` (one sentence)
- `jtbd` (the situation/motivation/outcome statement above)
- `firmographics` (industry, size, geo, stack)
- `persona` (title, seniority, daily workflow, KPI they're measured on)
- `triggers` (list of 5–8 observable events that mean "pain just crossed the threshold")
- `disqualifiers` (reasons to skip — e.g. "company < 10 employees", "already using competitor X")
- `one_line_offer` (`I help [X] get [Y] without [Z]`)

## Anti-patterns to flag and reject

- "My ICP is SaaS founders" → too broad, no JTBD, no trigger.
- "Anyone who needs marketing" → zero firmographic boundary.
- "People on LinkedIn with 'VP' in their title" → title-based, not pain-based.
- Offer line with more than one clause on each side of "get ... without" → cognitive load.
