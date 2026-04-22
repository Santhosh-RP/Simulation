# Phase 5 — Messaging

Every first-touch message must pass all four gates. Drop any element and the message becomes noise.

## The four gates

1. **Sharp observation** — a specific, true thing about *this* prospect, pulled from the scoring `reasons` list.
2. **One-line offer** — `I help [specific person] get [specific result] without [specific pain]`.
3. **Loss-aversion frame** — a quantified, concrete picture of what staying the same is costing them.
4. **Micro-yes CTA** — a one-tap "yes" ask that is NOT a meeting request.

## Why each gate exists

### Gate 1 — Sharp observation
Prospects filter aggressively. Generic openers ("Hi, hope you're doing well") get classified as spam in under a second. A sharp observation proves effort, relevance, and that you're not a bot. It's the cheapest possible signal-vs-noise move.

**Good**: "Saw you posted the Head of RevOps role on Apr 10 — usually means pipeline visibility is the bottleneck, not headcount."
**Bad**: "I saw you work at Acme and thought I'd reach out."

### Gate 2 — One-line offer (cognitive load theory)
If the offer takes more than 5 seconds to parse, the brain rejects it. One clause on each side of "get … without …":

- Person: who specifically
- Result: concrete, outcome-shaped
- Pain: the specific thing they DON'T want

**Good**: "I help Series B sales leaders hit quota in 60 days without hiring more SDRs."
**Bad**: "We're a revenue intelligence platform that leverages AI and machine learning to help go-to-market teams optimize their funnel end-to-end."

### Gate 3 — Loss-aversion frame (Kahneman)
People move faster to avoid loss than to gain upside. Make inaction visible and quantified.

Template: `Every week this stays unfixed, you're likely losing ~[X] in [Y].`

**Good**: "At your deal size, every week pipeline is under-reported ≈ 2 missed renewals this quarter."
**Bad**: "We can help you grow revenue." (gain frame, no quantification)

### Gate 4 — Micro-yes CTA (Cialdini — commitment & consistency)
First ask MUST be lower friction than a meeting. Options, in order of preference:

- "Want me to send a 3-line breakdown of where it's likely leaking?"
- "Want me to drop a 90-second Loom?"
- "Should I send one example from a company similar to yours?"
- "Want the one-pager?"

Never first-ask:
- "Can we hop on a 30-min call?"
- "Does Tuesday or Thursday work?"
- "Can I put time on your calendar?"

The first yes is cheap. The call comes after 2–3 micro-yeses.

## Message template

```
Hi [First name],

[Sharp observation — ONE sentence from scoring.reasons].

Usually when I see that, it means [interpretation tied to the pain].

[One-line offer: I help ... get ... without ...].

Rough math: [loss-aversion frame with a number].

[Micro-yes CTA]?

— [Your name]
```

Length cap: 75 words. If it's longer, cut.

## Worked example

Input lead:
- Company: Acme Inc (Series B, 120 employees, B2B SaaS)
- Signal: posted Head of RevOps role 2026-04-10; VP of Sales complained about pipeline visibility on LinkedIn
- Decision-maker: Jane Doe, VP Sales

Output:

> Hi Jane,
>
> Saw the Head of RevOps role you posted last week, 6 weeks after the Series B. Usually that's pipeline visibility breaking, not headcount.
>
> I help Series B sales leaders tighten forecast accuracy in 30 days without hiring ops.
>
> Rough math: at ACV ~$40k and 20% forecast drift, that's ≈ $400k of silent leakage this quarter.
>
> Want me to send a 3-line breakdown of where it usually leaks first?
>
> — [You]

Word count: 73. Passes all four gates.

## Anti-patterns (auto-reject before sending)

- Starts with "Hi, hope you're well"
- Starts with "I know you're busy"
- Mentions your own company name in the first two sentences
- Uses the word "synergy", "leverage", "best-in-class", or "revolutionary"
- Offer contains more than one "and"
- CTA is a meeting or calendar link
- No number anywhere in the message
- Message is > 100 words

## Output file (`hot-list/outreach.md`)

One section per lead, using:

```
## [Company] — [Decision-maker] — score [N], tier [A/B/C]

Signal: [signal_summary]
Angle: [which pain angle you chose]
CTA: [the micro-yes used]

[full message]
```
