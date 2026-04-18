# Phase 2 — In-Motion Signals

The single highest-leverage move in the whole pipeline. A B-tier message to an in-motion prospect beats an A-tier message to a static one.

## Signal taxonomy

Group signals into five families. Each has a distinct web-search query shape.

### A. Hiring signals
They're hiring for the pain. They've admitted it and are spending money on it.
- Open roles on their careers page
- Job posts on LinkedIn, Indeed, Wellfound, YC Jobs, We Work Remotely
- "Hiring X manager" announcements on social

Query shape: `"{role title}" "{industry or tool}" site:linkedin.com/jobs OR site:jobs.lever.co OR site:boards.greenhouse.io`

### B. Failed-attempt signals
They tried, it didn't work, they're vocal about it.
- Public posts: "we tried X and it didn't work"
- Reddit/Hacker News/forum complaints
- Negative G2/Capterra/Trustpilot reviews on a competitor
- Churn announcements ("moving off X")

Query shape: `"tried {competitor/approach}" "didn't work" OR "waste of money" OR "switching from"`

### C. Growth / funding / change signals
The cost of inaction just jumped.
- Funding announcements (Crunchbase, TechCrunch, press releases)
- Leadership hires (new VP of X)
- Product launches
- Expansion to new market
- Acquisitions

Query shape: `"{industry}" "raised" OR "series A" OR "series B" last 30 days`

### D. Public frustration signals
Decision-makers venting in public.
- LinkedIn posts with pain language
- Twitter/X threads
- Podcast quotes
- Conference talks where they describe the problem

Query shape: `"{pain phrase}" site:linkedin.com/posts OR site:twitter.com`

### E. Tool / vendor churn signals
Their current solution just broke.
- Vendor sunsetting a product
- Acquisition of their current tool
- Price hike announcements
- Public outage post-mortems

Query shape: `"{competitor} sunset" OR "{competitor} acquired" OR "{competitor} price increase"`

## Query construction rules

1. **One signal family per query.** Don't combine. Results get messy.
2. **Recency window.** Always constrain to the last 30–60 days. Stale signals are not signals.
3. **Geography / industry filter.** Always attach one firmographic constraint from the ICP.
4. **Exclude noise terms.** Add `-jobs` to failure searches, `-site:yourcompany.com` to everything.
5. **Expand with synonyms.** Each pain has 3–5 phrasings. Generate queries for all of them.

## Output format for raw hits (`hot-list/raw.jsonl`)

Each line is one JSON object:

```json
{"url": "...", "title": "...", "company": "...", "person": "...", "role": "...", "signal_family": "hiring", "signal_summary": "posted SDR Manager role 2026-04-10", "observed_date": "2026-04-10", "source_snippet": "...", "query": "..."}
```

`person` and `role` may be null at raw stage — enrichment fills them.

## How many signals to pull per ICP

- 40–60 raw hits is the sweet spot.
- If you pull <20, the ICP or query is too narrow. Widen synonyms.
- If you pull >100, the ICP is too broad. Narrow firmographics or recency.

## Script to run

```bash
python3 .claude/skills/lead-generator/scripts/search_signals.py hot-list/icp.yaml > hot-list/queries.txt
```

This expands the ICP into ~20–30 concrete WebSearch queries. Then, for each query, you (Claude) call `WebSearch`, parse the top 5–10 hits, and append to `raw.jsonl`.

## When a signal is NOT a signal

- The post is > 90 days old
- It's a recycled press release
- It's a career-page evergreen listing (always-open roles)
- The "frustration" is a generic rant with no company context
- It's the same exec talking about the same pain as a year ago (they're not in motion, they're stuck)
