# Proof run — voice agents for Indian real estate (2026-04-19, 70-lead expansion)

End-to-end execution of Phases 3 → 5 of the `lead-generator` skill on a live ICP: **voice agents for Indian real estate sales teams**.

Every LinkedIn URL below was verified via WebSearch (`"First Last" Company site:linkedin.com/in`) during this run. Candidates were dropped where SERP could not conclusively confirm role (e.g. Abhishek Lodha, Jitu Virwani, Neel Raheja, Rohit Kapoor, Kunal Walia, Getamber Anand, KT Jithendran, Shrikant Joshi, Sudhanshu Gupta, Gautam Thacker, Yash Miglani, Venkat K Narayana).

## Artifacts

| File | Phase | Rows |
|---|---|---|
| `icp.yaml` | 2 (input) | — |
| `raw.jsonl` | 3 — in-motion signals captured | 70 |
| `enriched.jsonl` | 4 — public-only enrichment | 70 |
| `scored.jsonl` | 5 — deterministic scoring | 70 |
| `hot-list.csv` | 5 — dedup + top-30 cap | 30 |

## Scoring formula

`score = 0.5 × signal_strength + 0.3 × icp_fit + 0.2 × accessibility`

Time-decay on signal freshness — days 0-7 → ×1.0, 8-30 → ×0.75, 31-60 → ×0.4, 61+ → ×0.1.
Tiers — ≥80 A, 60-79 B, 40-59 C, <40 D.
Amplify-family bonus (+10 each, cap +20) for `failed-attempt` and `tool-churn` signals.

## Tier distribution (today = 2026-04-19, N=70)

| Tier | Score | Count | Notes |
|---|---|---|---|
| A | 100 | 11 | Multiple fresh signals (≤30 d) + full accessibility |
| A | 97 | 2 | Title-mismatch -3 (SVP / CEO vs persona list) |
| A | 95 | 1 | Amplify bonus + minor fit penalty |
| A | 90 | 4 | Fresh dual signals with mild title/size mismatch |
| A | 80 | 35 | Single fresh signal + full accessibility |
| B | 77 | 10 | Single fresh signal + title or size mismatch |
| B | 70 | 7 | Single signal aged 25-35 days |

**Totals — A = 53, B = 17, C = 0, D = 0.** Top-30 hot-list cap returns only A-tier leads.

## Top 11 — A-tier, score 100

| # | Company | Decision-maker | LinkedIn (verified) | Signal family | Observed |
|---|---|---|---|---|---|
| 1 | Awfis | Amit Ramani (Founder / Chairman / MD) | https://in.linkedin.com/in/amit-ramani1504 | funding-change + hiring | 2026-03-28 |
| 2 | IndiQube | Rishi Das (Co-founder / CEO) | https://in.linkedin.com/in/rishi-das-cofounder-indiqube | funding-change + hiring | 2026-04-02 |
| 3 | WeWork India | Karan Virwani (MD / CEO) | https://in.linkedin.com/in/karanvirwani | funding-change + hiring | 2026-03-30 |
| 4 | Square Yards | Tanuj Shori (Founder / CEO) | https://www.linkedin.com/in/tanuj-shori-420b402/ | funding-change + hiring | 2026-04-01 |
| 5 | Settl | Bharath Bhaskar (Co-founder) | https://www.linkedin.com/in/bharathbhaskar/ | funding-change + hiring | 2026-04-03 |
| 6 | Signature Global | Pradeep Aggarwal (Founder / Chairman) | https://www.linkedin.com/in/pradeep-aggarwal/ | funding-change + hiring | 2026-04-04 |
| 7 | Smartworks | Neetish Sarda (Founder / MD) | https://in.linkedin.com/in/neetish-sarda-99183a10b | funding-change + hiring | 2026-03-27 |
| 8 | Innov8 | Dr. Ritesh Malik (Founder / CEO) | https://in.linkedin.com/in/drriteshmalik | funding-change + hiring | 2026-04-11 |
| 9 | Mahindra Lifespaces | Amit Kumar Sinha (MD / CEO) | https://in.linkedin.com/in/amitkumarsinha5 | funding-change + hiring | 2026-04-02 |
| 10 | TARC Ltd | Amar Sarin (CEO / MD) | https://in.linkedin.com/in/amar-sarin-40ab5236a | hiring + funding-change | 2026-04-08 |
| 11 | Info Edge (99acres) | Hitesh Oberoi (MD / CEO) | https://in.linkedin.com/in/hitesh-oberoi-131a9a32 | tool-churn + funding-change (+10 bonus) | 2026-04-12 |

Full ranked list is in `hot-list.csv`.

## Verification method

For each lead the run executed:

1. WebSearch query `"First Last" Company site:linkedin.com/in`
2. Took the top SERP match whose title/snippet explicitly referenced the target company and role
3. Extracted the `linkedin.com/in/<slug>` URL
4. Rejected the row if: (a) no direct profile URL, (b) role couldn't be confirmed, or (c) SERP returned a different person with the same name

Hard rule enforced throughout — **never fabricate**. If verification fails, exclude the lead.

## Reproduce

```
python3 .claude/skills/lead-generator/scripts/search_signals.py  hot-list/icp.yaml                # phase 3 queries
#   ... run WebSearch, capture raw hits to raw.jsonl ...
python3 .claude/skills/lead-generator/scripts/score_lead.py      --icp hot-list/icp.yaml \
                                                                  --today 2026-04-19 \
                                                                  hot-list/enriched.jsonl  > hot-list/scored.jsonl
python3 .claude/skills/lead-generator/scripts/update_hot_list.py hot-list/scored.jsonl      > hot-list/hot-list.csv
```
