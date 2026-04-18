# Phase 3 — Enrichment

Enrichment turns a raw signal hit into a complete lead record. Public sources only. Never invent a field.

## Fields to enrich per lead

Required:
- `company` — legal name
- `domain` — primary website
- `signal_family` + `signal_summary` + `observed_date` (already captured from Phase 2)
- `size_range` — one of: `1-10, 11-50, 51-200, 201-1000, 1000+`
- `industry`
- `country`

High-value:
- `decision_maker_name`
- `decision_maker_title`
- `decision_maker_public_profile_url`
- `recent_public_activity` — last 3 posts/talks/interviews, dated
- `stack_hints` — tools visible on job posts, website, or case studies
- `competitor_in_use` — if publicly mentioned
- `funding_stage` — if publicly known
- `last_news` — most recent newsworthy item with date

Optional:
- `linkedin_company_url`
- `public_email_or_form` — only if published on the site

## Source allowlist

Allowed:
- Company website (about, team, pricing, blog, changelog, careers)
- Company LinkedIn page (public)
- Person LinkedIn profile (public posts only)
- Press releases, Crunchbase public pages, news articles
- Public job boards
- Public review sites (G2, Capterra, Trustpilot) — for competitor churn only
- Podcast show notes, conference talk pages

Disallowed:
- Gated/logged-in content
- Leaked databases, pastebin dumps
- Purchased contact lists
- Guessed email patterns presented as confirmed
- Screenshots or private Slack/Discord content

## Guessed data rule

If a field is not on a public page, it is `null`. Do not guess. Do not pattern-generate emails (e.g. `first.last@domain`) and present them as found. You may note `email_pattern_hypothesis` as a separate, clearly-labelled field.

## Enrichment flow

For each raw hit:

1. Read `url` → identify `company` and `domain`.
2. `WebFetch` the homepage → extract industry, size hints, pricing tier.
3. `WebSearch` `"{company} linkedin"` → get company profile → extract size_range, industry.
4. `WebSearch` `"{company} {likely decision-maker title}"` → identify the human.
5. `WebSearch` `"{decision-maker name} site:linkedin.com/posts"` last 30 days → 3 recent activities.
6. Append enriched record to `hot-list/enriched.jsonl`.

Budget: ~3–5 WebSearch/WebFetch calls per lead, max. If a lead needs more, it's not worth it at this stage — move on.

## Output format (`hot-list/enriched.jsonl`)

```json
{
  "lead_id": "slug-of-company-date",
  "company": "Acme Inc",
  "domain": "acme.com",
  "signal_family": "hiring",
  "signal_summary": "posted 'Head of Revenue Operations' 2026-04-10",
  "observed_date": "2026-04-10",
  "size_range": "51-200",
  "industry": "B2B SaaS / Dev tools",
  "country": "US",
  "decision_maker_name": "Jane Doe",
  "decision_maker_title": "VP of Sales",
  "decision_maker_public_profile_url": "https://www.linkedin.com/in/janedoe",
  "recent_public_activity": [
    {"date": "2026-04-08", "type": "linkedin_post", "summary": "complained about pipeline visibility"}
  ],
  "stack_hints": ["Salesforce", "Outreach"],
  "competitor_in_use": null,
  "funding_stage": "Series B",
  "last_news": {"date": "2026-03-01", "summary": "raised $40M Series B"},
  "public_email_or_form": "contact@acme.com",
  "source_urls": ["https://..."]
}
```

## Dedup rule

If the same `company` appears twice with different signals, merge into one record and concatenate signals into a list. Do not create two leads for one company.
