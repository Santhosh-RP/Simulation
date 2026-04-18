#!/usr/bin/env python3
"""Expand an ICP YAML into a list of WebSearch queries, one per line.

Usage: python3 search_signals.py path/to/icp.yaml

Claude then calls WebSearch with each line and captures the top hits into
hot-list/raw.jsonl. Recency filters are baked into the query where possible.
"""

from __future__ import annotations

import sys
from pathlib import Path


def parse_minimal_yaml(text: str) -> dict:
    """Tiny YAML subset parser. Supports nested mappings, scalar values,
    and lists of scalars or lists of dicts. No external deps.

    Stack entries: (indent, container, parent_ref, parent_key). The
    parent_ref + parent_key pair lets us swap a container from dict to list
    the first time we see a '- ' item.
    """

    def _unquote(s: str):
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")
        ):
            return s[1:-1]
        if s == "[]":
            return []
        if s == "null" or s == "":
            return ""
        return s

    root: dict = {}
    # stack of (indent, container, parent, parent_key)
    stack: list = [(-1, root, None, None)]

    def current():
        return stack[-1]

    def pop_to(indent: int):
        while stack and stack[-1][0] >= indent and stack[-1][0] != -1:
            stack.pop()

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        pop_to(indent)
        c_indent, container, parent, pkey = current()

        if content.startswith("- "):
            # list item — swap container to list if it's still an empty dict
            if isinstance(container, dict) and not container and parent is not None:
                new_list: list = []
                parent[pkey] = new_list
                stack[-1] = (c_indent, new_list, parent, pkey)
                container = new_list
            if not isinstance(container, list):
                # unexpected; skip
                continue
            item_text = content[2:].strip()
            if ":" in item_text:
                # list of dicts: "- key: value"
                k, _, v = item_text.partition(":")
                d: dict = {}
                d[k.strip()] = _unquote(v) if v.strip() else ""
                container.append(d)
                # items that follow on deeper indent belong to this dict
                stack.append((indent, d, container, len(container) - 1))
            else:
                container.append(_unquote(item_text))
            continue

        if ":" in content:
            key, _, value = content.partition(":")
            key = key.strip()
            value = value.strip()
            if not isinstance(container, dict):
                # switching back from a list-dict context — pop and retry
                stack.pop()
                _, container, _, _ = current()
            if value == "":
                new_dict: dict = {}
                container[key] = new_dict
                stack.append((indent, new_dict, container, key))
            else:
                container[key] = _unquote(value)

    return root


def _listify(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x]
    return [str(v)]


def build_queries(icp: dict) -> list[str]:
    queries: list[str] = []

    industries = _listify(icp.get("firmographics", {}).get("industries"))
    geos = _listify(icp.get("firmographics", {}).get("geographies"))
    titles = _listify(icp.get("persona", {}).get("titles"))
    stack_bad = _listify(icp.get("firmographics", {}).get("stack_must_not_have"))
    triggers = _listify(icp.get("triggers"))
    pain = icp.get("pain", "") or ""

    geo_clause = f' "{geos[0]}"' if geos else ""
    ind_clause = f' "{industries[0]}"' if industries else ""

    # A. Hiring signals — one query per persona title
    for t in titles[:4]:
        queries.append(
            f'"{t}"{ind_clause}{geo_clause} (site:linkedin.com/jobs OR '
            f'site:jobs.lever.co OR site:boards.greenhouse.io) "posted"'
        )

    # B. Failed-attempt signals — one query per trigger phrase
    for trig in triggers[:5]:
        if not trig:
            continue
        queries.append(
            f'"{trig}" ("didn\'t work" OR "waste of money" OR "switching from"){ind_clause}'
        )

    # C. Funding / change signals
    if industries:
        queries.append(
            f'{ind_clause.strip()} ("raised" OR "series A" OR "series B" OR '
            f'"new VP" OR "new head of") last 30 days'
        )

    # D. Public frustration — pain-phrase on posts
    if pain:
        queries.append(
            f'"{pain[:80]}" (site:linkedin.com/posts OR site:twitter.com OR site:news.ycombinator.com)'
        )

    # E. Tool / vendor churn — for each disallowed tool
    for tool in stack_bad[:3]:
        queries.append(
            f'"{tool}" ("sunset" OR "acquired" OR "price increase" OR "end of life" OR '
            f'"switching from {tool}")'
        )

    return [q for q in queries if q.strip()]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: search_signals.py path/to/icp.yaml", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    icp = parse_minimal_yaml(path.read_text(encoding="utf-8"))
    for q in build_queries(icp):
        print(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
