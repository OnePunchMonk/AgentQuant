#!/usr/bin/env python3
"""
Manual/live verification for the #22 literature discovery tool
(fetch_and_extract_content): hits real academic/industry/blog URLs from
tests/fixtures/test_urls.json and checks the output has the expected
shape (text, title/authors/date where the page provides them, a
relevance score against the given query).

Not part of the default CI test suite: these are live network calls
against third-party sites, which are slow and can flake or rate-limit
independent of any code change here. Run manually:

    python3 scripts/verify_research_tool_urls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.tools.content_extraction import fetch_and_extract_content

FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "test_urls.json"


def main() -> int:
    cases = json.loads(FIXTURE_PATH.read_text())["urls"]
    n_ok, n_failed = 0, 0
    for case in cases:
        result = fetch_and_extract_content(case["url"], query=case.get("query", ""))
        ok = result["error"] is None and bool(result["text"])
        status = "OK" if ok else f"FAIL ({result['error']})"
        print(f"[{status}] {case['category']:20s} {case['url']}")
        if ok:
            print(f"         title={result['title']!r} authors={result['authors']} "
                  f"date={result['published_date']!r} relevance={result['relevance_score']} "
                  f"text_len={len(result['text'])}")
            n_ok += 1
        else:
            n_failed += 1

    print(f"\n{n_ok}/{len(cases)} URLs extracted successfully.")
    if n_ok < 5:
        print("FAIL: fewer than 5 URLs returned usable output (issue #22 acceptance criterion).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
