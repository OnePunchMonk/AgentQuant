# Literature URL extraction

`fetch_and_extract_content(url, query="")` is a registered research tool for reading a public HTML/XML page and returning `{url, title, authors, published_date, text, relevance_score, error}`. It is wired into the default tool registry and is intended to follow search results, so the agent can inspect the source before citing it as evidence.

The extractor uses Python's standard-library HTML parser. It recognizes common citation, OpenGraph, and Dublin Core metadata names, prefers a citation title over the document `<title>`, omits script/style/navigation/footer/form content, and caps returned text at 8,000 characters. Relevance is a lightweight lexical score: the fraction of distinct query words of at least three characters found in the title or extracted text. It is a ranking hint, not a scientific relevance judgment.

## Fetch safety and limits

- Only HTTP and HTTPS URLs with a public hostname are accepted. Credentials, localhost names, and private, loopback, link-local, reserved, or otherwise non-global resolved addresses are rejected.
- HTTP redirects are not followed. A caller may inspect and resubmit a redirect target, which is validated independently.
- Requests have a 15-second timeout and are limited to 2 MB of response data; parsed output is limited to 8,000 characters.
- Non-HTML/XML content, HTTP errors, request failures, redirects, and pages without extractable text are returned in the `error` field instead of raising through the tool loop.

The URL check resolves hostnames before fetching and rejects a hostname if any resolved address is non-public. As with ordinary client-side DNS checks, this does not pin the validated address to the HTTP connection; deployments with hostile DNS environments should enforce outbound network policy as well.

## Verification

Offline behavior is covered by mocked tests in `tests/test_tools.py`. The six candidate real URLs in `tests/fixtures/test_urls.json` are checked by the opt-in script:

```bash
python scripts/verify_research_tool_urls.py
```

The live check requires outbound DNS/HTTPS access and currently treats five successful extractions as the acceptance threshold. URLs can change, rate-limit, or block automated clients, so this check is not part of deterministic CI. A successful fetch verifies page accessibility and output shape only; it does not establish the accuracy of extracted metadata or the quality of relevance scores.
