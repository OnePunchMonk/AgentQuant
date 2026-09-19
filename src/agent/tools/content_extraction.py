"""
Content extraction for the Literature Discovery tool (research agent Phase 1,
issue #22): fetch a URL and pull out plain text + basic citation metadata
(title, authors, date) without pulling in a new HTML-parsing dependency --
`html.parser` (stdlib) is enough for the "get readable text + a few meta
tags" use case here.

Kept separate from `registry.py` so the parsing logic (pure, no network) is
unit-testable against fixture HTML without a live HTTP call.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tags whose contents are never readable body text. `head` is deliberately
# excluded: its only non-empty children we care about (script/style) are
# skipped directly, and `<title>` inside it must stay readable as a
# fallback title source (handled separately via `_in_title_tag`).
_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "form"}

MAX_CONTENT_CHARS = 8000
REQUEST_TIMEOUT_S = 15


@dataclass
class ExtractedContent:
    url: str
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    text: str = ""
    relevance_score: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "authors": self.authors,
            "published_date": self.published_date,
            "text": self.text,
            "relevance_score": self.relevance_score,
            "error": self.error,
        }


class _TextAndMetaParser(HTMLParser):
    """Collects visible text plus a handful of citation-relevant meta tags.

    Recognizes both `<meta name=... content=...>` and `<meta property=...
    content=...>` (OpenGraph/Dublin Core/citation_* conventions used by
    arXiv, SSRN, and most academic/blog publishing platforms) for title,
    author(s), and publish date.
    """

    _META_TITLE_KEYS = {"citation_title", "og:title", "dc.title"}
    _META_AUTHOR_KEYS = {"citation_author", "author", "dc.creator", "article:author"}
    _META_DATE_KEYS = {
        "citation_publication_date", "citation_date", "og:article:published_time",
        "article:published_time", "dc.date", "date",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta_title: Optional[str] = None
        self.title_tag: Optional[str] = None
        self.authors: List[str] = []
        self.published_date: Optional[str] = None
        self._text_parts: List[str] = []
        self._skip_depth = 0
        self._in_title_tag = False

    @property
    def title(self) -> Optional[str]:
        # Citation-specific meta tags (arXiv/SSRN/etc.) are more reliable
        # than the raw <title> tag, which often has a site name appended.
        return self.meta_title or self.title_tag

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr_dict = dict(attrs)
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title_tag = True
        if tag == "meta":
            key = (attr_dict.get("name") or attr_dict.get("property") or "").lower()
            content = attr_dict.get("content")
            if not content:
                return
            if key in self._META_TITLE_KEYS and not self.meta_title:
                self.meta_title = content.strip()
            elif key in self._META_AUTHOR_KEYS:
                self.authors.append(content.strip())
            elif key in self._META_DATE_KEYS and not self.published_date:
                self.published_date = content.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title_tag = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        stripped = data.strip()
        if not stripped:
            return
        if self._in_title_tag and not self.title_tag:
            self.title_tag = stripped
        else:
            self._text_parts.append(stripped)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._text_parts)).strip()


def parse_html(html: str) -> Dict[str, Any]:
    """Pure parsing step (no network) -- extracts title/authors/date/text
    from an HTML string. Never raises on malformed HTML: html.parser is
    tolerant, and any parser exception is caught and surfaced as an empty
    result rather than propagated."""
    parser = _TextAndMetaParser()
    try:
        parser.feed(html)
    except Exception as exc:  # malformed markup should degrade, not crash the tool
        logger.warning("HTML parsing failed: %s", exc)
        return {"title": None, "authors": [], "published_date": None, "text": ""}
    return {
        "title": parser.title,
        "authors": parser.authors,
        "published_date": parser.published_date,
        "text": parser.get_text()[:MAX_CONTENT_CHARS],
    }


def score_relevance(text: str, query: str) -> Optional[float]:
    """Simple, dependency-free relevance score: fraction of the query's
    distinct words (3+ chars, to skip stopwords like "the"/"and"/"for")
    that appear (case-insensitively) anywhere in the extracted text.
    Returns None if no query was supplied -- absence of a query is not the
    same as zero relevance and callers should not conflate the two."""
    if not query:
        return None
    query_words = {w for w in re.findall(r"[a-zA-Z']+", query.lower()) if len(w) >= 3}
    if not query_words:
        return None
    text_lower = text.lower()
    hits = sum(1 for w in query_words if w in text_lower)
    return round(hits / len(query_words), 3)


def fetch_and_extract_content(url: str, query: str = "") -> Dict[str, Any]:
    """Fetch `url` and extract readable text + citation metadata, scored
    for relevance against `query` if one is given.

    Fails closed like every other tool in this registry: network errors,
    non-HTML content, and non-2xx responses are reported in the `error`
    field of the returned dict, never raised -- the orchestrator/agent loop
    must be able to treat "the fetch failed" as an observable tool result,
    not a crash.
    """
    import requests

    if not url or not re.match(r"^https?://", url):
        return ExtractedContent(url=url, error=f"invalid or missing URL: {url!r}").to_dict()

    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": "AgentQuant-ResearchAgent/1.0 (+literature discovery tool)"},
        )
    except requests.RequestException as exc:
        return ExtractedContent(url=url, error=f"request failed: {exc}").to_dict()

    if resp.status_code >= 400:
        return ExtractedContent(url=url, error=f"HTTP {resp.status_code}").to_dict()

    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type and "xml" not in content_type:
        return ExtractedContent(
            url=url, error=f"unsupported content-type for text extraction: {content_type!r}"
        ).to_dict()

    parsed = parse_html(resp.text)
    if not parsed["text"] and not parsed["title"]:
        return ExtractedContent(url=url, error="no extractable text found on page").to_dict()

    return ExtractedContent(
        url=url,
        title=parsed["title"],
        authors=parsed["authors"],
        published_date=parsed["published_date"],
        text=parsed["text"],
        relevance_score=score_relevance(parsed["text"] + " " + (parsed["title"] or ""), query),
    ).to_dict()
