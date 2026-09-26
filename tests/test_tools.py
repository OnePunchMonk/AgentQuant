"""Tests for src/agent/tools -- registry wiring and the #22 literature
discovery tool (fetch_and_extract_content). Network calls are mocked so
this suite stays offline/deterministic; live-URL verification lives in
scripts/verify_research_tool_urls.py (opt-in, see its docstring)."""

from unittest.mock import Mock, patch

from src.agent.tools.content_extraction import (
    fetch_and_extract_content,
    parse_html,
    score_relevance,
)
from src.agent.tools.registry import get_default_registry

SAMPLE_HTML = """
<html>
<head>
  <title>Fallback Title</title>
  <meta name="citation_title" content="Momentum Strategies in Crisis Regimes">
  <meta name="citation_author" content="Jane Doe">
  <meta name="citation_author" content="John Smith">
  <meta name="citation_publication_date" content="2024-03-15">
  <script>var x = "should not appear in text";</script>
  <style>.h { color: red; }</style>
</head>
<body>
  <nav>Home | About | Contact</nav>
  <h1>Momentum Strategies in Crisis Regimes</h1>
  <p>We find that short window momentum outperforms in high volatility regimes.</p>
</body>
</html>
"""


def test_parse_html_extracts_citation_meta_over_title_tag():
    parsed = parse_html(SAMPLE_HTML)
    assert parsed["title"] == "Momentum Strategies in Crisis Regimes"
    assert parsed["authors"] == ["Jane Doe", "John Smith"]
    assert parsed["published_date"] == "2024-03-15"


def test_parse_html_excludes_script_style_and_nav_text():
    parsed = parse_html(SAMPLE_HTML)
    assert "should not appear" not in parsed["text"]
    assert "Home | About | Contact" not in parsed["text"]
    assert "short window momentum" in parsed["text"]


def test_parse_html_falls_back_to_title_tag_when_no_citation_meta():
    parsed = parse_html("<html><head><title>Plain Title</title></head><body><p>Body text.</p></body></html>")
    assert parsed["title"] == "Plain Title"


def test_parse_html_never_raises_on_malformed_markup():
    parsed = parse_html("<html><body><p>unclosed tags <div><span>text")
    assert "text" in parsed["text"]


def test_score_relevance_none_without_query():
    assert score_relevance("some text about momentum", "") is None


def test_score_relevance_counts_distinct_query_word_hits():
    text = "momentum strategies work well in crisis regimes"
    score = score_relevance(text, "momentum performance in crisis periods")
    # distinct 3+ char query words: momentum, performance, crisis, periods -> 2/4 hit
    assert score == 0.5


def test_score_relevance_full_match_is_one():
    text = "momentum strategies in crisis regimes"
    assert score_relevance(text, "momentum crisis") == 1.0


def _mock_response(status_code=200, content_type="text/html; charset=utf-8", text=SAMPLE_HTML):
    resp = Mock()
    resp.status_code = status_code
    resp.headers = {"Content-Type": content_type}
    resp.text = text
    return resp


def test_fetch_and_extract_content_happy_path():
    with patch("requests.get", return_value=_mock_response()) as mock_get:
        result = fetch_and_extract_content("https://arxiv.org/abs/1234.5678", query="momentum crisis")
    mock_get.assert_called_once()
    assert result["error"] is None
    assert result["title"] == "Momentum Strategies in Crisis Regimes"
    assert result["authors"] == ["Jane Doe", "John Smith"]
    assert result["relevance_score"] == 1.0


def test_fetch_and_extract_content_rejects_invalid_url():
    result = fetch_and_extract_content("not-a-url")
    assert result["error"] is not None
    assert "invalid" in result["error"]


def test_fetch_and_extract_content_reports_http_error_without_raising():
    with patch("requests.get", return_value=_mock_response(status_code=404, text="")):
        result = fetch_and_extract_content("https://example.com/missing")
    assert result["error"] == "HTTP 404"


def test_fetch_and_extract_content_reports_non_html_content_type():
    with patch("requests.get", return_value=_mock_response(content_type="application/pdf")):
        result = fetch_and_extract_content("https://example.com/paper.pdf")
    assert "unsupported content-type" in result["error"]


def test_fetch_and_extract_content_reports_network_failure_without_raising():
    import requests

    with patch("requests.get", side_effect=requests.ConnectionError("boom")):
        result = fetch_and_extract_content("https://example.com/unreachable")
    assert "request failed" in result["error"]


def test_fetch_and_extract_content_reports_empty_page():
    with patch("requests.get", return_value=_mock_response(text="<html><body></body></html>")):
        result = fetch_and_extract_content("https://example.com/blank")
    assert result["error"] == "no extractable text found on page"


def test_tool_registered_and_loads_in_default_registry():
    registry = get_default_registry()
    assert "fetch_and_extract_content" in registry.list_tools()
    tool = registry.get_tool("fetch_and_extract_content")
    assert tool.input_schema["required"] == ["url"]


def test_tool_appears_in_claude_tool_definitions():
    registry = get_default_registry()
    claude_tools = registry.to_claude_tools()
    names = [t["name"] for t in claude_tools]
    assert "fetch_and_extract_content" in names


def test_execute_tool_through_registry_dispatch():
    registry = get_default_registry()
    with patch("requests.get", return_value=_mock_response()):
        result = registry.execute_tool(
            "fetch_and_extract_content", {"url": "https://arxiv.org/abs/1234.5678", "query": "momentum"}
        )
    assert result["success"] is True
    assert result["output"]["title"] == "Momentum Strategies in Crisis Regimes"
