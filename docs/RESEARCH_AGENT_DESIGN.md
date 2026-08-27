# Deep Research Agent Design

A specialized agent for autonomous quantitative research: discovering, validating, and publishing trading hypotheses.

## Problem Statement

Current agent generates proposals from a fixed parameter grid + LLM reasoning. This works but doesn't:
- Discover novel strategies from literature/data
- Build confidence through research validation
- Generate publishable reasoning with citations
- Learn which research methodologies work

**Research Agent fixes this** by orchestrating multi-step investigations:

```
Research Plan → Literature Search → Hypothesis Formation → Backtest Validation
    ↓                                                              ↓
Data Analysis → Parameter Optimization → Final Report & Claims
```

## Architecture

### Three Specialized Sub-Agents

#### 1. **Literature Agent** (Tavily + Reasoning)
Discovers strategy ideas from the internet:

```
user_query: "momentum strategies in volatile markets"
    ↓
tools: search_strategy_research, extract_content
    ↓
outputs: [
  {
    "paper": "Title...",
    "source": "URL",
    "key_idea": "Use shorter windows in crisis regimes",
    "parameters": [{"fast": 5, "slow": 20}],
    "relevance_score": 0.85,
  },
  ...
]
```

Tool calls:
- `search_strategy_research()` — find related papers
- `fetch_content()` — extract key ideas
- `extract_citations()` — build citation graph

#### 2. **Hypothesis Agent** (Reasoning + Validation)
Transforms research into testable hypotheses:

```
literature_findings: [...]
    ↓
reasoning: "These papers suggest X is true in regimes Y.
           Here's my hypothesis and how to test it."
    ↓
outputs: [
  {
    "hypothesis": "Mean reversion wins in crisis regimes with wide bands",
    "regime_characteristic": "high_volatility",
    "proposed_parameters": [{window: 30, num_std: 3.0}],
    "research_basis": ["paper1", "paper2"],
    "predicted_sharpe": 0.6,
    "confidence": 0.7,
  },
  ...
]
```

Tool calls:
- `get_regime_context()` — understand current conditions
- `extract_parameter_recommendations()` — parameter space
- `search_market_sentiment()` — validate timing

#### 3. **Publication Agent** (Analysis + Reporting)
Validates and publishes findings:

```
hypothesis_results: [...]
    ↓
backtesting: "Sharpe 0.65 vs. predicted 0.6. Hypothesis confirmed."
    ↓
outputs: [
  {
    "title": "Mean Reversion in Crisis Regimes",
    "hypothesis": "...",
    "results": {
      "backtested_sharpe": 0.65,
      "generalization_gap": 0.08,
      "summary": "Hypothesis confirmed on validation data."
    },
    "citations": [...],
    "recommendations": ["Deploy in crisis regimes", "Monitor stability"],
    "falsifiable_claims": [
      "This strategy will outperform in VIX > 80th percentile"
    ],
  },
  ...
]
```

Tool calls:
- `run_benchmark_to_assess_quality()` — validate
- `fetch_citations()` — build references
- `generate_report()` — publication output

## Full Research Loop

```python
research_agent(
    research_task="Find momentum strategies optimized for low-volatility regimes",
    context=regime_context,
    depth="thorough",  # or "quick"
    max_hypotheses=5,
) → ResearchReport[]
```

### Execution Flow

1. **Planning**
   - Decompose research task
   - Identify relevant search terms, regimes, strategies
   - Set confidence/evidence thresholds

2. **Literature Search** (Literature Agent)
   - Search academic papers, industry research, blogs
   - Aggregate findings by theme
   - Extract key parameters and ideas

3. **Hypothesis Generation** (Hypothesis Agent)
   - Synthesize literature into testable hypotheses
   - Map to current market regime
   - Predict expected performance

4. **Validation** (Publication Agent)
   - Backtest hypotheses
   - Compute generalization gap
   - Score research quality (citations, evidence count, consistency)

5. **Publication**
   - Generate research report with full methodology
   - Attach falsifiable claims
   - Store in memory for future refinement

## Data Structures

### ResearchTask
```python
@dataclass
class ResearchTask:
    query: str  # "Find trend-following strategies for bears"
    depth: str  # "quick" | "thorough"
    context: RegimeContext
    strategy_types: List[str]  # ["momentum", "trend_following"]
    max_hypotheses: int = 5
    cite_threshold: float = 0.5  # Min relevance to cite
```

### ResearchFinding
```python
@dataclass
class ResearchFinding:
    source: str  # URL or paper title
    key_insight: str
    suggested_parameters: List[Dict]
    relevance_score: float  # 0-1
    citations: List[str]
```

### Hypothesis
```python
@dataclass
class Hypothesis:
    title: str
    description: str
    regime_characteristic: str
    proposed_parameters: List[Dict]
    research_basis: List[ResearchFinding]
    predicted_sharpe: float
    confidence: float
    falsifiable_claims: List[str]
```

### ResearchReport
```python
@dataclass
class ResearchReport:
    title: str
    hypothesis: Hypothesis
    methodology: str  # How was it tested
    results: Dict  # Backtest results
    generalization_gap: float
    quality_score: float  # Research rigor scoring
    citations: List[str]
    recommendations: List[str]
    timestamp: str
```

## Integration with Main Agent

The research agent runs **upstream** of the proposal generator:

```
Main Agent Loop:
    analyze → hypothesize → backtest → reflect → store
                   ↑
            Research Agent (optional)
            if regime_confidence < 0.7:
                run deep research
                → high-quality proposals
```

**Triggering conditions:**
- Low regime confidence (`< 0.7`)
- New market regime detected
- No recent research on current regime
- Prior proposals consistently underperforming

## Tools Required

| Tool | Purpose | Status |
|------|---------|--------|
| `search_strategy_research` | Find papers/blog posts | ✓ Exists |
| `fetch_and_extract_content` | Get full text from URL | TODO |
| `search_market_sentiment` | Market conditions | ✓ Exists |
| `run_hypothesis_test` | Statistical validation | TODO |
| `generate_markdown_report` | Publication format | TODO |
| `extract_citations_graph` | Citation relationships | TODO |

## Quality Metrics

### Research Quality Scoring

```python
quality_score = (
    0.3 * evidence_density +      # # citations / hypotheses
    0.25 * specificity +          # Parameter precision
    0.2 * backtest_rigor +        # Generalization gap
    0.15 * consistency +          # Results across regimes
    0.1 * recency                 # Age of sources
)
```

### Falsifiable Claim Tracking

Each hypothesis ships with explicit predictions:
- "This will outperform in regimes with VIX > 80th percentile"
- "Sharpe ratio will exceed 0.5 on unseen data"
- "Strategy will win on 70%+ of trades vs. benchmark"

Scored later against realized outcomes.

## Deployment Considerations

### Compute Budget
- **Quick research**: 2-3 tool calls, 1-2 hypotheses, ~30s
- **Thorough research**: 10-15 tool calls, 3-5 hypotheses, ~2 mins

### Cost
- Tavily: ~$0.01-0.05 per search
- Backtest: ~0.1s per hypothesis
- Claude: ~$0.001 per research task (token-efficient)

### Constraints
- No offline sources (only web-searchable)
- Hypothesis space is finite (parameter grid bounded)
- Literature coverage biased toward online content

## Success Criteria

✓ **Discover novel strategies** not in the parameter grid
✓ **Cite research** with real sources and extracts
✓ **Predict accurately** (realized Sharpe within 15% of prediction)
✓ **Work in low-confidence regimes** (improve generalization)
✓ **Produce publications** suitable for archival

## Implementation Roadmap

### Phase 1: Literature Agent
- [ ] Implement `fetch_and_extract_content()`
- [ ] Test on 5-10 known papers
- [ ] Validate citation extraction

### Phase 2: Hypothesis Agent
- [ ] Wire to Literature Agent
- [ ] Generate 3-5 hypotheses per research task
- [ ] Score by regime applicability

### Phase 3: Publication Agent
- [ ] Validate hypothesis against held-out backtest data
- [ ] Compute quality metrics
- [ ] Generate markdown reports

### Phase 4: Full Loop
- [ ] Integrate into main agent (optional invocation)
- [ ] Track falsifiable claims accuracy
- [ ] Use research quality to weight proposal confidence

## References

- `docs/TOOL_INTEGRATION_GUIDE.md` — Tool system
- `src/agent/tools/` — Available tools
- `DESIGN.md` — Main agent architecture
