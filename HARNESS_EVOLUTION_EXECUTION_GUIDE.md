# Harness Evolution Execution Guide

Complete guide to running the end-to-end harness evolution POC demonstrating self-improving agent architecture.

## What We Built

### 1. **Tool Orchestration Framework**
- Tool registry with 5 composable tools
- Claude tool-use orchestrator
- Proposal parsing from JSON responses
- Graceful fallback to baseline if tools unavailable

### 2. **Harness Evolution Runner**
- Two-epoch POC demonstration
- Automatic harness evolution based on results
- Improvement measurement and tracking
- Checkpoint persistence

### 3. **Measurement System**
- Sharpe ratio tracking
- Generalization gap computation
- Tool efficiency metrics
- Checkpoint history

## Setup Instructions

### Prerequisites
```bash
# Python 3.10+
python3 --version

# Clone repo and navigate to it
cd AgentQuant
```

### Install Dependencies

```bash
# Install core + LLM dependencies
pip install -e ".[llm]"

# This installs:
# - anthropic>=0.30 (Claude API for tool-use)
# - tavily-python>=0.3 (Web search)
# - All other required packages
```

### Configure Environment

```bash
# Create .env file from template
cp .env.example .env

# Add your API keys to .env
# ⚠️  NEVER commit .env to git (it's in .gitignore)

# Required:
ANTHROPIC_API_KEY=sk-...    # From https://console.anthropic.com

# Optional:
TAVILY_API_KEY=tvly-...     # From https://tavily.com (enables web search)
GOOGLE_API_KEY=...          # Existing, for fallback LLM
```

### Verify Setup

```bash
# Check that all tools are working
python3 scripts/verify_tools.py

# Expected output:
# ✓ Tool Registry
# ✓ Orchestrator
# (API keys optional for initial check)
```

## Running the POC

### Basic Run (Default: 2 Epochs)

```bash
python3 scripts/harness_evolution_poc.py
```

### Customized Run

```bash
# Specify strategy, asset, and epochs
python3 scripts/harness_evolution_poc.py \
  --strategy momentum \
  --asset SPY \
  --epochs 2

# Save results to custom file
python3 scripts/harness_evolution_poc.py \
  --output my_harness_evolution.json
```

## What Happens During Execution

### Epoch 1: Baseline (v1_base)

```
1. Load market data (5 years of OHLCV)
   └─ SPY, QQQ, IWM, TLT, GLD

2. Analyze
   ├─ Compute technical features
   ├─ Detect market regime (VIX percentile)
   └─ Build RegimeContext

3. Hypothesize with Tools
   ├─ Tool 1: get_regime_context() → retrieve stored knowledge
   ├─ Tool 2: search_market_sentiment() → search web for market conditions (if Tavily key)
   ├─ Claude reasoning with tools
   └─ Parse proposals (JSON) → Proposal objects

4. Backtest
   ├─ Test each proposal with realistic costs
   ├─ Compute Sharpe, returns, drawdown
   └─ Select best

5. Reflect & Score
   ├─ Score falsifiable claims
   ├─ Measure generalization gap
   └─ Accept or retry

6. Store
   └─ Persist best result and statistics

CHECKPOINT v1_base:
  best_sharpe: 0.45 (example)
  generalization_gap: 0.12
  tool_calls_made: 3
  proposals_considered: 5
  accepted: true
```

### Evolution Step

```
Analyze what worked:
  - 3 tool calls (good!)
  - Sharpe > 0.3 (good!)
  - Gap < 0.15 (acceptable)

Decision: PROMOTE TOOL-BASED PROPOSALS
  Rationale: Tools helped discover good parameters
  
Evolution: v1_base → v2_tool_aware
  Change: Increase weight on tool-discovered parameters
```

### Epoch 2: Evolved (v2_tool_aware)

```
1-6. Same process as Epoch 1, but with evolved harness

CHECKPOINT v2_tool_aware:
  best_sharpe: 0.52 (IMPROVED!)
  generalization_gap: 0.08 (IMPROVED!)
  tool_calls_made: 4
  proposals_considered: 5
  accepted: true
```

### Measurement

```
IMPROVEMENT SUMMARY:
  ✓ Sharpe Delta:      +0.07 (+15.6%)
  ✓ Gap Reduction:     -0.04 (4%)
  ✓ Tool Efficiency:   Stable (3→4 calls)
  ✓ Status:            Evolved harness outperforms
```

## Output Files

### `harness_evolution.json` (Default)

Complete results in JSON format:

```json
{
  "strategy": "momentum",
  "asset": "SPY",
  "checkpoints": [
    {
      "epoch": 1,
      "timestamp": "2026-08-28T12:00:00",
      "harness_version": "v1_base",
      "best_sharpe": 0.45,
      "avg_sharpe": 0.32,
      "generalization_gap": 0.12,
      "tool_calls_made": 3,
      "proposals_considered": 5,
      "accepted": true,
      "results": [...]  // Full backtest results
    },
    {
      "epoch": 2,
      "harness_version": "v2_tool_aware",
      "best_sharpe": 0.52,
      "avg_sharpe": 0.38,
      "generalization_gap": 0.08,
      "tool_calls_made": 4,
      "proposals_considered": 5,
      "accepted": true,
      "results": [...]
    }
  ],
  "improvement": {
    "sharpe_delta": 0.07,
    "sharpe_pct": 15.6,
    "gap_reduction": 0.04,
    "tool_efficiency": {
      "epoch_0_tools": 3,
      "epoch_1_tools": 4
    },
    "epochs": 2
  }
}
```

## Interpreting Results

### Sharpe Improvement

```
sharpe_delta = 0.07  (positive = improvement)
sharpe_pct = 15.6%   (relative improvement)

Interpretation: Evolved harness improved risk-adjusted returns by 15.6%
```

### Generalization Gap

```
gap_reduction = 0.04  (positive = reduced gap)

High gap (>0.15) = harness overfitting to training data
Low gap (<0.05) = harness generalizes well

Goal: Gap reduction shows evolved harness generalizes better
```

### Tool Efficiency

```
epoch_0_tools: 3  (tools called in epoch 1)
epoch_1_tools: 4  (tools called in epoch 2)

More tools ≠ better. Efficiency = results per tool call.
If Sharpe ↑ and tools ↑, tool effectiveness is high.
```

## Troubleshooting

### "No module named 'pydantic'"

```bash
# Reinstall dependencies
pip install -e ".[llm]"
```

### "ANTHROPIC_API_KEY not found"

```bash
# Check .env is created and contains the key
cat .env | grep ANTHROPIC_API_KEY

# If missing, add it:
export ANTHROPIC_API_KEY=sk-your-key-here

# Or add to .env:
ANTHROPIC_API_KEY=sk-your-key-here
```

### "Tavily search failed"

Tavily is optional. If TAVILY_API_KEY not set, tools degrade gracefully:
- Sentiment search returns empty results
- Agent continues with other tools
- Falls back to ProposalGenerator

### "Agent running too slowly"

First run loads 5 years of data and runs backtests. Normal time: 5-15 minutes.

Optimize with:
```bash
# Use smaller period (e.g., 2 years)
# Modify config.yaml: data.yfinance_period = "2y"

# Or skip one asset to reduce compute
# Modify config.yaml: universe = ["SPY", "QQQ"]
```

## Next Steps

### 1. Extend Evolution Strategy
Current: Only 2 epochs, basic harness adaptation.

Future:
- Multi-epoch loop (4+ epochs)
- Grid evolution (add/remove parameters)
- Prompt template evolution
- Tool definition evolution

### 2. Walk-Forward Validation
Current: Simple train/test split.

Future:
- Implement three-way split: train / validation / held-out
- Only held-out touched at end
- Compute true generalization gap

### 3. Research Agent
Implement full research loop:
- Literature search → hypothesis generation → validation
- See `docs/RESEARCH_AGENT_DESIGN.md`

### 4. Multi-Agent Swarm
Deploy specialist agents:
- Literature Agent (research)
- Hypothesis Agent (reasoning)
- Critic Agent (evaluation)
- Regime Analyst (market conditions)

## References

- `docs/TOOL_INTEGRATION_GUIDE.md` — Tool system architecture
- `docs/RESEARCH_AGENT_DESIGN.md` — Research agent specification
- `DESIGN.md` — Agent graph architecture
- `CHANGELOG.md` — Project history and features
- `scripts/README.md` — Script documentation

## Success Criteria

✓ **Tool orchestration works** — Claude tool-use loop executes without error  
✓ **Proposals parsed** — JSON proposals extracted from Claude responses  
✓ **Harness evolves** — Second epoch uses evolved harness  
✓ **Improvement measured** — Sharpe delta computed correctly  
✓ **Results saved** — JSON checkpoint history persisted  

## Citation

If you use this harness evolution POC in research, please cite:

```
@software{agentquant_2026,
  title={AgentQuant: Self-Improving Agentic System for Quantitative Trading},
  author={OnePunchMonk},
  year={2026},
  url={https://github.com/OnePunchMonk/AgentQuant}
}
```

And reference the research papers this is based on:
- Weng et al. (2026) "Harness Engineering for Self-Improvement"
- arXiv:2607.07663 "Recursive Self-Improvement in AI"
- arXiv:2607.12227 "Rethinking the Evaluation of Harness Evolution"
