# AgentQuant Scripts

Utility scripts for running the agent, measuring harness improvement, and evolving the harness.

## Harness Evolution POC

End-to-end demonstration of harness evolution: run agent → measure improvement → evolve → repeat.

### Setup

```bash
# Install dependencies
pip install -e ".[llm]"

# Set API keys
export ANTHROPIC_API_KEY=sk-...  # For Claude tool-use
export TAVILY_API_KEY=tvly-...    # For web search
```

### Run

```bash
# Basic run (2 epochs)
python scripts/harness_evolution_poc.py

# Customize strategy and asset
python scripts/harness_evolution_poc.py --strategy momentum --asset SPY --epochs 2

# Save results to custom file
python scripts/harness_evolution_poc.py --output my_results.json
```

### What It Does

**Epoch 1: Baseline**
1. Load market data (SPY, QQQ, IWM, TLT, GLD)
2. Run agent with v1_base harness
   - Analyze market regime
   - Generate proposals via tool orchestrator (Claude + Tavily)
   - Backtest proposals
   - Reflect and score falsifiable claims
3. Record checkpoint: Sharpe, generalization gap, tool efficiency

**Evolution Step**
- Analyze what worked in epoch 1
- If tools helped (>2 tool calls + Sharpe > 0.3): promote tool-based proposals
- If generalization was good (gap < 0.1): encourage conservative retrain
- Otherwise: collect more market context

**Epoch 2: Evolved Harness**
1. Run agent with evolved harness (v2_*)
2. Backtest and measure
3. Record checkpoint

**Measurement**
- Sharpe improvement delta
- Generalization gap reduction
- Tool efficiency (calls per epoch)
- Improvement percentage

### Output

`harness_evolution.json`:
```json
{
  "strategy": "momentum",
  "asset": "SPY",
  "checkpoints": [
    {
      "epoch": 1,
      "harness_version": "v1_base",
      "best_sharpe": 0.45,
      "generalization_gap": 0.12,
      "tool_calls_made": 3,
      "proposals_considered": 5,
      "accepted": true
    },
    {
      "epoch": 2,
      "harness_version": "v2_tool_aware",
      "best_sharpe": 0.52,
      "generalization_gap": 0.08,
      "tool_calls_made": 4,
      "proposals_considered": 5,
      "accepted": true
    }
  ],
  "improvement": {
    "sharpe_delta": 0.07,
    "sharpe_pct": 15.6,
    "gap_reduction": 0.04,
    "epochs": 2
  }
}
```

## What the POC Demonstrates

✓ **Agent with tools** — Uses Claude tool-use + Tavily for market context  
✓ **Proposal parsing** — Extracts JSON proposals from Claude responses  
✓ **Harness evolution** — Modifies approach based on what worked  
✓ **Measurement** — Tracks Sharpe, gaps, and efficiency  
✓ **Reproducibility** — Full checkpoint history saved  

## Next Steps

### Short-term
1. Implement walk-forward validation (train/val/held-out split)
2. Implement structured falsifiable-claim evaluation across epochs
3. Extend evolution strategy (parameter grid adaptation)

### Medium-term
1. Research agent implementation (literature search, hypothesis generation)
2. Multi-agent swarm (specialist agents)
3. Automated hyperparameter tuning

### Long-term
1. Continuous learning from live market data
2. Publication of discovered strategies
3. Regulatory compliance and risk management
