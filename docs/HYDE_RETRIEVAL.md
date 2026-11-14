# Synthetic hypothesis retrieval sidecar

`src.agent.hyde_retrieval.HydeRetrievalTool` implements an offline HyDE-inspired
research step: an injected generator emits synthetic strategy descriptions, the
tool retrieves similar market-time-visible evidence, and maps the idea to a
canonical point in `ParameterGrid`. The tool does not evaluate candidates or
claim validation Sharpe.

```python
from src.agent.hyde_retrieval import Evidence, HydeRetrievalTool, HypothesisRequest

# Supply an object with generate(request) -> a sequence of SyntheticHypothesis.
# Evidence rows may be adapted from MemoryService.visible_trials/visible_notes.
tool = HydeRetrievalTool(generator, evidence_rows)
candidates = tool.run(HypothesisRequest(
    regime="high volatility", strategy_types=("momentum",), as_of="2024-06-30",
    max_hypotheses=3, max_evidence=8,
))
```

Evidence without parseable market-end and known-at dates is excluded. `as_of`
cuts off the market period; `knowledge_as_of` optionally tightens the decision
time for when the record became available and otherwise defaults to `as_of`.
Retrieval records keep their source, both dates, evidence tier, and any explicitly supplied measured metric.
Similarities and quality scores only prioritize inspection; measured metrics
remain separately labeled evidence and are not inferred from text. The default
matcher is deterministic lexical cosine, while a caller can inject a local
embedding similarity function. Generation is injected and never creates a
network or model dependency. Proposal and evidence limits cap work, and mapped
parameters always come from the strategy's configured canonical grid. Ideas
with neither usable parameter hints nor retrieved parameter evidence are omitted
instead of receiving an arbitrary grid default. Request caps are 16 hypotheses
and 64 evidence rows. Candidate
`evaluation_status` starts as `unevaluated` and must be evaluated by the normal
backtest and quality gate before any performance claim is made.

## Optional OpenAI generator

`ProviderHydeGenerator` is an opt-in adapter in `src.agent.generators.provider_hyde`.
Install `agentquant[llm]` and set `OPENAI_API_KEY` to construct it without an
injected client. It asks for a bounded JSON array, accepts only requested
strategy types and exact canonical parameter-grid entries, and caps both
completion and description size. Its `metadata` records the model and
temperature; generation is a proposal step, not evidence of strategy
performance. Tests can inject a client implementing
`client.chat.completions.create`, so they need no SDK credentials or API call.
