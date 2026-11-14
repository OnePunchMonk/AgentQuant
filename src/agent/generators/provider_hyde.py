"""Optional OpenAI-backed HyDE proposal adapter.

The OpenAI SDK is imported only when the adapter is instantiated without an
injected client. Model output is treated as untrusted and snapped to the
canonical strategy parameter grid before it reaches the retrieval pipeline.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from src.agent.hyde_retrieval import HypothesisRequest, SyntheticHypothesis
from src.agent.parameter_grid import ParameterGrid

MAX_OUTPUT_TOKENS = 512
MAX_DESCRIPTION_CHARS = 600
MAX_COMPLETION_CHARS = 12_000
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2


class ProviderHydeGenerator:
    """Generate bounded synthetic descriptions using an optional OpenAI client.

    Pass a client implementing ``client.chat.completions.create`` for tests or
    custom transports. With no client, ``openai`` must be installed and
    ``OPENAI_API_KEY`` configured. ``temperature`` is recorded with the model
    identifier in ``metadata`` for reproducibility.
    """

    def __init__(self, client: Any | None = None, *, model: str = DEFAULT_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE, grids: ParameterGrid | None = None):
        if not model.strip():
            raise ValueError("model must be non-empty")
        if not math.isfinite(temperature) or not 0 <= temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required for provider-backed generation")
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError("Install agentquant[llm] to use ProviderHydeGenerator") from exc
            client = OpenAI()
        self.client = client
        self.model = model
        self.temperature = float(temperature)
        self.grids = grids or ParameterGrid()

    @property
    def metadata(self) -> dict[str, object]:
        return {"provider": "openai", "model": self.model,
                "temperature": self.temperature, "max_output_tokens": MAX_OUTPUT_TOKENS}

    def generate(self, request: HypothesisRequest) -> list[SyntheticHypothesis]:
        if request.max_hypotheses == 0:
            return []
        allowed = tuple(dict.fromkeys(
            strategy for strategy in request.strategy_types if self.grids.get_grid(strategy)
        ))
        if not allowed:
            return []
        prompt = {
            "task": "Propose synthetic, testable trading strategy descriptions. Do not claim performance.",
            "regime": request.regime[:160],
            "allowed_strategy_types": allowed,
            "max_hypotheses": min(request.max_hypotheses, 16),
            "parameter_grids": {name: self.grids.get_grid(name) for name in allowed},
            "output_format": [{"description": "short string", "strategy_type": "one allowed type",
                               "proposed_params": "one exact parameter-grid object"}],
        }
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "system", "content": "Return only a JSON array. Follow the supplied schema."},
                      {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))}],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or len(content) > MAX_COMPLETION_CHARS:
            return []
        try:
            rows = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(rows, list):
            return []
        out: list[SyntheticHypothesis] = []
        for row in rows:
            if len(out) >= min(request.max_hypotheses, 16):
                break
            if not isinstance(row, dict):
                continue
            description = row.get("description")
            strategy = row.get("strategy_type")
            params = row.get("proposed_params", {})
            if (not isinstance(description, str) or not description.strip() or
                    not isinstance(strategy, str) or strategy not in allowed or
                    not isinstance(params, dict)):
                continue
            description = description.strip()[:MAX_DESCRIPTION_CHARS]
            canonical = next((grid_row for grid_row in self.grids.get_grid(strategy)
                              if params == grid_row), None)
            if canonical is None:
                continue
            out.append(SyntheticHypothesis(description, strategy, dict(canonical)))
        return out
