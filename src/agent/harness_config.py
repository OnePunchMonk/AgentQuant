"""
Harness Configuration — Editable Surface for Agent Evolution

Stores and evolves harness parameters across iterations.
Each epoch can propose modifications to these surfaces.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UnsupportedHarnessKnobError(ValueError):
    """Raised when a harness config sets a knob the runtime does not (yet) wire through."""


@dataclass
class HarnessConfig:
    """Complete harness configuration for one epoch."""

    # Identity
    version: str  # "v1_base", "v2_tool_aware", etc.
    epoch: int
    created: str  # ISO timestamp

    # Proposal Strategy
    use_tools: bool = False
    use_web_search: bool = False
    use_ensemble: bool = False
    tool_weight: float = 0.5  # How much to trust tool proposals vs grid

    # Prompt Configuration
    prompt_template: str = "default"
    prompt_context: Dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.2
    max_retries: int = 2

    # Parameter Grid Configuration
    grid_adaptation_strategy: Optional[str] = None  # "shrink_to_winners", "expand_neighborhood", etc.
    grid_params: Dict[str, List[Any]] = field(default_factory=dict)
    grid_focus_regions: List[Dict[str, Any]] = field(default_factory=list)  # High-performance regions

    # Claim Scoring Configuration
    track_falsifiable_claims: bool = False
    claim_weighting: float = 0.5  # Weight in final score

    # Evaluation Thresholds
    min_acceptable_sharpe: float = 0.3
    max_acceptable_drawdown: float = 0.20
    required_win_rate: float = 0.0

    # Metadata
    description: str = ""
    reasoning: str = ""
    expected_improvement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created"] = str(data["created"])
        return data

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessConfig":
        """Create from dictionary."""
        return cls(**data)


class HarnessConfigManager:
    """Manages harness configurations across epochs."""

    def __init__(self, config_dir: str = ".harness"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def save_config(self, config: HarnessConfig) -> None:
        """Save harness config to file."""
        outfile = self.config_dir / f"{config.version}.json"
        with open(outfile, "w") as f:
            f.write(config.to_json())
        logger.info(f"Saved harness config: {outfile}")

    def load_config(self, version: str) -> HarnessConfig:
        """Load harness config from file."""
        infile = self.config_dir / f"{version}.json"
        with open(infile, "r") as f:
            data = json.load(f)
        return HarnessConfig.from_dict(data)

    def list_configs(self) -> List[str]:
        """List all saved harness versions."""
        return [f.stem for f in self.config_dir.glob("*.json")]


# Predefined harness configurations for each epoch

def harness_v1_base() -> HarnessConfig:
    """v1: Baseline (grid search only)."""
    from datetime import datetime

    return HarnessConfig(
        version="v1_base",
        epoch=1,
        created=datetime.now().isoformat(),
        use_tools=False,
        use_web_search=False,
        use_ensemble=False,
        prompt_template="grid_search_default",
        temperature=0.2,
        max_retries=2,
        grid_adaptation_strategy=None,
        track_falsifiable_claims=False,
        min_acceptable_sharpe=0.3,
        max_acceptable_drawdown=0.20,
        description="Baseline grid search (no tools)",
        reasoning="Establish baseline performance with grid sampling",
        expected_improvement="Reference point (0%)",
    )


def harness_v2_tool_aware() -> HarnessConfig:
    """v2: Enable tool orchestration."""
    from datetime import datetime

    config = harness_v1_base()
    config.version = "v2_tool_aware"
    config.epoch = 2
    config.created = datetime.now().isoformat()
    config.use_tools = True
    config.use_web_search = True
    config.tool_weight = 0.6
    config.prompt_template = "tool_aware_default"
    config.track_falsifiable_claims = True
    config.description = "Enable tool orchestration (Claude + Tavily)"
    config.reasoning = "Tools help gather market context; enable web search integration"
    config.expected_improvement = "+10-15% (tools provide better context)"
    return config


def harness_v3_prompt_tuned() -> HarnessConfig:
    """v3: Refine LLM prompt based on v2 learnings."""
    from datetime import datetime

    config = harness_v2_tool_aware()
    config.version = "v3_prompt_tuned"
    config.epoch = 3
    config.created = datetime.now().isoformat()
    config.prompt_template = "tool_aware_tuned_v2_learnings"
    config.prompt_context = {
        "emphasis": ["short windows in crisis", "momentum in bull markets"],
        "avoid": ["parameter combinations that underperformed"],
    }
    config.description = "Optimized prompt based on v2 learnings"
    config.reasoning = "Use v2 results to refine LLM prompt; emphasize what worked"
    config.expected_improvement = "+3-5% (refined guidance to Claude)"
    return config


def harness_v4_grid_evolved() -> HarnessConfig:
    """v4: Adapt parameter grid toward high-performing regions."""
    from datetime import datetime

    config = harness_v3_prompt_tuned()
    config.version = "v4_grid_evolved"
    config.epoch = 4
    config.created = datetime.now().isoformat()
    config.grid_adaptation_strategy = "shrink_to_winners"
    config.grid_focus_regions = [
        {"description": "High-Sharpe regions", "samples_per_region": 5},
        {"description": "Stable combinations", "samples_per_region": 3},
    ]
    config.description = "Parameter grid evolved toward high-performers"
    config.reasoning = "Focus search on discovered high-performing regions"
    config.expected_improvement = "+5-8% (concentrated search)"
    return config


def harness_v5_multi_agent() -> HarnessConfig:
    """v5: Multi-agent ensemble."""
    from datetime import datetime

    config = harness_v4_grid_evolved()
    config.version = "v5_multi_agent"
    config.epoch = 5
    config.created = datetime.now().isoformat()
    config.use_ensemble = True
    config.tool_weight = 0.4  # Balance tool vs grid vs random
    config.prompt_context = {
        "ensemble_strategies": [
            "tool_based_proposals",
            "grid_search_winners",
            "random_exploration",
        ],
        "voting_method": "sharpe_weighted",
    }
    config.description = "Multi-agent ensemble (tools + grid + random)"
    config.reasoning = "Combine multiple proposal strategies; ensemble voting"
    config.expected_improvement = "+2-4% (ensemble diversity)"
    return config


def harness_v6_research() -> HarnessConfig:
    """v6: Research-informed proposals."""
    from datetime import datetime

    config = harness_v5_multi_agent()
    config.version = "v6_research"
    config.epoch = 6
    config.created = datetime.now().isoformat()
    config.prompt_template = "research_informed"
    config.prompt_context = {
        "research_sources": ["academic_papers", "industry_research", "strategy_blogs"],
        "hypothesis_generation": "enabled",
        "citation_tracking": "enabled",
    }
    config.description = "Research agent discovers ideas; tools validate"
    config.reasoning = "Research agent searches literature; validates via tools"
    config.expected_improvement = "+3-7% (novel research-backed ideas)"
    return config


@dataclass
class EffectiveHarnessConfig:
    """
    The runtime configuration actually applied to an agent run, derived from
    a requested HarnessConfig.

    This is what proposal generation, tool admission, prompts, and the
    stopping policy read from -- not the raw HarnessConfig -- so "requested"
    and "effective" settings can be reported side by side and any knob the
    runtime does not actually wire through is caught at resolve time instead
    of being silently ignored.
    """

    use_tools: bool
    use_web_search: bool
    prompt_template: str
    prompt_context: Dict[str, Any]
    min_acceptable_sharpe: float
    max_acceptable_drawdown: float
    max_iterations: Optional[int]
    config_hash: str
    requested: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_tools": self.use_tools,
            "use_web_search": self.use_web_search,
            "prompt_template": self.prompt_template,
            "prompt_context": self.prompt_context,
            "min_acceptable_sharpe": self.min_acceptable_sharpe,
            "max_acceptable_drawdown": self.max_acceptable_drawdown,
            "max_iterations": self.max_iterations,
            "config_hash": self.config_hash,
            "requested": self.requested,
        }


# Knobs on HarnessConfig that the runtime currently has no wiring for at all.
# If a caller sets one of these away from its dataclass default, resolution
# must fail loudly rather than pretend the knob had an effect.
_UNSUPPORTED_IF_NON_DEFAULT = {
    "use_ensemble": False,
    "grid_adaptation_strategy": None,
    "grid_focus_regions": [],
    "claim_weighting": 0.5,
    "tool_weight": 0.5,
    "required_win_rate": 0.0,
}


def resolve_effective_config(
    harness_spec: "HarnessConfig",
    *,
    max_iterations_override: Optional[int] = None,
) -> EffectiveHarnessConfig:
    """
    Turn a requested HarnessConfig into the EffectiveHarnessConfig the agent
    graph actually reads. Raises UnsupportedHarnessKnobError for any knob the
    runtime doesn't implement rather than silently dropping it.
    """
    for field_name, default in _UNSUPPORTED_IF_NON_DEFAULT.items():
        value = getattr(harness_spec, field_name, default)
        if value != default:
            raise UnsupportedHarnessKnobError(
                f"harness_config.{field_name}={value!r} has no effect in this runtime "
                f"(supported knobs: use_tools, use_web_search, prompt_template, "
                f"prompt_context, min_acceptable_sharpe, max_acceptable_drawdown, "
                f"max_retries-as-max_iterations). Refusing to silently ignore it."
            )

    requested = harness_spec.to_dict()
    config_hash = hashlib.sha256(
        json.dumps(requested, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    return EffectiveHarnessConfig(
        use_tools=harness_spec.use_tools,
        use_web_search=harness_spec.use_web_search,
        prompt_template=harness_spec.prompt_template,
        prompt_context=dict(harness_spec.prompt_context or {}),
        min_acceptable_sharpe=harness_spec.min_acceptable_sharpe,
        max_acceptable_drawdown=harness_spec.max_acceptable_drawdown,
        max_iterations=max_iterations_override,
        config_hash=config_hash,
        requested=requested,
    )


def get_harness_sequence() -> List[HarnessConfig]:
    """Get the full harness evolution sequence."""
    return [
        harness_v1_base(),
        harness_v2_tool_aware(),
        harness_v3_prompt_tuned(),
        harness_v4_grid_evolved(),
        harness_v5_multi_agent(),
        harness_v6_research(),
    ]
