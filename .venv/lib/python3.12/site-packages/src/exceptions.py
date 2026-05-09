"""
Custom Exceptions for AgentQuant
==================================

Defines domain-specific exception types for clear error semantics
and programmatic error handling throughout the platform.
"""


class AgentQuantError(Exception):
    """Base exception for all AgentQuant errors."""


# --- Data Layer ---

class DataNotFoundError(AgentQuantError):
    """Raised when requested market data is not available."""


class CacheError(AgentQuantError):
    """Raised when reading from or writing to the data cache fails."""


class FREDApiError(AgentQuantError):
    """Raised when the FRED API request fails."""


# --- Feature Engineering ---

class InsufficientDataError(AgentQuantError):
    """Raised when there is not enough data to compute a feature."""


class InsufficientWarmupError(AgentQuantError):
    """Raised when a backtest requests signals before the minimum warmup period."""


class StationarityWarning(UserWarning):
    """Emitted when a feature series appears non-stationary."""


# --- Strategy ---

class StrategyNotFoundError(AgentQuantError):
    """Raised when a requested strategy name is not in the registry."""

    def __init__(self, name: str, available: list = None):
        available_str = ", ".join(available) if available else "unknown"
        super().__init__(
            f"Strategy '{name}' not found in registry. Available: {available_str}"
        )


class StrategyValidationError(AgentQuantError):
    """Raised when strategy parameters fail validation."""


# --- Backtest ---

class BacktestFailedError(AgentQuantError):
    """Raised when the backtest engine fails to produce a result."""


class SignalGenerationError(AgentQuantError):
    """Raised when a strategy fails to generate valid signals."""


# --- Agent / Planner ---

class ProposalValidationError(AgentQuantError):
    """Raised when an LLM proposal fails parameter validation."""


class LLMUnavailableError(AgentQuantError):
    """Raised when no LLM planner is available (no valid API key)."""


class MaxIterationsExceededError(AgentQuantError):
    """Raised when the agent exceeds max_iterations without finding an acceptable result."""


# --- Config ---

class ConfigValidationError(AgentQuantError):
    """Raised when configuration fails Pydantic validation."""
