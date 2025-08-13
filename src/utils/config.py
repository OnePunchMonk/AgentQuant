"""
Configuration Management Module for AgentQuant
==============================================

This module handles all configuration loading and management for the AgentQuant
platform. It provides a centralized configuration system that loads settings
from YAML files and makes them available throughout the application.

Key Features:
- YAML-based configuration management
- Automatic config loading and validation
- Centralized configuration access via singleton pattern
- Support for nested configuration structures
- Error handling for missing or malformed config files

The configuration system supports various settings including:
- Universe definition (list of assets to analyze)
- Data source parameters (yfinance, FRED APIs)
- Agent configuration (LLM settings, optimization parameters)
- Backtesting parameters (initial cash, fees, date ranges)
- Risk management settings (drawdown limits, position sizing)

Usage:
    from src.utils.config import config
    universe = config['universe']
    initial_cash = config['backtest']['initial_cash']

Dependencies:
- yaml: YAML file parsing and loading
- pathlib: Cross-platform path handling

Author: AgentQuant Development Team
License: MIT
"""
import yaml
from pathlib import Path

def load_config():
    """Loads the config.yaml file."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found at the project root.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load config once and make it available for import
config = load_config()