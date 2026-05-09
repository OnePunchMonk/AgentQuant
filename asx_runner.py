"""
ASX Mining Runner
================

Run the AgentQuant loop specifically for the ASX mining sector.
"""

import logging
import os
from dotenv import load_dotenv
from pathlib import Path

# Override config path before importing other src modules
os.environ["AGENTQUANT_CONFIG"] = str(Path(__file__).parent / "asx_config.yaml")

import src.utils.config
from src.utils.config import AppConfig, load_config

# Re-load config with the ASX specific one
asx_config_path = Path(__file__).parent / "asx_config.yaml"
src.utils.config.config = load_config(asx_config_path)

from src.agent.runner import main as run_main
from src.data.asx_ingest import fetch_asx_mining_data

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Ensure ASX data is fetched/cached
    logger.info("Fetching ASX Mining and Commodity data...")
    fetch_asx_mining_data()
    
    # Run the standard agent loop with ASX configuration
    run_main()
