"""Where AgentQuant writes run state (results db, manifests, bundles, ledgers, holdout guards).

One directory, `.agentquant/` in the working directory by default. Set AGENTQUANT_RUNS_DIR before
import to put it elsewhere. Nothing here is checked in.
"""
import os
from pathlib import Path

RUNS_DIR = Path(os.environ.get("AGENTQUANT_RUNS_DIR", ".agentquant"))
