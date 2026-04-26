"""Tests for proposal generator fallback chain."""

import pytest

from src.agent.context_builder import RegimeContext
from src.agent.proposal_generator import Proposal, ProposalGenerator


def test_fallback_without_api_key(monkeypatch):
    """Without an API key, generator should use grid search."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    gen = ProposalGenerator()
    ctx = RegimeContext(regime_label="LowVol-Bull")
    proposals = gen.generate(ctx, n_proposals=3, strategy_type="momentum")
    assert len(proposals) > 0
    assert all(isinstance(p, Proposal) for p in proposals)
    methods = {p.generation_method for p in proposals}
    assert "llm" not in methods  # no LLM key → grid_search or random


def test_proposals_have_valid_params(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    gen = ProposalGenerator()
    ctx = RegimeContext(regime_label="Crisis-Bear")
    proposals = gen.generate(ctx, n_proposals=5, strategy_type="momentum")
    for p in proposals:
        assert "fast_window" in p.params
        assert "slow_window" in p.params
        assert p.params["fast_window"] < p.params["slow_window"]


def test_proposals_count(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    gen = ProposalGenerator()
    ctx = RegimeContext(regime_label="MidVol-Neutral")
    proposals = gen.generate(ctx, n_proposals=4, strategy_type="momentum")
    assert len(proposals) == 4
