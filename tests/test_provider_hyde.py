import json
from types import SimpleNamespace

from src.agent.generators.provider_hyde import ProviderHydeGenerator
from src.agent.hyde_retrieval import HypothesisRequest


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.call = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def request(**kwargs):
    return HypothesisRequest("high volatility", ("momentum", "mean_reversion"), "2024-01-01",
                             max_hypotheses=kwargs.get("max_hypotheses", 3))


def test_provider_hyde_bounds_call_and_keeps_only_canonical_output():
    content = json.dumps([
        {"description": "  A bounded momentum hypothesis.  ", "strategy_type": "momentum",
         "proposed_params": {"fast_window": 5, "slow_window": 20}},
        {"description": "unknown", "strategy_type": "invented", "proposed_params": {}},
        {"description": "off-grid", "strategy_type": "momentum",
         "proposed_params": {"fast_window": 1, "slow_window": 2}},
    ])
    client = FakeClient(content)
    generator = ProviderHydeGenerator(client, model="fixture-model", temperature=0.35)

    result = generator.generate(request())

    assert len(result) == 1
    assert result[0].description == "A bounded momentum hypothesis."
    assert result[0].strategy_type == "momentum"
    assert result[0].proposed_params == {"fast_window": 5, "slow_window": 20}
    assert client.call["max_tokens"] <= 512
    assert client.call["model"] == "fixture-model"
    assert client.call["temperature"] == 0.35
    assert generator.metadata["model"] == "fixture-model"
    assert generator.metadata["temperature"] == 0.35


def test_provider_hyde_rejects_malformed_or_overlong_response():
    assert ProviderHydeGenerator(FakeClient("not json")).generate(request()) == []
    assert ProviderHydeGenerator(FakeClient("x" * 12_001)).generate(request()) == []


def test_provider_hyde_zero_limit_skips_provider_call():
    client = FakeClient("[]")
    assert ProviderHydeGenerator(client).generate(request(max_hypotheses=0)) == []
    assert client.call is None
