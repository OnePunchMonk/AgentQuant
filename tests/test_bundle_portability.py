"""Replay preflight must fail before executing research on missing or changed data."""

from argparse import Namespace

import pytest

from scripts import export_research_memo as cli
from src.agent.episode_bundle import EpisodeBundle


def _bundle(tmp_path, *, source="synthetic", data_hash="abc", splits_hash=None):
    split = tmp_path / "split.json"
    split.write_text("{}")
    bundle = EpisodeBundle(
        run_id="preflight",
        episode_result={},
        run_manifest={},
        dataset_identity={"data_hash": data_hash, "source": source,
                          "splits_path": str(split), "splits_sha256": splits_hash},
        software_identity={},
        rerun_command=("python scripts/export_research_memo.py --episodes 4 --seeds 7 "
                       "--n-mutations 1 --cost-bps 5 --max-iterations 1"
                       + (f" --user-data {source} --user-data-ticker DEMO" if source != "synthetic" else "")),
    )
    return bundle.save(tmp_path)


def _args(path, tmp_path):
    return Namespace(fresh_replay=str(path), output=str(tmp_path / "memo"),
                     holdout_guard_path=str(tmp_path / "guard.json"))


def test_missing_private_data_never_executes_research(tmp_path, monkeypatch):
    path = _bundle(tmp_path, source=str(tmp_path / "private.csv"))
    monkeypatch.setattr(cli, "_run_fresh", lambda args: pytest.fail("research executed"))
    with pytest.raises(SystemExit, match="Private OHLCV data required"):
        cli._fresh_replay(_args(path, tmp_path))


def test_changed_split_never_executes_research(tmp_path, monkeypatch):
    path = _bundle(tmp_path, splits_hash="0" * 64)
    monkeypatch.setattr(cli, "_run_fresh", lambda args: pytest.fail("research executed"))
    with pytest.raises(SystemExit, match="Episode split identity mismatch"):
        cli._fresh_replay(_args(path, tmp_path))


def test_saved_command_is_not_shell_executed(tmp_path, monkeypatch):
    path = _bundle(tmp_path)
    bundle = EpisodeBundle.load(path)
    bundle.rerun_command += " ; touch /tmp/should-never-run"
    bundle.save(tmp_path)
    monkeypatch.setattr(cli, "_run_fresh", lambda args: pytest.fail("research executed"))
    with pytest.raises(SystemExit, match="unsupported arguments"):
        cli._fresh_replay(_args(path, tmp_path))
