"""Tests for scripts/measure_cold_install.py's reporting logic. Does not
actually build venvs / hit the network -- subprocess.run is mocked so this
stays fast and offline like the rest of the suite."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import measure_cold_install  # noqa: E402


def _fake_completed(returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_measure_one_reports_error_without_raising_on_failed_install(tmp_path):
    with patch.object(measure_cold_install, "subprocess") as mock_sp:
        mock_sp.run.side_effect = [
            _fake_completed(),  # venv creation
            _fake_completed(),  # pip upgrade
            _fake_completed(returncode=1, stderr="ERROR: no matching distribution"),  # install fails
        ]
        result = measure_cold_install._measure_one("[dev]", "dev")

    assert "error" in result
    assert "no matching distribution" in result["error"]
    assert result["extras"] == "[dev]"


def test_measure_one_uses_none_label_for_empty_extras():
    with patch.object(measure_cold_install, "subprocess") as mock_sp:
        mock_sp.run.side_effect = [
            _fake_completed(),
            _fake_completed(),
            _fake_completed(returncode=1, stderr="boom"),
        ]
        result = measure_cold_install._measure_one("", "base")
    assert result["extras"] == "(none)"


def test_dir_size_bytes_sums_only_files(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_bytes(b"y" * 50)
    assert measure_cold_install._dir_size_bytes(tmp_path) == 150
