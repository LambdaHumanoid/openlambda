from pathlib import Path
from egoalign.sample_pipeline import plan
import json
from types import SimpleNamespace


def test_basket_creates_required_execution_state(tmp_path, monkeypatch):
    from egoalign import sample_pipeline
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "output"
    monkeypatch.setenv("EGOALIGN_ALLOW_SIM_REPLAY", "1")
    monkeypatch.setattr("sys.argv", ["sample_pipeline", "basket", "--raw-root", str(raw),
                                    "--output-root", str(output), "--robot-xml", "/model.xml"])
    def run(*args, **kwargs):
        assert json.loads((output / "state.json").read_text())["status"] == "running"
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(sample_pipeline.subprocess, "run", run)
    assert sample_pipeline.main() == 0


def test_basket_pipeline():
    commands = plan("basket", Path("/raw"), Path("/output"), Path("/model.xml"), "/python")
    assert len(commands) == 5
    assert commands[0][3] == "stage-a" and commands[1][3] == "calibrate"
    assert commands[3][3] == "endpoint-align" and commands[4][3] == "promote"
    assert "--force-all-rounds" not in commands[3]
    assert "--gate-budget-limit-improvement-only" not in commands[3]


def test_unknown_sample_rejected():
    import pytest
    with pytest.raises(ValueError, match="public sample"):
        plan("pedal", Path("/raw"), Path("/output"), None, "/python")
