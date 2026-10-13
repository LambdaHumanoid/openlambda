"""Exercise the replay adapter without starting a simulation or opening ports."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from egoalign import replay_only, sample_pipeline
from scripts.human_data import run_endpoint_retarget_dataset as endpoint

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "tools/run_sim_replay.sh"


def test_entry_exists_and_is_executable():
    assert ENTRY.is_file() and os.access(ENTRY, os.X_OK)
    assert not (ROOT / "run.sh").exists()


@pytest.mark.parametrize("driver,allow,missing", [
    (None, "1", "EGOALIGN_REPLAY_DRIVER"),
    (sys.executable, None, "EGOALIGN_ALLOW_SIM_REPLAY"),
    (sys.executable, "0", "EGOALIGN_ALLOW_SIM_REPLAY"),
])
def test_entry_requires_explicit_configuration(driver, allow, missing):
    env = {key: value for key, value in os.environ.items()
           if key not in {"EGOALIGN_REPLAY_DRIVER", "EGOALIGN_ALLOW_SIM_REPLAY"}}
    if driver is not None:
        env["EGOALIGN_REPLAY_DRIVER"] = driver
    if allow is not None:
        env["EGOALIGN_ALLOW_SIM_REPLAY"] = allow
    result = subprocess.run([str(ENTRY)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert missing in result.stderr


def test_entry_forwards_arguments_and_exit_code(tmp_path):
    env = dict(os.environ, EGOALIGN_REPLAY_DRIVER=sys.executable,
               EGOALIGN_ALLOW_SIM_REPLAY="1")
    arguments = ["path with spaces", "--flag=value", "$literal", ""]
    result = subprocess.run([
        str(ENTRY), "-c",
        "import json, sys; print(json.dumps(sys.argv[1:])); sys.exit(7)",
        *arguments,
    ], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 7
    assert json.loads(result.stdout) == arguments


def test_sample_and_replay_only_use_named_entry():
    plan = sample_pipeline.plan("basket", Path("/raw"), Path("/output"),
                                Path("/model.xml"), "/python")
    assert plan[2][0] == str(ENTRY)
    args = SimpleNamespace(source_root=Path("/stage_a"), replay_root=Path("/replay"),
                           output_root=Path("/final"), workers=1, reuse_replay=False)
    assert replay_only.commands(args)[0][0] == str(ENTRY)


@pytest.mark.parametrize("workers", [1, 2])
def test_endpoint_rounds_use_named_entry(monkeypatch, workers):
    monkeypatch.setenv("EGOALIGN_REPLAY_DRIVER", "/custom/parallel-driver")
    monkeypatch.setenv("EGOALIGN_REPLAY_MAX_WORKERS", "4")
    calls = []
    monkeypatch.setattr(endpoint, "run_checked",
                        lambda command, **kwargs: calls.append((command, kwargs)))
    if workers == 1:
        endpoint.run_replay(Path("/stage_a"), Path("/replay"))
    else:
        endpoint.run_parallel_replay(Path("/stage_a"), Path("/replay"), workers)
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == str(ENTRY)
    assert command[command.index("--replay-workers") + 1] == str(workers)
    assert kwargs["env"]["SONIC_REPLAY_WRITE_VIDEO"] == "false"
