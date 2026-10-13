import hashlib
from pathlib import Path
import pytest
from egoalign.runtime_common import port_set, verify_runtime, clean_environment


def test_private_ports():
    assert port_set(28700, 2) == list(range(28700,28705)) + list(range(28800,28805))
    for base, workers in [(5550,1),(65534,1),(28700,0),(28700,17)]:
        with pytest.raises(ValueError):
            port_set(base, workers)


def test_pinned_runtime(tmp_path):
    (tmp_path / "file").write_bytes(b"exact")
    manifest = {"files": {"file": hashlib.sha256(b"exact").hexdigest()}}
    verify_runtime(tmp_path, manifest)
    (tmp_path / "file").write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible hash"):
        verify_runtime(tmp_path, manifest)
    with pytest.raises(ValueError, match="manifest path"):
        verify_runtime(tmp_path, {"files": {"../escape": "bad"}})


def test_environment_isolation():
    result = clean_environment(Path("/runtime"), {
        "SONIC_REPLAY_UPPER_BODY_DATASET": "danger", "ROBOT_IP": "hardware",
        "SIM_BRIDGE_MODE": "ros2", "TMUX": "parent", "PATH": "/bin"})
    assert result["SIM_BRIDGE_MODE"] == "mock"
    assert result["PATH"] == "/bin"
    assert "SONIC_REPLAY_UPPER_BODY_DATASET" not in result
    assert "ROBOT_IP" not in result and "TMUX" not in result
