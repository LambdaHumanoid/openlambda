"""Public help must match the supported processing modes."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("command", ["candidate", "evaluate", "endpoint-align"])
def test_alignment_help_omits_removed_experiments(command):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, EGOALIGN_DATA_PYTHON=sys.executable,
               EGOALIGN_SIM_PYTHON=sys.executable)
    result = subprocess.run([sys.executable, "-m", "egoalign", command, "--help"],
                            cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for removed in ("--experimental-", "--feet-initial-", "--gate-foot-",
                    "--force-all-rounds", "--foot-"):
        assert removed not in result.stdout
    if command != "evaluate":
        assert "--endpoint-parts {hands}" in result.stdout
        assert "--hand-endpoint-side {both}" in result.stdout
