"""Run the model package tests without changing the caller's working directory."""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    tests = Path(__file__).resolve().parents[1] / "tests"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(tests), *sys.argv[1:]])
    raise SystemExit(result.returncode)
