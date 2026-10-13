"""Check all public help entrypoints without launching a runtime."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
failed = []
for command in ('endpoint-align', 'replay-only', 'candidate', 'evaluate', 'audit', 'calibrate', 'promote', 'stage-a', 'stage-c', 'render-replay'):
    result = subprocess.run([sys.executable, '-m', 'egoalign', command, '--help'],
                            cwd=root, text=True, capture_output=True)
    print(command, 'PASS' if result.returncode == 0 else 'FAIL')
    if result.returncode:
        print(result.stderr[-2000:])
        failed.append(command)
raise SystemExit(bool(failed))
