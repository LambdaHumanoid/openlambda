#!/usr/bin/env bash
# Install the model package only; collection uses its own installers/environments.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: PYTHON=python3.11 bash scripts/install.sh [serve,tracking,fast]"
    echo "Creates/reuses .venv at the repository root; does not activate it."
    exit 0
fi
fail() { echo "ERROR: $*" >&2; exit 1; }
(( $# <= 1 )) || fail "Expected at most one comma-separated extras argument."
extras="${1:-}"
[[ -z "$extras" || "$extras" =~ ^(serve|tracking|fast)(,(serve|tracking|fast))*$ ]] ||
    fail "Supported extras: serve,tracking,fast."
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$project_root/.venv"
if [[ ! -e "$venv" ]]; then
    python="${PYTHON:-python3.11}"
    command -v "$python" >/dev/null || fail "Set PYTHON to a Python 3.11 or 3.12 interpreter."
    "$python" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Python 3.11 or 3.12 required"'
    "$python" -m venv "$venv"
fi
python="$venv/bin/python"
[[ -x "$python" ]] || fail "Existing $venv is not a usable environment; left unchanged."
"$python" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Existing environment must use Python 3.11 or 3.12"'
target="$project_root"
if [[ -n "$extras" ]]; then target="$target[$extras]"; fi
"$python" -m pip install -e "$target"
"$python" -m pip check
"$python" -m lambda0 doctor
echo "Installed. Activate with: source \"$venv/bin/activate\""
