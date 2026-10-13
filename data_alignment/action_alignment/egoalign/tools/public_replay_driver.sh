#!/usr/bin/env bash
set -euo pipefail
package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${package_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${EGOALIGN_DATA_PYTHON:-python3}" -m egoalign.public_runtime "$@"
