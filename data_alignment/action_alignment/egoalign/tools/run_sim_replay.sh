#!/usr/bin/env bash
# Simulation replay adapter; forwards pipeline arguments to the driver.
set -euo pipefail
if [[ -z "${EGOALIGN_REPLAY_DRIVER:-}" ]]; then
  echo "EGOALIGN_REPLAY_DRIVER is required: configure a compatible isolated SONIC/MuJoCo batch replay driver." >&2
  exit 2
fi
if [[ "${EGOALIGN_ALLOW_SIM_REPLAY:-}" != "1" ]]; then
  echo "Set EGOALIGN_ALLOW_SIM_REPLAY=1 after verifying the driver's simulation-only port isolation." >&2
  exit 2
fi
exec "$EGOALIGN_REPLAY_DRIVER" "$@"
