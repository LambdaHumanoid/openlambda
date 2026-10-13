"""Explicit paths: no developer-machine fallback, no implicit robot connection."""
import os
from pathlib import Path
import shutil


def validate_replay_workers(workers):
    """Require explicit capacity for a custom driver; bundled replay is single-lane."""
    driver = os.environ.get("EGOALIGN_REPLAY_DRIVER", "")
    bundled = Path(__file__).resolve().parents[1] / "tools/public_replay_driver.sh"
    resolved = Path(shutil.which(driver) or driver).expanduser().resolve() if driver else bundled
    if resolved == bundled.resolve():
        maximum = 1
    else:
        try:
            maximum = int(os.environ.get("EGOALIGN_REPLAY_MAX_WORKERS", "1"))
        except ValueError as exc:
            raise ValueError("EGOALIGN_REPLAY_MAX_WORKERS must be a positive integer") from exc
        if maximum < 1:
            raise ValueError("EGOALIGN_REPLAY_MAX_WORKERS must be a positive integer")
    if workers < 1 or workers > maximum:
        raise ValueError(
            f"Replay driver supports 1..{maximum} workers, requested {workers}. "
            "Use --workers 1 with the bundled driver; only a compatible custom "
            "driver may declare EGOALIGN_REPLAY_MAX_WORKERS."
        )

def asset_path(relative):
    root = os.environ.get("EGOALIGN_ASSET_ROOT")
    root = root or str(Path(__file__).resolve().parents[1] / "assets")
    return Path(root).expanduser().resolve() / relative

def human_skeleton_path():
    """External SONIC skeleton; never depend on the caller's working directory."""
    override = os.environ.get("EGOALIGN_HUMAN_JOINTS_PATH")
    path = (Path(override).expanduser().resolve() if override else
            asset_path("gear_sonic/data/human/human_joints_info.pkl"))
    if not path.is_file():
        raise FileNotFoundError(
            f"Human skeleton asset missing: {path}. Set EGOALIGN_HUMAN_JOINTS_PATH "
            "to the trusted local file, or configure EGOALIGN_ASSET_ROOT. "
            "See docs/public_runtime.md."
        )
    return path
