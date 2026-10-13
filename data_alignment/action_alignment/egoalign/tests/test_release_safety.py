"""Negative release cases: no physics, GPU jobs or real dataset deletion."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egoalign.io_safety import validate_output_root
from egoalign import runtime, replay_only, sample_pipeline, __main__ as cli
from egoalign.verify_sample import verify
from scripts.human_data.processing import apply_sonic_mujoco_feedback as stage_c


@pytest.mark.parametrize("kind", ["same", "ancestor", "descendant", "symlink", "home", "root", "cwd"])
def test_output_guard_rejects_unsafe_paths(tmp_path, kind):
    raw = tmp_path / "raw"
    raw.mkdir()
    link = tmp_path / "link"
    link.symlink_to(raw, target_is_directory=True)
    target = {"same": raw, "ancestor": tmp_path, "descendant": raw / "output",
              "symlink": link, "home": Path.home(), "root": Path("/"), "cwd": Path.cwd()}[kind]
    with pytest.raises(ValueError):
        validate_output_root(target, [raw])


def test_output_guard_accepts_disjoint_sibling(tmp_path):
    assert validate_output_root(tmp_path / "out", [tmp_path / "raw"]) == tmp_path / "out"


@pytest.mark.parametrize("kind", ["same", "ancestor", "descendant", "symlink"])
@pytest.mark.parametrize("entry", ["stage_a", "stage_c"])
def test_direct_entry_rejects_overlap_before_deletion(tmp_path, monkeypatch, entry, kind):
    # Stage A imports its documented optional data environment; skip only that
    # entry when running the lightweight test dependency set.
    if entry == "stage_a":
        pytest.importorskip("lerobot.common.datasets.lerobot_dataset")
        from scripts.human_data.processing import convert_raw_pico_lerobot_to_sonic_gr00t as module
    else:
        module = stage_c
    raw = tmp_path / "raw"
    raw.mkdir()
    link = tmp_path / "link"
    link.symlink_to(raw, target_is_directory=True)
    target = {"same": raw, "ancestor": tmp_path, "descendant": raw / "out", "symlink": link}[kind]
    monkeypatch.setattr(module.shutil, "rmtree", lambda *a, **k: pytest.fail("Deletion must not be reached"))
    args = SimpleNamespace(source_root=raw, output_root=target, ignore_alignment=False,
                           require_alignment=True, overwrite=True, telemetry_root=tmp_path / "telemetry")
    with pytest.raises(ValueError, match="disjoint"):
        if entry == "stage_a":
            monkeypatch.setattr(module, "parse_args", lambda: args)
            module.main()
        else:
            module.process_dataset(args)


def test_stage_c_rejects_output_containing_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_c.shutil, "rmtree", lambda *a, **k: pytest.fail("Unsafe deletion"))
    args = SimpleNamespace(source_root=tmp_path / "raw", output_root=tmp_path / "out",
                           telemetry_root=tmp_path / "out/telemetry", overwrite=True)
    with pytest.raises(ValueError, match="disjoint"):
        stage_c.process_dataset(args)


@pytest.mark.parametrize("discovery", ["empty", "invalid"])
def test_stage_a_bad_input_preserves_existing_output(tmp_path, monkeypatch, discovery):
    pytest.importorskip("lerobot.common.datasets.lerobot_dataset")
    from scripts.human_data.processing import convert_raw_pico_lerobot_to_sonic_gr00t as module
    raw, output = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing result")
    args = SimpleNamespace(source_root=raw, output_root=output, ignore_alignment=True,
                           require_alignment=False, overwrite=True, episodes="all")
    monkeypatch.setattr(module, "parse_args", lambda: args)
    def discover(*args, **kwargs):
        if discovery == "invalid":
            raise ValueError("Invalid source")
        return []
    monkeypatch.setattr(module, "discover_episodes", discover)
    monkeypatch.setattr(module.shutil, "rmtree", lambda *a, **k: pytest.fail("Unsafe deletion"))
    with pytest.raises(ValueError):
        module.main()
    assert marker.read_text() == "existing result"


def camera_fixture(tmp_path, frames=1):
    stage, final = tmp_path / "stage", tmp_path / "final"
    keys = ["observation.images.ego_view", "observation.images.level_cam"]
    state = stage_c.physical_qpos_to_isaaclab_offset_state(np.zeros((frames, 43)))
    columns = {"observation.state": state.tolist(), "action.motion_token": np.zeros((frames, 64)).tolist(),
               "episode_index": [0] * frames, "timestamp": (np.arange(frames) / 50).tolist(),
               **{key: [f"videos/{key}.mp4"] * frames for key in keys}}
    info = {"features": {key: {"dtype": "video"} for key in keys}, "chunks_size": 1000,
            "video_path": "videos/{video_key}.mp4"}
    for root in (stage, final):
        for name in ("data", "videos", "meta"):
            (root / name).mkdir(parents=True)
        pq.write_table(pa.table(columns), root / "data/episode_000000.parquet")
        (root / "meta/info.json").write_text(json.dumps(info))
        for key in keys:
            (root / f"videos/{key}.mp4").write_bytes(b"fixture, not encoded video")
    control = tmp_path / "control.npz"
    np.savez(control, body_q=np.zeros((frames, 29)), left_hand_q=np.zeros((frames, 7)),
             right_hand_q=np.zeros((frames, 7)), token_state=np.zeros((frames, 64)))
    return stage, final, control, keys


def test_complete_camera_fixture_passes(tmp_path):
    stage, final, control, _ = camera_fixture(tmp_path)
    assert len(verify(stage, final, control)["camera_sha256"]) == 2


def test_multi_frame_capture_passes(tmp_path):
    stage, final, control, _ = camera_fixture(tmp_path, frames=4)
    assert verify(stage, final, control)["frames"] == 4


@pytest.mark.parametrize("field", ["body_q", "left_hand_q", "right_hand_q", "token_state"])
@pytest.mark.parametrize("damage", ["one_row", "one_column", "flat", "extra_row", "empty",
                                    "missing", "nan", "inf", "string", "complex", "bool", "object"])
def test_invalid_capture_payload_rejected(tmp_path, field, damage):
    stage, final, control, _ = camera_fixture(tmp_path, frames=4)
    with np.load(control, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    values = payload[field]
    if damage == "missing":
        del payload[field]
    else:
        if damage == "one_row":
            values = values[:1]
        elif damage == "one_column":
            values = values[:, :1]
        elif damage == "flat":
            values = values[0]
        elif damage == "extra_row":
            values = np.concatenate([values, values[:1]])
        elif damage == "empty":
            values = values[:0]
        elif damage in {"nan", "inf"}:
            values[0, 0] = float(damage)
        else:
            values = values.astype({"string": str, "complex": complex,
                                    "bool": bool, "object": object}[damage])
        payload[field] = values
    np.savez(control, **payload)
    with pytest.raises(ValueError):
        verify(stage, final, control)


@pytest.mark.parametrize("damage", ["missing", "extra", "both_missing", "metadata", "reference",
                                   "swapped_reference", "duplicate_mapping", "bytes"])
def test_camera_corruption_fails(tmp_path, damage):
    stage, final, control, keys = camera_fixture(tmp_path)
    if damage in {"missing", "both_missing"}:
        (final / f"videos/{keys[1]}.mp4").unlink()
        if damage == "both_missing":
            (stage / f"videos/{keys[1]}.mp4").unlink()
    elif damage == "extra":
        (final / "videos/unexpected.mp4").write_bytes(b"extra")
    elif damage == "metadata":
        info = json.loads((final / "meta/info.json").read_text())
        del info["features"][keys[1]]
        (final / "meta/info.json").write_text(json.dumps(info))
    elif damage == "bytes":
        (final / f"videos/{keys[1]}.mp4").write_bytes(b"changed")
    elif damage == "duplicate_mapping":
        for root in (stage, final):
            info = json.loads((root / "meta/info.json").read_text())
            info["video_path"] = f"videos/{keys[0]}.mp4"
            (root / "meta/info.json").write_text(json.dumps(info))
    else:
        for root in (stage, final):
            path = root / "data/episode_000000.parquet"
            data = pq.read_table(path).to_pydict()
            data[keys[1]] = [f"videos/{keys[0]}.mp4" if damage == "swapped_reference" else "videos/missing.mp4"]
            pq.write_table(pa.table(data), path)
    with pytest.raises(ValueError, match="[Cc]amera"):
        verify(stage, final, control)


@pytest.mark.parametrize("driver", ["unset", "bundled", "symlink", "unknown_custom"])
def test_worker_defaults_fail_closed(tmp_path, monkeypatch, driver):
    bundled = Path(runtime.__file__).resolve().parents[1] / "tools/public_replay_driver.sh"
    monkeypatch.delenv("EGOALIGN_REPLAY_DRIVER", raising=False)
    monkeypatch.delenv("EGOALIGN_REPLAY_MAX_WORKERS", raising=False)
    if driver == "bundled":
        monkeypatch.setenv("EGOALIGN_REPLAY_DRIVER", str(bundled))
        monkeypatch.setenv("EGOALIGN_REPLAY_MAX_WORKERS", "4")
    elif driver == "symlink":
        link = tmp_path / "driver"
        link.symlink_to(bundled)
        monkeypatch.setenv("EGOALIGN_REPLAY_DRIVER", str(link))
        monkeypatch.setenv("EGOALIGN_REPLAY_MAX_WORKERS", "4")
    elif driver == "unknown_custom":
        monkeypatch.setenv("EGOALIGN_REPLAY_DRIVER", "/custom/driver")
    runtime.validate_replay_workers(1)
    with pytest.raises(ValueError, match="supports 1..1"):
        runtime.validate_replay_workers(2)


def test_custom_driver_capacity(monkeypatch):
    monkeypatch.setenv("EGOALIGN_REPLAY_DRIVER", "/custom/driver")
    monkeypatch.setenv("EGOALIGN_REPLAY_MAX_WORKERS", "4")
    runtime.validate_replay_workers(4)
    with pytest.raises(ValueError):
        runtime.validate_replay_workers(5)


def test_replay_only_rejects_workers_before_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("EGOALIGN_REPLAY_DRIVER", raising=False)
    with pytest.raises(SystemExit) as error:
        replay_only.main(["--source-root", str(tmp_path / "source"), "--replay-root", str(tmp_path / "replay"),
                          "--output-root", str(tmp_path / "output"), "--workers", "2", "--dry-run"])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_endpoint_rejects_workers_before_reading_inputs(tmp_path, monkeypatch):
    from scripts.human_data import run_endpoint_retarget_dataset as endpoint
    monkeypatch.delenv("EGOALIGN_REPLAY_DRIVER", raising=False)
    monkeypatch.setattr("sys.argv", ["endpoint", "--raw-dataset", "/missing/raw", "--baseline-dataset",
                                    "/missing/base", "--output-root", str(tmp_path / "out"),
                                    "--state-json", "/missing/state", "--mode", "full", "--workers", "2"])
    with pytest.raises(ValueError, match="supports 1..1"):
        endpoint.main()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("outcome", [0, 7, "exception", "interrupt"])
def test_sample_persists_terminal_status(tmp_path, monkeypatch, outcome):
    raw, output = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    monkeypatch.setenv("EGOALIGN_ALLOW_SIM_REPLAY", "1")
    monkeypatch.setattr("sys.argv", ["sample", "basket", "--raw-root", str(raw), "--output-root", str(output),
                                    "--robot-xml", str(tmp_path / "model.xml")])
    def run(*args, **kwargs):
        if outcome == "exception":
            raise OSError("Cannot launch")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=outcome)
    monkeypatch.setattr(sample_pipeline.subprocess, "run", run)
    if outcome in {"exception", "interrupt"}:
        with pytest.raises((OSError, KeyboardInterrupt)):
            sample_pipeline.main()
    else:
        assert sample_pipeline.main() == outcome
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == ("complete" if outcome == 0 else "failed")
    assert state["finished_at"] >= state["started_at"]
    assert state["step"] == (5 if outcome == 0 else 1)
    assert not (output / "state.json.tmp").exists()


@pytest.mark.parametrize("command", ["candidate", "stage-a", "render-replay"])
def test_cli_keeps_callers_working_directory(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["egoalign", command, "--input", "relative/input"])
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda argv, **kw: calls.append((argv, kw)) or 0)
    assert cli.main() == 0
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][0][-1] == "relative/input"
