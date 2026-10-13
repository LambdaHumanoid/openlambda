"""Offline visualization validates telemetry without starting a controller."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from egoalign.render_replay import load_telemetry, main


def test_body_gray_is_visual_only():
    mujoco = pytest.importorskip("mujoco")
    from egoalign.render_replay import set_hand_color
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <asset><mesh name="pelvis" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
      <worldbody><body name="pelvis"><geom type="mesh" mesh="pelvis" rgba=".2 .2 .2 1"/>
        <body name="left_wrist_yaw_link"><geom size=".1" rgba="1 1 1 1"/></body>
        <body name="right_wrist_yaw_link"><geom size=".1" rgba="1 1 1 1"/></body>
      </body></worldbody></mujoco>''')
    original = model.geom_rgba.copy()
    mass, inertia = model.body_mass.copy(), model.body_inertia.copy()
    assert set_hand_color(model, "original")["geom_ids"] == []
    assert np.array_equal(original, model.geom_rgba)
    result = set_hand_color(model, "body-gray")
    assert result["geom_ids"] == [1, 2]
    assert np.allclose(model.geom_rgba, [.2, .2, .2, 1])
    assert np.array_equal(mass, model.body_mass)
    assert np.array_equal(inertia, model.body_inertia)


def telemetry(tmp_path, change=None):
    sim = {
        "monotonic_ns": np.array([9, 11, 20, 31], dtype=np.int64),
        "joint_names": np.array([f"joint_{n}" for n in range(29)]),
        "root_position": np.zeros((4, 3)),
        "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (4, 1)),
        "body_joint_position": np.zeros((4, 29)), "fps": np.array(30.),
    }
    control = {"capture_monotonic_ns": np.array([10, 20, 30], dtype=np.int64),
               "last_left_hand_action": np.zeros((3, 7)),
               "last_right_hand_action": np.zeros((3, 7))}
    if change:
        change(sim, control)
    paths = tmp_path / "episode.sim.npz", tmp_path / "episode.control.npz"
    np.savez(paths[0], **sim)
    np.savez(paths[1], **control)
    return paths


def test_preceding_command_and_boundary_clamp(tmp_path):
    sim, control, indices = load_telemetry(*telemetry(tmp_path))
    assert indices.tolist() == [0, 0, 1, 2]
    assert sim["body_joint_position"].shape == (4, 29)


@pytest.mark.parametrize("field,value,message", [
    ("monotonic_ns", np.array([1, 1, 2, 3]), "strictly increasing"),
    ("monotonic_ns", np.array([1, 2, 3, 4]), "do not overlap"),
    ("body_joint_position", np.full((4, 29), np.nan), "non-finite"),
    ("root_position", np.zeros((4, 2)), "shape"),
    ("root_quaternion_wxyz", np.ones((4, 4)), "unit length"),
    ("joint_names", np.array(["same"] * 29), "unique"),
    ("fps", np.array(0.), "positive"),
])
def test_invalid_sim_telemetry(tmp_path, field, value, message):
    paths = telemetry(tmp_path, lambda sim, control: sim.update({field: value}))
    with pytest.raises(ValueError, match=message):
        load_telemetry(*paths)


def test_invalid_hand_capture(tmp_path):
    paths = telemetry(tmp_path, lambda sim, control: control.update(
        last_left_hand_action=np.zeros((2, 6))))
    with pytest.raises(ValueError, match="last_left_hand_action"):
        load_telemetry(*paths)


def test_help_does_not_import_optional_dependencies():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
sys.modules['mujoco'] = None
sys.modules['av'] = None
from egoalign.render_replay import main
main(['--help'])
"""
    process = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "--output-dir" in process.stdout


def test_dispatcher_exposes_render_command():
    env = dict(os.environ, EGOALIGN_SIM_PYTHON=sys.executable)
    process = subprocess.run([sys.executable, "-m", "egoalign", "render-replay", "--help"],
                             env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "--sim" in process.stdout and "--control" in process.stdout


def test_dimensions_checked_before_rendering():
    with pytest.raises(SystemExit) as error:
        main(["--sim", "unused", "--control", "unused", "--output-dir", "unused",
              "--runtime-root", ".", "--scene", "unused", "--width", "101"])
    assert error.value.code == 2
