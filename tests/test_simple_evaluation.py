"""Result integrity and backend adapter checks without a simulator or model installation."""

from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


ROOT = Path(__file__).parents[1]

eval_spec = importlib.util.spec_from_file_location(
    "openlambda_eval_simple", ROOT / "scripts/eval_simple.py"
)
evaluation = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(evaluation)

runner_spec = importlib.util.spec_from_file_location(
    "openlambda_simple_eval_runner", ROOT / "scripts/_simple_eval.py"
)
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


class EvaluationCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.selection = {
            "task": "bend_pick", "level": "level-0", "env": "simple/Test-v0",
            "backend": "amo", "policy": "openlambda_amo", "episode_start": 3,
            "num_episodes": 2, "save_video": False,
        }

    def result(self, lines, returncode=0):
        (self.output / "eval_stats.txt").write_text(lines)
        return evaluation.summarize(self.output, self.selection, returncode)

    def test_failed_episode_is_a_completed_evaluation(self):
        result = self.result("episode_3: True\nepisode_4: False\n")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["exit_code"], 0)

    def test_wrong_or_incomplete_episode_sets_cannot_report_a_score(self):
        for lines in (
            "episode_3: True\n",
            "episode_3: True\nepisode_3: False\nepisode_4: True\n",
            "episode_0: True\nepisode_1: True\n",
            "episode_3: True\nepisode_4: True\nepisode_5: True\n",
            "episode_3: True\nepisode_4: unknown\n",
        ):
            with self.subTest(lines=lines):
                result = self.result(lines)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["success_rate"])
                self.assertNotEqual(result["exit_code"], 0)

    def test_native_failure_overrides_complete_episode_stats(self):
        result = self.result("episode_3: True\nepisode_4: True\n", -15)
        self.assertEqual(result["exit_code"], 143)
        self.assertIsNone(result["success_rate"])

    def test_requested_video_must_exist_for_every_episode(self):
        self.selection["save_video"] = True
        result = self.result("episode_3: True\nepisode_4: False\n")
        self.assertEqual(result["missing_video_episode_ids"], ["episode_3", "episode_4"])
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["success_rate"])


class SimpleBackendAndTasksConfigTests(unittest.TestCase):
    def test_tasks_config_maps_amo_and_decoupled_wbc_backends(self):
        tasks = json.loads((ROOT / "configs/simple/tasks.json").read_text())
        self.assertEqual(
            set(tasks),
            {
                "bend_pick",
                "tabletop_grasp",
                "handover",
                "xmove_pick",
                "xmove_bend_pick",
                "mobile_pick_place",
            },
        )
        for name in ("bend_pick", "tabletop_grasp"):
            self.assertEqual(tasks[name]["backend"], "amo")
            self.assertEqual(tasks[name]["policy"], "openlambda_amo")
            self.assertEqual(tasks[name]["eval_module"], "simple.cli.eval")
            self.assertEqual(tasks[name]["default_success_criteria"], 0.9)
        for name in ("handover", "xmove_pick", "xmove_bend_pick", "mobile_pick_place"):
            self.assertEqual(tasks[name]["backend"], "decoupled_wbc")
            self.assertEqual(tasks[name]["policy"], "openlambda_decoupled_wbc")
            self.assertEqual(tasks[name]["eval_module"], "simple.cli.eval_decoupled_wbc")
            self.assertEqual(tasks[name]["default_action_horizon"], 20)
        self.assertEqual(tasks["mobile_pick_place"]["default_success_criteria"], 0.4)

    def test_validate_environment_backend(self):
        runner.validate_environment_backend(
            "simple/G1WholebodyBendPickMP-v0",
            "simple.envs.loco_manipulation:LocoManipulationEnv",
            "amo",
        )
        runner.validate_environment_backend(
            "simple/G1WholebodyXMovePickTeleop-v0",
            "simple.envs.sonic_loco_manip:SonicLocoManipEnv",
            "decoupled_wbc",
        )
        with self.assertRaises(RuntimeError):
            runner.validate_environment_backend(
                "simple/G1WholebodyXMovePickTeleop-v0",
                "simple.envs.sonic_loco_manip:SonicLocoManipEnv",
                "amo",
            )
        with self.assertRaises(RuntimeError):
            runner.validate_environment_backend(
                "simple/G1WholebodyBendPickMP-v0",
                "simple.envs.loco_manipulation:LocoManipulationEnv",
                "decoupled_wbc",
            )
        with self.assertRaises(ValueError):
            runner.validate_environment_backend(
                "simple/Test-v0",
                "simple.envs.loco_manipulation:LocoManipulationEnv",
                "unknown",
            )


@unittest.skipIf(np is None, "numpy is required for adapter unit tests")
class DecoupledWbcAdapterTests(unittest.TestCase):
    def _load_agent_class(self):
        class FakeActionCmd(dict):
            def __init__(self, action_type, **kwargs):
                super().__init__(**kwargs)
                self.type = action_type

        class FakeWbcPolicy:
            def __init__(self):
                self.observations = []
                self.goals = []
                self.last_goal_time = 0.0

            def set_observation(self, obs):
                self.observations.append(obs)

            def set_goal(self, goal):
                self.goals.append(goal)

            def get_action(self, time):
                return {
                    "q": np.zeros(43, dtype=np.float32),
                    "base_height_command": self.goals[-1]["base_height_command"],
                }

        class FakeRobotModel:
            def __init__(self):
                self.joint_to_dof_index = {
                    f"j_{i}": i for i in range(43) if i not in (12, 13, 14)
                }
                self.joint_to_dof_index.update(
                    {
                        "waist_yaw_joint": 12,
                        "waist_roll_joint": 13,
                        "waist_pitch_joint": 14,
                    }
                )

            def get_joint_group_indices(self, group):
                assert group == "upper_body"
                return list(range(12, 43))

            def get_body_actuated_joints(self, q):
                return q[:29]

            def get_hand_actuated_joints(self, q, side):
                return q[29:36] if side == "left" else q[36:43]

        class FakeSonicDecoupledWbcAgent:
            def __init__(self, robot, **kwargs):
                self.robot = robot
                self._action_queue = deque()
                self._dwbc_robot_model = FakeRobotModel()
                self._wbc_policy = FakeWbcPolicy()
                self._control_frequency = 50.0
                self._clock_mode = "sim_step"
                self._sim_time = 0.0

            def _now(self):
                return self._sim_time

            def _advance_clock(self):
                self._sim_time += 1.0 / self._control_frequency

            def _build_wbc_observation(self, proprio):
                return {"proprio": proprio}

            def queue_action(self, action_cmd):
                self._action_queue.append(action_cmd)

            def get_action(self, observation, instruction=None, **kwargs):
                return self._action_queue.popleft()

            def reset(self, **kwargs):
                self._sim_time = 0.0
                self._action_queue = deque()

        fake_simple = types.ModuleType("simple")
        fake_agents = types.ModuleType("simple.agents")
        fake_sonic = types.ModuleType("simple.agents.sonic_decoupled_wbc_agent")
        fake_sonic.SonicDecoupledWbcAgent = FakeSonicDecoupledWbcAgent
        fake_core = types.ModuleType("simple.core")
        fake_action = types.ModuleType("simple.core.action")
        fake_action.ActionCmd = FakeActionCmd
        fake_baselines = types.ModuleType("simple.baselines")
        fake_client = types.ModuleType("simple.baselines.client")
        fake_client.HttpActionClient = object

        common_spec = importlib.util.spec_from_file_location(
            "simple.baselines._openlambda_common",
            ROOT / "examples/simple/simple/baselines/_openlambda_common.py",
        )
        common_mod = importlib.util.module_from_spec(common_spec)
        common_spec.loader.exec_module(common_mod)

        injected = {
            "simple": fake_simple,
            "simple.agents": fake_agents,
            "simple.agents.sonic_decoupled_wbc_agent": fake_sonic,
            "simple.core": fake_core,
            "simple.core.action": fake_action,
            "simple.baselines": fake_baselines,
            "simple.baselines.client": fake_client,
            "simple.baselines._openlambda_common": common_mod,
        }
        saved = {name: sys.modules.get(name) for name in injected}
        sys.modules.update(injected)
        try:
            agent_spec = importlib.util.spec_from_file_location(
                "simple.baselines.openlambda_decoupled_wbc",
                ROOT / "examples/simple/simple/baselines/openlambda_decoupled_wbc.py",
            )
            agent_mod = importlib.util.module_from_spec(agent_spec)
            agent_spec.loader.exec_module(agent_mod)
            return agent_mod.OpenlambdaDecoupledWbcAgent
        finally:
            for name, old in saved.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

    def test_rejects_non_g1_sonic_robot(self):
        Agent = self._load_agent_class()

        class WrongRobot:
            uid = "g1_wholebody"

        with self.assertRaises(ValueError):
            Agent(WrongRobot(), "127.0.0.1", 8000)

    def test_state_horizon_height_snap_and_reset(self):
        Agent = self._load_agent_class()

        class FakeRobot:
            uid = "g1_sonic"
            joint_names = [f"j_{i}" for i in range(43)]

            def prepare_obs(self):
                return {"q": np.zeros(43, dtype=np.float32)}

        class FakeClient:
            def __init__(self):
                self.calls = []

            def query_action(self, obs, instruction, state_dict, cond, history, dataset):
                self.calls.append((instruction, state_dict["states"].copy(), dict(history), dataset))
                chunk = np.zeros((50, 36), dtype=np.float32)
                chunk[:, 31] = 0.7404  # within 1e-3 of 0.74 -> snapped to float32(0.74)
                chunk[:, 32:36] = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
                return chunk, 0.0, None

        old_horizon = os.environ.get("SIMPLE_ACTION_HORIZON")
        os.environ["SIMPLE_ACTION_HORIZON"] = "4"
        try:
            client = FakeClient()
            agent = Agent(FakeRobot(), "127.0.0.1", 8000, client=client)
            qpos = np.arange(43, dtype=np.float32) * 0.01
            obs = {
                "joint_qpos": qpos,
                "head_stereo_left": np.zeros((8, 8, 3), dtype=np.uint8),
            }
            for step in range(5):
                cmd = agent.get_action(obs, instruction="pick up the bottle", info={"episode_index": 2})
                self.assertEqual(cmd.type, "decoupled_wbc")

            # Horizon=4 -> queried at step 0 and step 4
            self.assertEqual(len(client.calls), 2)
            instr0, state0, hist0, ds0 = client.calls[0]
            self.assertEqual(instr0, "pick up the bottle")
            self.assertEqual(ds0, "simple")
            self.assertEqual(state0.shape, (1, 32))
            np.testing.assert_allclose(state0[0, 28:31], qpos[[13, 14, 12]])
            self.assertAlmostEqual(float(state0[0, 31]), float(np.float32(0.74)), places=6)
            self.assertTrue(hist0.get("reset"))
            self.assertEqual(hist0["session_id"], "openlambda-decoupled-wbc-ep2")
            self.assertEqual(hist0["action_exec_horizon"], 4)

            _, _, hist1, _ = client.calls[1]
            self.assertNotIn("reset", hist1)
            self.assertEqual(hist1["step_index"], 4)

            # Height snap checked on goal sent to WBC
            self.assertAlmostEqual(
                float(agent._wbc_policy.goals[0]["base_height_command"][0]),
                float(np.float32(0.74)),
                places=7,
            )

            # Reset restores initial state
            agent.reset()
            self.assertEqual(agent._global_step_idx, 0)
            self.assertTrue(agent._reset_history)
            self.assertEqual(len(agent._action_queue), 0)
        finally:
            if old_horizon is None:
                os.environ.pop("SIMPLE_ACTION_HORIZON", None)
            else:
                os.environ["SIMPLE_ACTION_HORIZON"] = old_horizon


if __name__ == "__main__":
    unittest.main()
