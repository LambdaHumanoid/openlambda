"""OpenLambda HTTP policy for SIMPLE's decoupled_wbc (GR00T-WBC) controller.

Adapted from SIMPLE psi0_decoupled_wbc.py, copyright (c) 2025 Songlin Wei and Contributors,
licensed under SIMPLE's LICENSE terms.
"""

from __future__ import annotations

import os
import time

import numpy as np
from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.baselines.client import HttpActionClient
from simple.core.action import ActionCmd

from ._openlambda_common import (
    require_instruction,
    upper_joint_targets,
    upper_state,
    validate_action_chunk,
    waist_joint_targets,
)

DEFAULT_HEIGHT_F32 = float(np.float32(0.74))


class OpenlambdaDecoupledWbcAgent(SonicDecoupledWbcAgent):
    def __init__(
        self,
        robot,
        host: str,
        port: int,
        upsample_factor: int = 1,
        client=None,
        **kwargs,
    ):
        if robot.uid != "g1_sonic":
            raise ValueError(
                f"OpenLambda decoupled_wbc evaluation requires g1_sonic, got {robot.uid!r}."
            )
        super().__init__(robot, **kwargs)
        self.server_ip = host
        self.server_port = port
        self.upsample_factor = upsample_factor
        self._action_horizon = int(os.environ.get("SIMPLE_ACTION_HORIZON", "20"))
        self._height_snap = os.environ.get("SIMPLE_HEIGHT_SNAP", "1") == "1"
        self._state_mode = os.environ.get("SIMPLE_STATE_MODE", "measured")
        self._sample_offset = float(os.environ.get("SIMPLE_INTERP_SAMPLE_OFFSET", "0.0"))
        self.client = client if client is not None else HttpActionClient(host, port)
        self._global_step_idx = 0
        self._session_idx = 0
        self._session_id = self._make_session_id()
        self._last_height_command = DEFAULT_HEIGHT_F32
        self._prev_waist_rpy = np.zeros((1, 3), dtype=np.float32)
        self._reset_history = True

        indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.sonic_upper_joint_names = [
            name
            for name, idx in self._dwbc_robot_model.joint_to_dof_index.items()
            if idx in indices
        ]

    def _make_session_id(self) -> str:
        return f"openlambda-decoupled-wbc-{os.getpid()}-{self._session_idx}"

    def _current_time(self) -> float:
        if hasattr(self, "_now"):
            return self._now()
        return time.monotonic()

    def _tick_clock(self) -> None:
        if hasattr(self, "_advance_clock"):
            self._advance_clock()

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        require_instruction(instruction)
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]
        proprio_row = observation["joint_qpos"][None]
        # Measured waist (roll, pitch, yaw) from G1 qpos indices [13, 14, 12].
        cur_waist_rpy = proprio_row[:, [13, 14, 12]].astype(np.float32)

        if not self._action_queue:
            if self._state_mode == "frozen":
                tail = np.array([[0.0, 0.0, 0.0, DEFAULT_HEIGHT_F32]], dtype=np.float32)
            elif self._state_mode == "prev_waist":
                tail = np.concatenate(
                    [self._prev_waist_rpy, np.array([[self._last_height_command]], dtype=np.float32)],
                    axis=1,
                )
            else:
                tail = np.concatenate(
                    [cur_waist_rpy, np.array([[self._last_height_command]], dtype=np.float32)],
                    axis=1,
                )

            states = np.concatenate(
                [upper_state(observation["joint_qpos"])[None], tail],
                axis=1,
            ).astype(np.float32)

            ep_idx = int(info.get("episode_index", -1)) if info is not None else -1
            session_id = (
                f"openlambda-decoupled-wbc-ep{ep_idx}" if ep_idx >= 0 else self._session_id
            )
            history = {
                "session_id": session_id,
                "episode_index": ep_idx,
                "step_index": int(self._global_step_idx),
                "action_exec_horizon": int(self._action_horizon),
            }
            if self._reset_history:
                history["reset"] = True

            pred_action, *_ = self.client.query_action(
                {"rgb_head_stereo_left": observation["head_stereo_left"]},
                instruction,
                {"states": states},
                {},
                history=history,
                dataset="simple",
            )
            pred_action = validate_action_chunk(pred_action)
            self._reset_history = False

            steps_to_queue = (
                min(int(pred_action.shape[0]), self._action_horizon)
                if self._action_horizon > 0
                else int(pred_action.shape[0])
            )
            for i in range(steps_to_queue):
                action = pred_action[i]
                nav_cmd = action[32:36].copy()
                height_cmd = action[31:32].copy()
                if self._height_snap and abs(float(height_cmd[0]) - 0.74) < 1e-3:
                    height_cmd[0] = DEFAULT_HEIGHT_F32
                for _ in range(self.upsample_factor):
                    self.queue_action(
                        ActionCmd(
                            "vla_cmd",
                            target_upper_body_pose={
                                **upper_joint_targets(self.robot.joint_names, action),
                                **waist_joint_targets(action),
                            },
                            navigate_cmd=nav_cmd.copy(),
                            base_height_command=height_cmd.copy(),
                        )
                    )

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type != "vla_cmd":
            raise ValueError(f"Unexpected action type {action_cmd.type!r} from queue.")

        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)
        t_now = self._current_time()

        control_freq = self._control_frequency
        target_upper_body_pose = np.array(
            [action_cmd["target_upper_body_pose"][name] for name in self.sonic_upper_joint_names]
        )
        goal = {
            "target_upper_body_pose": target_upper_body_pose,
            "navigate_cmd": action_cmd["navigate_cmd"],
            "base_height_command": action_cmd["base_height_command"],
            "target_time": t_now + 1.0 / control_freq,
            "interpolation_garbage_collection_time": t_now - 2.0 / control_freq,
            "timestamp": t_now,
        }
        self._wbc_policy.set_goal(goal)
        if getattr(self, "_clock_mode", None) == "sim_step":
            self._wbc_policy.last_goal_time = t_now + self._sample_offset

        wbc_action = self._wbc_policy.get_action(time=t_now + self._sample_offset)
        self._tick_clock()

        executed_height = wbc_action.get("base_height_command")
        if executed_height is not None:
            h_val = float(np.asarray(executed_height).item())
            if self._height_snap and abs(h_val - 0.74) < 1e-3:
                self._last_height_command = DEFAULT_HEIGHT_F32
            else:
                self._last_height_command = float(np.float32(h_val))

        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(
            wbc_action["q"], side="left"
        )
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(
            wbc_action["q"], side="right"
        )

        action_cmd = ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )

        self._prev_waist_rpy = cur_waist_rpy
        self._last_pred_action = action_cmd
        self._global_step_idx += 1
        return action_cmd

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._global_step_idx = 0
        self._session_idx += 1
        self._session_id = self._make_session_id()
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_height_command = DEFAULT_HEIGHT_F32
        self._prev_waist_rpy = np.zeros((1, 3), dtype=np.float32)
