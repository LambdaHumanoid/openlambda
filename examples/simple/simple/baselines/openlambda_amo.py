"""OpenLambda HTTP policy for SIMPLE's AMO controller.

Adapted from SIMPLE psi0.py, copyright (c) 2025 Songlin Wei and Contributors,
licensed under SIMPLE's LICENSE terms.
"""

import os

import numpy as np
from simple.agents.primitive_agent import PrimitiveAgent
from simple.baselines.client import HttpActionClient
from simple.core.action import ActionCmd

from ._openlambda_common import (
    require_instruction,
    upper_joint_targets,
    upper_state,
    validate_action_chunk,
    waist_joint_targets,
)


class OpenlambdaAmoAgent(PrimitiveAgent):
    def __init__(
        self,
        robot,
        host: str,
        port: int,
        upsample_factor=1,
        client=None,
        skip_stand_warmup: bool = False,
        **kwargs,
    ):
        if robot.uid != "g1_wholebody":
            raise ValueError(
                f"OpenLambda AMO evaluation requires g1_wholebody, got {robot.uid!r}."
            )
        super().__init__(robot, **kwargs)
        self.server_ip = host
        self.server_port = port
        self.upsample_factor = upsample_factor
        self.client = client if client is not None else HttpActionClient(host, port)
        self._global_step_idx = 0
        self._session_idx = 0
        self._session_id = self._make_session_id()
        self._skip_stand_warmup = skip_stand_warmup
        self._reset_history = True

    def _make_session_id(self):
        return f"openlambda-amo-{os.getpid()}-{self._session_idx}"

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        require_instruction(instruction)
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if self._global_step_idx == 0 and not self._skip_stand_warmup:
            for _ in range(60):
                self.queue_loco_command(
                    command=[0, 0, 0, 0, 0, 0, 0, 0],
                    motion_type="stand",
                    keep_waist_pose=False,
                )

        if not self._action_queue:
            # Original SIMPLE observes commanded torso RPY and absolute height.
            states = np.concatenate(
                (
                    upper_state(observation["joint_qpos"]),
                    [
                        self.robot.command[6],
                        self.robot.command[5],
                        self.robot.command[4],
                        self.robot.command[3] + 0.75,
                    ],
                )
            ).astype(np.float32)[None]
            history = {
                "session_id": self._session_id,
                "episode_index": int(info.get("episode_index", -1)) if info is not None else -1,
                "step_index": int(self._global_step_idx),
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
            # Execute the complete native model chunk, including upsampling.
            for action in pred_action:
                for _ in range(self.upsample_factor):
                    command = [
                        action[32],  # vx
                        action[35],  # target yaw
                        action[33],  # vy
                        action[31] - 0.75,
                        # Independent torso YPR commands are not waist PD targets.
                        *self.robot.command[4:7],
                        action[34],  # turning flag
                    ]
                    self.queue_action(
                        ActionCmd(
                            "eval_move_actuators",
                            target_qpos=upper_joint_targets(self.robot.joint_names, action),
                            action_command=command,
                            waist_qpos=waist_joint_targets(action),
                        )
                    )

        action_cmd = super().get_action(observation, instruction, **kwargs)
        self._last_pred_action = action_cmd
        self._global_step_idx += 1
        return action_cmd

    def reset(self):
        super().reset()
        self._global_step_idx = 0
        self._session_idx += 1
        self._session_id = self._make_session_id()
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
