from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RtcContext:
    session_id: str
    action_exec_horizon: int
    inference_delay: int
    action_prefix: np.ndarray | None
    extrapolating: bool


class RtcSessionStore:
    """Keep normalized RTC action chunks isolated by SIMPLE client session."""

    def __init__(
        self,
        *,
        num_actions: int,
        training_max_delay: int,
        allow_extrapolation: bool,
        max_sessions: int = 2048,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        if training_max_delay < 0 or training_max_delay > num_actions:
            raise ValueError("training_max_delay must be in [0,num_actions].")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive.")
        self.num_actions = num_actions
        self.training_max_delay = training_max_delay
        self.allow_extrapolation = allow_extrapolation
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, tuple[int, np.ndarray]] = OrderedDict()

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    def prepare(self, history: Any) -> RtcContext:
        if not isinstance(history, dict):
            raise ValueError("RTC requests require a history object.")
        session_id = history.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("RTC history.session_id must be a non-empty string.")
        horizon_value = history.get("action_exec_horizon")
        if isinstance(horizon_value, bool) or not isinstance(horizon_value, int):
            raise ValueError("RTC history.action_exec_horizon must be an integer.")
        if not 1 <= horizon_value <= self.num_actions:
            raise ValueError(
                "RTC action_exec_horizon must be in "
                f"[1,{self.num_actions}], got {horizon_value}."
            )
        inference_delay = self.num_actions - horizon_value
        extrapolating = inference_delay > self.training_max_delay
        if extrapolating and not self.allow_extrapolation:
            raise ValueError(
                f"RTC inference_delay={inference_delay} exceeds the training "
                f"maximum {self.training_max_delay}."
            )

        previous = None if history.get("reset") else self._sessions.get(session_id)
        action_prefix = None
        if previous is not None:
            previous_horizon, previous_actions = previous
            if previous_horizon != horizon_value:
                raise ValueError(
                    "RTC action_exec_horizon changed inside one session: "
                    f"previous={previous_horizon}, current={horizon_value}."
                )
            action_prefix = previous_actions[horizon_value:].copy()
            self._sessions.move_to_end(session_id)

        return RtcContext(
            session_id=session_id,
            action_exec_horizon=horizon_value,
            inference_delay=inference_delay,
            action_prefix=action_prefix,
            extrapolating=extrapolating,
        )

    def commit(self, context: RtcContext, normalized_actions: np.ndarray) -> None:
        actions = np.asarray(normalized_actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != self.num_actions:
            raise ValueError(
                "RTC normalized action chunk must have shape "
                f"[{self.num_actions},D], got {actions.shape}."
            )
        if not np.isfinite(actions).all():
            raise ValueError("RTC normalized action chunk contains non-finite values.")
        self._sessions[context.session_id] = (
            context.action_exec_horizon,
            actions.copy(),
        )
        self._sessions.move_to_end(context.session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
