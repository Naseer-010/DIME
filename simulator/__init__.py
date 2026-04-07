from __future__ import annotations

import gymnasium as gym
from gymnasium.error import Error

from .cascade_guard_env import CascadeGuardMissionControlEnv


def register_env() -> None:
    try:
        gym.register(
            id="CascadeGuardMissionControl-v0",
            entry_point="simulator.cascade_guard_env:CascadeGuardMissionControlEnv",
        )
    except Error:
        pass


register_env()

__all__ = ["CascadeGuardMissionControlEnv", "register_env"]
