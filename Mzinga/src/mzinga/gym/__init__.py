from mzinga.gym.hive_env import HiveEnv

try:
    from gymnasium.envs.registration import register

    _has_gymnasium = True
except ImportError:
    _has_gymnasium = False

if _has_gymnasium:
    for mode in ("self_play", "vs_opponent"):
        register(
            id=f"mzinga/Hive-{mode}-v0",
            entry_point="mzinga.gym.hive_env:HiveEnv",
            kwargs={"mode": mode},
        )
    register(
        id="mzinga/Hive-v0",
        entry_point="mzinga.gym.hive_env:HiveEnv",
    )
