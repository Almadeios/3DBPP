from gymnasium.envs.registration import register

register(
    id='Physics-v0',
    entry_point='environments.packing_env:PackingEnv',
)