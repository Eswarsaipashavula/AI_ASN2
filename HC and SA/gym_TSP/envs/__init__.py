from gym.envs.registration import register

register(
    id="tsp-v0",
    entry_point="gym_TSP.envs.tsp_env:TSPEnv",
)
