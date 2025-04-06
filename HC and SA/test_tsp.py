import gym
import gym_TSP

env = gym.make("tsp-v0")

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated  # Proper termination handling
    env.render()
