import gym
from gym import spaces
import numpy as np

class TSPEnv(gym.Env):
    """Custom Gym Environment for TSP"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_cities=500):
        super(TSPEnv, self).__init__()
        self.num_cities = num_cities

        # Define observation space: (current city, visited cities)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(num_cities),  # Current city
            spaces.MultiBinary(num_cities)  # Visited cities (binary flags)
        ))

        # Define action space: choosing the next city
        self.action_space = spaces.Discrete(num_cities)

        # Initialize city coordinates (random positions in 2D space)
        self.cities = np.random.rand(num_cities, 2)*100

        # Compute Euclidean distance matrix between cities
        self.distances = np.linalg.norm(
            self.cities[:, None, :] - self.cities[None, :, :], axis=-1
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state"""
        super().reset(seed=seed)  # Ensure compatibility with Gym
        
        self.visited = np.zeros(self.num_cities, dtype=np.int8)  # Store as int8
        self.current_city = np.random.randint(self.num_cities)
        self.visited[self.current_city] = 1
        self.total_distance = 0

        return self._get_obs(), {}  # Return obs and empty info dict

    def _get_obs(self):
        """Return the current state as (current city, visited cities)"""
        return (self.current_city, self.visited.copy())  # ✅ Now correctly formatted

    def step(self, action):
        if self.visited[action]:  
            reward = -100
            terminated = False
        else:
            distance = self.distances[self.current_city][action]
            self.total_distance += distance
            self.visited[action] = True
            reward = -distance
            self.current_city = action
            terminated = self.visited.all()

        truncated = False  # No early termination logic, so always False
        return (self.current_city, self.visited.copy()), reward, terminated, truncated, {}

    def render(self, mode="human"):
        """Print the current state of the environment"""
        print(f"Current city: {self.current_city}, Visited: {self.visited}")

# Test environment
# if __name__ == "__main__":
#     env = TSPEnv(num_cities=10)
#     obs, _ = env.reset()  # ✅ Fix: Unpack tuple correctly
#     done = False
#     while not done:
#        action = env.action_space.sample()
#        obs, reward, terminated, truncated, _ = env.step(action)
#        done = terminated or truncated  # Combine both termination conditions
#        env.render()
