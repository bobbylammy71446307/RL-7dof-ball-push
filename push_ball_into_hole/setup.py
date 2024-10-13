import gymnasium as gym
from push import MujocoFetchPushEnv

# Register the custom FetchPush environment
gym.register(
    id='MujocoFetchPush-v0',
    entry_point='push:MujocoFetchPushEnv',  # Ensure the module path is correct based on your file structure
    max_episode_steps=50,
)

# Test the registered environment
env = gym.make('MujocoFetchPush-v0', render_mode= 'human')
env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Sample a random action
    env.step(action)  # Apply the action
    env.render()  # Render the environment (ensure MuJoCo rendering setup is correct)
env.close()


