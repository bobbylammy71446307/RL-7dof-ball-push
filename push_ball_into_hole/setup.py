import gymnasium as gym
from push import PushingBallEnv

# Register the custom FetchPush environment
gym.register(
    id='PushingBall',
    entry_point='push:PushingBallEnv',  # Ensure the module path is correct based on your file structure
    max_episode_steps=50,
)

# Test the registered environment
env = gym.make('PushingBall', render_mode= 'human')
env.reset()
# Run a loop for a few steps
for _ in range(1000):
    # Render the environment
    env.render()

    # Sample a random action
    action = env.action_space.sample()
    #action = [0.1, 0.1, 0.1, 0.1]

    # Step through the environment with the sampled action
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode terminates, reset the environment
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment when done
env.close()


