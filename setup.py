import gymnasium as gym

# Load your custom environment
env = gym.make('CustomFetchPush-v0')
env.reset()

# Run a test loop
for _ in range(100):
    action = env.action_space.sample()  # Use random actions for now
    observation, reward, done, info = env.step(action)
    env.render()  # Optionally, visualize the environment
    if done:
        env.reset()

env.close()  # Always close the environment at the end


