import gymnasium as gym
import mujoco

print(f"MuJoCo version: {mujoco.__version__}")

# Create a FetchReach-v2 environment
env = gym.make("FetchPush-v2", render_mode="human")

# Reset the environment to its initial state
observation, info = env.reset(seed=42)

# Run a loop for a few steps
for _ in range(1000):
    # Render the environment
    env.render()

    # Sample a random action
    action = env.action_space.sample()

    # Step through the environment with the sampled action
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode terminates, reset the environment
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment when done
env.close()

