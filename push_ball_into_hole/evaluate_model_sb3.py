import gymnasium as gym
from stable_baselines3 import PPO

gym.register(
    id="PushingBall-v0",
    entry_point="push:PushingBallEnv",  # Replace with the correct path
    max_episode_steps=1000,
)

# Initialize the environment
env = gym.make("PushingBall-v0", render_mode="human")

# Load the trained model
model = PPO.load("ppo_pushing_ball")

# Test the policy
obs = env.reset()
done = False
total_reward = 0

while not done:
    # Predict the next action using the trained model
    action, _states = model.predict(obs, deterministic=True)  # Set deterministic=False for exploration
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward: {total_reward}")
env.close()
