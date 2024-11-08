import gymnasium as gym
import torch
import numpy as np
from PPO_network import PushNetwork  # Assuming this has the network definition

# Define dimensions based on your environment's observation and action spaces
obs_dim = 25      # Replace with the actual observation dimension
goal_dim = 3       # Replace with the actual goal dimension
action_dim = 4     # Replace with the actual action dimension

# Initialize the network and load the saved policy
network = PushNetwork(obs_dim, goal_dim, action_dim)
network.load_state_dict(torch.load("trained_policy.pth"))
network.eval()  # Set the network to evaluation mode

# Set up the environment
env = gym.make('PushingBall', render_mode='human')
state, info = env.reset()
observation = state['observation']
achieved_goal = state['achieved_goal']
desired_goal = state['desired_goal']

# Run the environment with the loaded policy
for _ in range(100):
    # Convert observation data to tensors
    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    achieved_goal_tensor = torch.tensor(achieved_goal, dtype=torch.float32).unsqueeze(0)
    desired_goal_tensor = torch.tensor(desired_goal, dtype=torch.float32).unsqueeze(0)

    # Select action using the policy
    with torch.no_grad():
        action_mean, _ = network(obs_tensor, achieved_goal_tensor, desired_goal_tensor)
    action = action_mean.squeeze().numpy()

    # Step the environment with the chosen action
    next_state, reward, terminated, truncated, _ = env.step(action)

    # Check if episode is done
    if terminated or truncated:
        break

    # Update observations
    observation = next_state['observation']
    achieved_goal = next_state['achieved_goal']
    desired_goal = next_state['desired_goal']

env.close()
