import gymnasium as gym
from push import PushingBallEnv
import torch
import numpy as np
from PPO_network import PushNetwork, PPOAgent


# Register the custom FetchPush environment
gym.register(
    id='PushingBall',
    entry_point='push:PushingBallEnv',  # Ensure the module path is correct based on your file structure
    max_episode_steps=50,
)

# Test the registered environment
env = gym.make('PushingBall', render_mode= 'human')

# Define dimensions based on your environment's observation and action spaces
obs_dim = 25      # Replace with the actual observation dimension
goal_dim = 3       # Replace with the actual goal dimension
action_dim = 4     # Replace with the actual action dimension

# Create an instance of your network
network = PushNetwork(obs_dim, goal_dim, action_dim)

# Create an instance of PPOAgent
agent = PPOAgent(network)

# Hyperparameters
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5
num_epochs = 100
num_steps_per_update = 2048
mini_batch_size = 64
ppo_epochs = 10
max_steps_per_episode = 200

# Rollout buffer
class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.achieved_goals = []
        self.desired_goals = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.observations = []
        self.achieved_goals = []
        self.desired_goals = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

buffer = RolloutBuffer()

# Training loop
total_steps = 0
episode = 0

while episode < num_epochs:
    state, info = env.reset()
    observation = state['observation']
    achieved_goal = state['achieved_goal']
    desired_goal = state['desired_goal']

    episode_rewards = 0  # To track episode rewards

    for step in range(max_steps_per_episode):
        # Select action using PPOAgent
        action, log_prob, value = agent.select_action(observation, achieved_goal, desired_goal)

        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition in buffer
        buffer.observations.append(observation)
        buffer.achieved_goals.append(achieved_goal)
        buffer.desired_goals.append(desired_goal)
        buffer.actions.append(action)
        buffer.log_probs.append(log_prob)
        buffer.values.append(value)
        buffer.rewards.append(reward)
        buffer.dones.append(done)

        episode_rewards += reward

        # Move to next state
        observation = next_state['observation']
        achieved_goal = next_state['achieved_goal']
        desired_goal = next_state['desired_goal']

        total_steps += 1

        # Check if it's time to update the network
        if total_steps % num_steps_per_update == 0:
            # Compute returns and advantages
            returns = []
            advantages = []
            G = 0
            adv = 0
            last_value = 0

            if not done:
                _, _, last_value = agent.select_action(observation, achieved_goal, desired_goal)
            else:
                last_value = 0

            for i in reversed(range(len(buffer.rewards))):
                mask = 1.0 - buffer.dones[i]
                G = buffer.rewards[i] + gamma * G * mask
                delta = buffer.rewards[i] + gamma * last_value * mask - buffer.values[i]
                adv = delta + gamma * gae_lambda * adv * mask
                returns.insert(0, G)
                advantages.insert(0, adv)
                last_value = buffer.values[i]

            # Convert lists to numpy arrays
            observations = np.array(buffer.observations)
            achieved_goals = np.array(buffer.achieved_goals)
            desired_goals = np.array(buffer.desired_goals)
            actions = np.array(buffer.actions)
            old_log_probs = np.array(buffer.log_probs)
            returns = np.array(returns)
            advantages = np.array(advantages)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Optimize policy for ppo_epochs
            for _ in range(ppo_epochs):
                # Create mini-batches
                indices = np.arange(len(buffer.rewards))
                np.random.shuffle(indices)
                for start in range(0, len(buffer.rewards), mini_batch_size):
                    end = start + mini_batch_size
                    mb_indices = indices[start:end]

                    # Mini-batch data
                    batch_data = {
                        'observations': observations[mb_indices],
                        'achieved_goals': achieved_goals[mb_indices],
                        'desired_goals': desired_goals[mb_indices],
                        'actions': actions[mb_indices],
                        'old_log_probs': old_log_probs[mb_indices],
                        'returns': returns[mb_indices],
                        'advantages': advantages[mb_indices],
                    }

                    # Compute loss
                    total_loss = agent.compute_loss(
                        batch_data, gamma, gae_lambda, clip_epsilon, value_loss_coef, entropy_coef
                    )

                    # Update network
                    agent.update(total_loss, max_grad_norm)

            # Clear buffer
            buffer.clear()

        if done:
            break

    episode += 1
    print(f"Episode {episode} completed. Total Reward: {episode_rewards}")

env.close()