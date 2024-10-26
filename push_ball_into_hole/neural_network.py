import torch
import torch.nn as nn

class PushNetwork(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim):
        super(PushNetwork, self).__init__()
        
        # Observation processing network
        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),  # Layer Norm
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # Layer Norm
            nn.ReLU(),
        )
        
        # Goal processing network
        self.goal_net = nn.Sequential(
            nn.Linear(goal_dim * 2, 256),
            nn.LayerNorm(256),  # LayerNorm
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # LayerNorm
            nn.ReLU(),
        )
        
        # Combine features
        combined_dim = 128 + 128
        
        # Policy network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),  # Output action mean
        )
        
        # Value network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Output state value
        )
        
    def forward(self, observation, achieved_goal, desired_goal):
        # Process observation
        obs_features = self.obs_net(observation)
        
        # Process goals
        goal_input = torch.cat([achieved_goal, desired_goal], dim=1)  # Use dim=1 for batch dimension
        goal_features = self.goal_net(goal_input)
        
        # Combine features
        combined_features = torch.cat([obs_features, goal_features], dim=1)
        
        # Compute action mean (policy output)
        action_mean = self.policy_net(combined_features)
        
        # Compute value estimate (value function output)
        value = self.value_net(combined_features)
        
        return action_mean, value

# Corrected dimensions based on your environment
obs_dim = 24       # Size of the 'observation' vector
goal_dim = 3       # Size of the 'achieved_goal' and 'desired_goal' vectors
action_dim = 4     # Size of your action space

# Batch size for the example
batch_size = 2     # Adjust as needed

# Create the network
policy_network = PushNetwork(obs_dim, goal_dim, action_dim)

# Example inputs (replace with actual tensors from your data)
observation = torch.randn(batch_size, obs_dim)
achieved_goal = torch.randn(batch_size, goal_dim)
desired_goal = torch.randn(batch_size, goal_dim)

# Forward pass
action_mean, value = policy_network(observation, achieved_goal, desired_goal)

# Print the outputs
print("Action Mean:\n", action_mean)
print("Action Mean Shape:", action_mean.shape)
print("\nValue:\n", value)
print("Value Shape:", value.shape)
