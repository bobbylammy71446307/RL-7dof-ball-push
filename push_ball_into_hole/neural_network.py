import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Your existing network architecture (unchanged)
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

# Set dimensions based on your environment
obs_dim = 24       # Size of the 'observation' vector
goal_dim = 3       # Size of the 'achieved_goal' and 'desired_goal' vectors
action_dim = 4     # Size of your action space

# Create the network (unchanged)
network = PushNetwork(obs_dim, goal_dim, action_dim)

# Create an optimizer
learning_rate = 1e-3
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# Training parameters
num_epochs = 100
batch_size = 64
num_batches = 100  # Number of batches per epoch

# Lists to store loss values
total_loss_values = []
policy_loss_values = []
value_loss_values = []

# Training loop
for epoch in range(num_epochs):
    epoch_total_loss = 0.0
    epoch_policy_loss = 0.0
    epoch_value_loss = 0.0
    for _ in range(num_batches):
        # Generate random input data
        observation = torch.randn(batch_size, obs_dim)
        achieved_goal = torch.randn(batch_size, goal_dim)
        desired_goal = torch.randn(batch_size, goal_dim)
        
        # Forward pass
        action_mean, value = network(observation, achieved_goal, desired_goal)
        
        # Generate synthetic target actions and target values
        # For demonstration, define target actions and values as functions of the inputs
        target_action = observation[:, :action_dim]  # Use first few observation features as target action
        target_value = (desired_goal - achieved_goal).norm(dim=1, keepdim=True)  # Distance between goals
        
        # Compute policy loss (MSE between action_mean and target_action)
        policy_loss = nn.functional.mse_loss(action_mean, target_action)
        
        # Compute value loss (MSE between value and target_value)
        value_loss = nn.functional.mse_loss(value, target_value)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        epoch_total_loss += total_loss.item()
        epoch_policy_loss += policy_loss.item()
        epoch_value_loss += value_loss.item()
    
    # Average losses for the epoch
    average_total_loss = epoch_total_loss / num_batches
    average_policy_loss = epoch_policy_loss / num_batches
    average_value_loss = epoch_value_loss / num_batches
    
    total_loss_values.append(average_total_loss)
    policy_loss_values.append(average_policy_loss)
    value_loss_values.append(average_value_loss)
    
    # Print losses every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {average_total_loss:.4f}, "
              f"Policy Loss: {average_policy_loss:.4f}, Value Loss: {average_value_loss:.4f}")

# Plot the losses over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), total_loss_values, label='Total Loss')
plt.plot(range(1, num_epochs + 1), policy_loss_values, label='Policy Loss')
plt.plot(range(1, num_epochs + 1), value_loss_values, label='Value Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()

