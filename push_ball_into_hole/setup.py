import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Register the custom environment
gym.register(
    id='PushingBall-v0',
    entry_point='push:PushingBallEnv',
    max_episode_steps=200,
)

# Function to create a new instance of the environment
def make_env():
    return gym.make('PushingBall-v0')

# Vectorize the environment (PPO works well with multiple parallel environments)
vec_env = make_vec_env(make_env, n_envs=4)

# Define the PPO model using MultiInputPolicy
model = PPO('MultiInputPolicy', vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_pushing_ball")

# Test the trained model
obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()

vec_env.close()



