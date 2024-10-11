
import os
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations


#I will be creating an environment where a robot arm pushes a ball to a hole

#I will have push.xml for the environment and robot.xml for the robot arm
#fetch_env.py and push.py will be the files that will be used to create the environment


#


class PushBallIntoHole(gym.Env):
    def __init__(self):
        self.env = gym.make("FetchPush-v3", render_mode="human")
        self.observation, self.info = self.env.reset(seed=42)
        self.action = self.env.action_space.sample()
        self.reward = 0
        self.terminated = False
        self.truncated = False
    
    def step(self):
        self.observation, self.reward, self.