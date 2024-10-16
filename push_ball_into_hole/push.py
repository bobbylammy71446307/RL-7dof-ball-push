import os

from gymnasium.utils.ezpickle import EzPickle

#from gymnasium_robotics.envs.fetch import MujocoFetchEnv
from base import MujocoSimulation

# Ensure we get the path separator correct on windows
# Use os.path.expanduser to handle the tilde (~) for the home directory
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "fetch", "push.xml")



class PushingBallEnv(MujocoSimulation, EzPickle):
    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoSimulation.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_end_effector=True,
            n_substeps=20,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)