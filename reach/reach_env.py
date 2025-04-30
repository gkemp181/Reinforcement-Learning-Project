import numpy as np
import gymnasium as gym
import gymnasium_robotics

class RandomizedFetchWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.u = env.unwrapped  # MujocoFetchPickAndPlaceEnv

    def reset(self, **kwargs):
        # 1) do the normal reset — this gives you obs with a random goal
        obs, info = super().reset(**kwargs)
        u     = self.unwrapped
        model = u.model
        data  = u.data
        utils  = u._utils
        rng    = u.np_random

        # 2) fix the robot slides to home
        for name, val in zip(
            ["robot0:slide0","robot0:slide1","robot0:slide2"],
            [0.405,       0.48,        0.0],
        ):
            utils.set_joint_qpos(model, data, name, val)

        # 5) forward-kinematics + fresh obs
        u._mujoco.mj_forward(model, data)
        obs = u._get_obs()

        return obs, info


def create_env(render_mode=None):

    gym.register_envs(gymnasium_robotics)

    # create and wrap your env
    base_env = gym.make("FetchReach-v3", render_mode=render_mode)
    env      = RandomizedFetchWrapper(base_env)
    
    return env