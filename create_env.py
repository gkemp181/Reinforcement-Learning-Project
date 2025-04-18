import numpy as np
import gymnasium as gym
import gymnasium_robotics

class RandomizedFetchWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.u = env.unwrapped  # MujocoFetchPickAndPlaceEnv

    def reset(self, **kwargs):
        # 1) do the normal reset
        obs, info = super().reset(**kwargs)

        u     = self.unwrapped        
        model = u.model
        data  = u.data
        utils = u._utils
        rng   = u.np_random            # same RNG that the env uses

        # 2) restore robot slides to home
        for name, val in zip(
            ["robot0:slide0", "robot0:slide1", "robot0:slide2"],
            [0.405,      0.48,       0.0],
        ):
            utils.set_joint_qpos(model, data, name, val)

        # 3) sample block (x,y) *around the XML’s initial gripper*…
        home_xy    = u.initial_gripper_xpos[:2]  # [1.3419, 0.7491]
        obj_range  = u.obj_range                # 0.15 m
        min_dist   = u.distance_threshold       # 0.10 m

        # draw until > min_dist
        offset = rng.uniform(-obj_range, obj_range, size=2)
        while np.linalg.norm(offset) < min_dist:
            offset = rng.uniform(-obj_range, obj_range, size=2)

        # 4) write block joint:
        blk_qpos = utils.get_joint_qpos(model, data, "object0:joint")
        blk_qpos[0:2] = home_xy + offset
        blk_qpos[2]    = 0.42     # fixed table-top height :contentReference[oaicite:0]{index=0}
        utils.set_joint_qpos(model, data, "object0:joint", blk_qpos)

        # 5) forward kinematics
        u._mujoco.mj_forward(model, data)

        # refresh obs so we have up‑to‑date positions
        obs = u._get_obs()
        block_pos = obs["observation"][:3]
        goal_pos  = obs["desired_goal"]
        # if overlap, try again
        
        if np.linalg.norm(block_pos - goal_pos) < 10*u.distance_threshold:
            return self.reset(**kwargs)
        
        return obs, info


def create_env():
    gym.register_envs(gymnasium_robotics)

    # create and wrap your env
    base_env = gym.make("FetchPickAndPlace-v3", render_mode="human")
    env      = RandomizedFetchWrapper(base_env)

    return env