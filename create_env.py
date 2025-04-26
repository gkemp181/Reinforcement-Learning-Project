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

        # pull out the actual goal so we can avoid it
        goal_pos = obs["desired_goal"][:2].copy()

        # 3) now sample a block (x,y) that is:
        #    a) at least min_dist from the gripper home
        #    b) at least min_dist from the goal
        home_xy  = u.initial_gripper_xpos[:2]
        obj_range = u.obj_range
        min_dist  = u.distance_threshold

        while True:
            offset = rng.uniform(-obj_range, obj_range, size=2)
            # 3a) must be outside the “too-close to gripper” zone
            if np.linalg.norm(offset) < min_dist:
                continue
            candidate_xy = home_xy + offset
            # 3b) must be outside the “too-close to goal” zone
            if np.linalg.norm(candidate_xy - goal_pos) < min_dist:
                continue
            # if we get here, both checks passed
            break

        # 4) actually place the block
        blk_qpos = utils.get_joint_qpos(model, data, "object0:joint")
        blk_qpos[0:2] = candidate_xy
        blk_qpos[2]    = 0.42  # table height
        utils.set_joint_qpos(model, data, "object0:joint", blk_qpos)

        # 5) forward-kinematics + fresh obs
        u._mujoco.mj_forward(model, data)
        obs = u._get_obs()

        return obs, info


def create_env(render_mode=None, sparse=False):

    gym.register_envs(gymnasium_robotics)

    if(sparse):
        environment = "FetchPickAndPlace-v3"
    else:
        environment = "FetchPickAndPlaceDense-v3"

    # create and wrap your env
    base_env = gym.make(environment, render_mode=render_mode)
    env      = RandomizedFetchWrapper(base_env)

    return env