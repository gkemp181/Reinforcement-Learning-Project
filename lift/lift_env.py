import numpy as np
import gymnasium as gym
import gymnasium_robotics
import mujoco
import random

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

        # ——— raise the target above the table ———
        # how far above the table you want the goal:
        raise_z = 0.1 + random.random()*0.2
        # copy the original desired goal, but bump its Z:
        new_goal = obs["desired_goal"].copy()
        new_goal[2] = blk_qpos[2] + raise_z
        # override the env’s goal so that _get_obs() will see it:
        u.goal = new_goal
        # also move the invisible “target” site in the MuJoCo sim:
        target_site = "target0"
        # option A: use mujoco.mj_name2id
        sid = mujoco.mj_name2id(model,
                                mujoco.mjtObj.mjOBJ_SITE,
                                target_site)
        data.site_xpos[sid] = new_goal

        # 5) forward-kinematics + fresh obs
        u._mujoco.mj_forward(model, data)
        obs = u._get_obs()

        return obs, info


def create_env(render_mode=None):

    gym.register_envs(gymnasium_robotics)

    # create and wrap your env
    base_env = gym.make("FetchPickAndPlace-v3", render_mode=render_mode)
    env      = RandomizedFetchWrapper(base_env)

    return env