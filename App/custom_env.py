import numpy as np
import gymnasium as gym
import gymnasium_robotics
import mujoco

class CustomFetchWrapper(gym.Wrapper):
    def __init__(self, env, block_xy=None, goal_xyz=None):
        super().__init__(env)
        self.u = env.unwrapped  # MujocoFetchPickAndPlaceEnv
        # stash your fixed coords (or None to randomize)
        self.default_block_xy = (np.array(block_xy, dtype=float)
                                 if block_xy is not None else None)
        self.default_goal_xyz = (np.array(goal_xyz, dtype=float)
                                 if goal_xyz is not None else None)

    def reset(self, *args, **kwargs):
        # 1) do the normal reset — gets you a random goal in obs
        obs, info = super().reset(*args, **kwargs)
        u     = self.unwrapped
        model = u.model
        data  = u.data
        utils = u._utils
        rng   = u.np_random

        # 2) reset the robot slides to your home pose
        for name, val in zip(
            ["robot0:slide0","robot0:slide1","robot0:slide2"],
            [0.405,       0.48,        0.0],
        ):
            utils.set_joint_qpos(model, data, name, val)

        # 3) pick block position
        if self.default_block_xy is None:
            # — original random‐sampling —
            home_xy   = u.initial_gripper_xpos[:2]
            obj_range = u.obj_range
            min_dist  = u.distance_threshold
            while True:
                offset = rng.uniform(-obj_range, obj_range, size=2)
                if np.linalg.norm(offset) < min_dist:
                    continue
                cand = home_xy + offset
                if np.linalg.norm(cand - obs["desired_goal"][:2]) < min_dist:
                    continue
                break
            block_xy = cand
        else:
            block_xy = self.default_block_xy

        # place the block
        blk_qpos = utils.get_joint_qpos(model, data, "object0:joint")
        blk_qpos[0:2] = block_xy
        blk_qpos[2]    = 0.42  # table height
        utils.set_joint_qpos(model, data, "object0:joint", blk_qpos)

        # 4) pick goal position
        if self.default_goal_xyz is None:
            # — original “raise above table” logic —
            raise_z = 0.1 + rng.uniform(0, 0.2)
            new_goal = obs["desired_goal"].copy()
            new_goal[2] = blk_qpos[2] + raise_z
        else:
            new_goal = self.default_goal_xyz

        # override the goal both in the env and in the MuJoCo site
        u.goal = new_goal
        sid = mujoco.mj_name2id(model,
                                mujoco.mjtObj.mjOBJ_SITE,
                                "target0")
        data.site_xpos[sid] = new_goal

        # 5) forward‐kinematics + fresh obs
        u._mujoco.mj_forward(model, data)
        obs = u._get_obs()

        return obs, info


def create_env(render_mode=None, block_xy=None, goal_xyz=None):
    gym.register_envs(gymnasium_robotics)
    base_env = gym.make("FetchPickAndPlace-v3", render_mode=render_mode)
    u = base_env.unwrapped

    # 1) compute table center in world coords
    #    – X,Y: same as the gripper’s initial XY (over table center)
    #    – Z: the table‐top height the wrapper uses (0.42 m)
    center_xy = u.initial_gripper_xpos[:2]        # e.g. [1.366, 0.750]
    table_z   = 0.42                              # match blk_qpos[2] in your wrapper
    table_center = np.array([*center_xy, table_z])

    # 2) turn your “relative” block_xy into an absolute XY
    if block_xy is not None:
        rel = np.array(block_xy, dtype=float)
        abs_block_xy = center_xy + rel
    else:
        abs_block_xy = None

    # 3) turn your “relative” goal_xyz into an absolute XYZ
    if goal_xyz is not None:
        rel = np.array(goal_xyz, dtype=float)
        abs_goal_xyz = table_center + rel
    else:
        abs_goal_xyz = None

    # 4) build the wrapped env with those absolutes
    env = CustomFetchWrapper(
        base_env,
        block_xy=abs_block_xy,
        goal_xyz=abs_goal_xyz
    )
    return env