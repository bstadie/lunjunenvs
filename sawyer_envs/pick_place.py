from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from sawyer_envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path
from sawyer_envs.multitask_env import MultitaskEnv
from sawyer_envs.base import SawyerXYZEnv


class SawyerPickAndPlaceEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,
            reward_type='hand_and_obj_distance',
            indicator_threshold=0.06,
            shaped_init=False,  # reset with 50% with object in hand.
            fix_object=False,
            # Make the object slightly offset so that we can extend to multiple objects.
            obj_init_pos=(0.05, 0.4, 0.02),
            hand_init_pos=(0.2, 0.525, 0.2),
            fixed_goal=None,
            hide_goal_markers=False,
            **kwargs
    ):
        self.horizon = 200
        self.cur_step = 0
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(self, model_name=self.model_name, **kwargs)
        self.shaped_init = shaped_init
        if obj_low is None:
            obj_low = self.hand_low
        if obj_high is None:
            obj_high = self.hand_high

        self.fix_object = fix_object

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.obj_init_pos = np.array(obj_init_pos)
        self.hand_init_pos = np.array(hand_init_pos)

        self.fixed_goal = None if fixed_goal is None else np.array(fixed_goal)
        self._state_goal = None

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_obj_space = Box(
            np.concatenate((self.hand_low, self.hand_low)),
            np.concatenate((self.hand_high, self.hand_high)),
        )
        self.hand_obj_dot_space = Box(
            np.concatenate((self.hand_low, self.hand_low, obj_low)),
            np.concatenate((self.hand_high, self.hand_high, obj_high)),
        )
        self.hand_space = Box(self.hand_low, self.hand_high)
        self.obj_space = Box(np.array(obj_low), np.array(obj_high))
        self.gripper_space = Box(np.array([-0.03] * 4), np.array([0.03] * 4), dtype=float)
        """
        The gym reacher environment returns the following.
            cos(theta), sin(theta), goal, theta_dot, delta
        Here instead, we return
            cos(theta), sin(theta), theta_dot, goal_posts (target + distractor), delta (with fingertip)
        We don't ever return the position of the finger tip.
        """
        self.observation_space = Dict([
            ('desired_goal', self.hand_obj_space),
            ('achieved_goal', self.hand_obj_space),
            ('observation', self.hand_obj_dot_space),
            ('state_observation', self.hand_obj_dot_space),
            ('state_desired_goal', self.hand_obj_space),
            ('state_achieved_goal', self.hand_obj_space),
            ('state_delta', self.hand_obj_space),
            ('state_touch_distance', self.hand_space),
            ('state_gripper', self.gripper_space),
        ])

    model_name = get_asset_full_path('sawyer_pick_and_place.xml')

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = .5
        self.viewer.cam.lookat[2] = 0.2
        self.viewer.cam.distance = 1.6
        self.viewer.cam.elevation = -25
        self.viewer.cam.azimuth = -45

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation((action[3], -action[3]))
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker()
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info(ob)
        done = False
        self.cur_step += 1
        if self.cur_step > self.horizon:
            self.cur_step = 0
            done = True
            #print(done)
        return ob, reward, done, info

    def _get_obs(self):
        e = self.get_endeff_pos()
        dot = self.get_endeff_vel()
        b = self.get_obj_pos()
        l, r = self.get_gripper_pos()
        flat_obs = np.concatenate((e, b, dot))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs[:6],
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs[:6],
            state_delta=flat_obs[:6] - self._state_goal,
            state_touch_distance=flat_obs[:3] - [0, 0, 0.03] - flat_obs[3:6],
            state_gripper=np.concatenate([l[:2] - e[:2], r[:2] - e[:2]])
        )

    PICK_REACH_KEYS = ('pick_reach_sparse', 'pick_reach_dense',)

    def _get_info(self, ob):
        achieved_goals = ob['state_achieved_goal']
        desired_goals = ob['state_desired_goal']
        obj_pos = achieved_goals[3:6]
        obj_goals = desired_goals[3:6]
        if self.reward_type in self.PICK_REACH_KEYS:
            obj_dist = np.linalg.norm(obj_goals - obj_pos, axis=-1)
            is_successful = obj_dist < 0.02
            return dict(success=is_successful)
        else:
            return {}

    def get_obj_pos(self):
        return self.data.get_body_xpos('obj_0').copy()

    def _set_goal_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (self._state_goal[:3])
        # note: add offset for object goal
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = (self._state_goal[3:6])
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (-1000)
            self.data.site_xpos[self.model.site_name2id('obj-goal-site'), 2] = (-1000)

    def _set_hand_goal(self, goal):
        """debugging function"""
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (goal)

    def _set_obj_goal(self, goal):
        """debugging function"""
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = (goal)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def sample_object(self, floored=True):
        pos = np.random.uniform(
            self.obj_space.low,
            self.obj_space.high,
        )
        if floored:
            pos[-1] = 0.02
        return pos

    def reset_model(self):
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker()

        if self.shaped_init and np.random.rand() < self.shaped_init:  # initialized with the object in the gripper at 50%
            obj_goal = self.sample_object(False) if self.fixed_goal is None else self.obj_init_pos
            self._reset_hand(obj_goal + [0, 0, 0.03])
            self.put_obj_in_hand()
        else:
            obj_goal = self.sample_object() if self.fixed_goal is None else self.obj_init_pos
            self._reset_hand()
            self._set_obj_xyz(obj_goal)
        return self._get_obs()

    def _reset_hand(self, pos=None):
        if pos is None:
            pos = self.hand_init_pos
        for _ in range(10):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', self.effector_quat)
            self.do_simulation(None, self.frame_skip)

    def put_obj_in_hand(self):
        new_obj_pos = self.data.get_site_xpos('endEffector')
        new_obj_pos[1] -= 0.01
        for i in range(5):
            self.do_simulation([1, -1])
        self._set_obj_xyz(new_obj_pos)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        hand_goal = state_goal[:3]
        for _ in range(30):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', self.effector_quat)
            # keep gripper closed
            self.do_simulation([1, -1])
        self._set_obj_xyz(state_goal[3:6])
        self.sim.forward()

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goals(self, batch_size, p_obj_in_hand=0.5):
        if self.fixed_goal is not None:
            goals = np.repeat(self.fixed_goal.copy()[None], batch_size, 0)
        else:
            goals = np.random.uniform(
                self.hand_obj_space.low,
                self.hand_obj_space.high,
                size=(batch_size, self.hand_obj_space.low.size),
            )
        # num_objs_in_hand = int(batch_size * p_obj_in_hand)
        # # Put object in hand
        # goals[:num_objs_in_hand, 3:] = goals[:num_objs_in_hand, :3].copy()
        # goals[:num_objs_in_hand, 4] -= 0.01
        # # Put object one the table (not floating)
        # goals[num_objs_in_hand:, 5] = self.obj_init_pos[2]
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        assert np.isclose(obs['state_achieved_goal'], obs['observation'][:, :6]).all(), 'need to be close'
        gripper_state = obs['state_gripper']
        hand_pos = achieved_goals[:, :3]
        obj_pos = achieved_goals[:, 3:6]
        hand_goals = desired_goals[:, :3]
        obj_goals = desired_goals[:, 3:6]
        hand_vel = achieved_goals[:, 6:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, axis=1)
        obj_distances = np.linalg.norm(obj_goals - obj_pos, axis=1)
        hand_and_obj_distances = hand_distances + obj_distances
        touch_distances = np.linalg.norm(hand_pos - obj_pos, axis=1)
        touch_and_obj_distance = touch_distances + obj_distances

        if self.reward_type == 'hand_distance':
            r = -hand_distances
        elif self.reward_type == 'hand_success':
            r = -(hand_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'obj_distance':
            r = -obj_distances
        elif self.reward_type == 'obj_success':
            r = -(obj_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'hand_and_obj_distance':
            r = -hand_and_obj_distances
        elif self.reward_type == 'hand_and_obj_success':
            r = -(hand_and_obj_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'touch_distance':
            r = -touch_distances
        elif self.reward_type == 'touch_success':
            r = -(touch_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'touch_and_obj_distance':
            r = -touch_and_obj_distance
        # start of Ge's shaped rewards
        elif self.reward_type == 'reach_dense':
            r = -hand_distances
        elif self.reward_type == 'hover_dense':
            _ = obj_pos - hand_pos + [0, 0, 0.03] + [0, 0, 0.05]
            _ = np.linalg.norm(_, axis=1)
            r = -_
        elif self.reward_type == 'touch_dense':
            offset = np.zeros_like(hand_pos)
            offset[:, -1] = - 0.03
            offset[:, :2] = (gripper_state[:, :2] + gripper_state[:, 2:]) / 2
            _ = obj_pos - (hand_pos + offset)
            touch_distances = np.linalg.norm(_, axis=1)
            touch_distances_xy = np.linalg.norm(_[:, :2], axis=1)
            gripper_opening = np.linalg.norm(gripper_state[:, :2] - gripper_state[:, 2:])
            dh = 0.1 - hand_pos[:, -1]
            h = hand_pos[:, -1] - 0.04 - 0.03
            opening = (gripper_opening - 0.04) / 2 + h * 1
            r = -touch_distances - (0 if touch_distances_xy < opening else 10) * np.maximum(dh, 0)
        # todo: make prong open
        elif self.reward_type == 'push_dense':
            _ = hand_pos - [0, 0, 0.03] - obj_pos
            touch_distances = np.linalg.norm(_, axis=1)
            r = -touch_distances - obj_distances
        elif self.reward_type == 'pick_dense':
            shape_fn = lambda x: np.select([x > -0.3], [10 * (x + 0.3) ** 2 + x], x)
            offset = np.zeros_like(hand_pos)
            offset[:, -1] = - 0.03
            offset[:, :2] = (gripper_state[:, :2] + gripper_state[:, 2:]) / 2
            _ = obj_pos - (hand_pos + offset)
            touch_distances = np.linalg.norm(_, axis=1)
            touch_distances_xy = np.linalg.norm(_[:, :2], axis=1)
            gripper_opening = np.linalg.norm(gripper_state[:, :2] - gripper_state[:, 2:])
            dh = 0.083 - hand_pos[:, -1]
            opening = 0.02
            height_penalty = (-10 if touch_distances_xy > opening else 0) * np.maximum(dh, 0)
            grip_reward = (5 * (0.1 - gripper_opening)) if touch_distances_xy < opening and (
                    hand_pos[:, -1] - obj_pos[:, -1]) < 0.065 else ((gripper_opening - 0.1) * 5)
            # add all rewards
            r = shape_fn(-touch_distances)
            r += height_penalty
            r += grip_reward
        elif self.reward_type == 'pick_reach_dense':
            shape_fn = lambda x: np.select([x > -0.3], [10 * (x + 0.3) ** 2 + x], x)
            offset = np.zeros_like(hand_pos)
            offset[:, -1] = - 0.03
            offset[:, :2] = (gripper_state[:, :2] + gripper_state[:, 2:]) / 2
            _ = obj_pos - (hand_pos + offset)
            touch_distances = np.linalg.norm(_, axis=1)
            touch_distances_xy = np.linalg.norm(_[:, :2], axis=1)
            gripper_opening = np.linalg.norm(gripper_state[:, :2] - gripper_state[:, 2:])
            dh = 0.083 - hand_pos[:, -1]
            opening = 0.02
            height_penalty = (-10 if touch_distances_xy > opening else 0) * np.maximum(dh, 0)
            obj_reward = -np.minimum(obj_distances, 1.5)  # clip in case of block falling off the table.
            grip_reward = (5 * (0.1 - gripper_opening)) if touch_distances_xy < opening and (
                    hand_pos[:, -1] - obj_pos[:, -1]) < 0.065 else ((gripper_opening - 0.1) * 5)
            # add all rewards
            r = shape_fn(-touch_distances)
            r += height_penalty
            r += grip_reward
            r += obj_reward * 5
        elif self.reward_type == 'pick_place_dense':
            raise NotImplemented
        # end of Ge's shaped rewards
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'obj_distance',
            'hand_and_obj_distance',
            'touch_distance',
            'hand_success',
            'obj_success',
            'hand_and_obj_success',
            'touch_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        self._set_goal_marker()


def pick_place_env(**kwargs):
    from .flat_goal_env import FlatGoalEnv
    return FlatGoalEnv(SawyerPickAndPlaceEnv(**kwargs),
                       obs_keys=('state_observation', 'state_desired_goal',
                                 'state_delta', 'state_touch_distance', 'state_gripper'),
                       goal_keys=('state_desired_goal',))


from gym.envs import register

# note: kwargs are not passed in to the constructor when entry_point is a function.
register(
    id="SawyerReach-v0",
    entry_point="sawyer_envs.pick_place:pick_place_env",
    kwargs=dict(frame_skip=5, reward_type="reach_dense",
                mocap_low=(-0.05, 0.35, 0.05),
                mocap_high=(0.45, 0.7, 0.35),
                hand_low=(-0.05, 0.35, 0.05),
                hand_high=(0.45, 0.7, 0.35),
                obj_low=(0.05, 0.45, 0.02),
                obj_high=(0.35, 0.6, 0.02)
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
#import gym
#gym.make("SawyerReach-v0")

register(
    id="SawyerHover-v0",
    entry_point="rl_maml_tf.envs.sawyer.pick_place:pick_place_env",
    # confined to the 2D plane.
    kwargs=dict(frame_skip=5, reward_type="hover_dense",
                mocap_low=(-0.05, 0.35, 0.05),
                mocap_high=(0.45, 0.7, 0.35),
                hand_low=(-0.05, 0.35, 0.05),
                hand_high=(0.45, 0.7, 0.35),
                obj_low=(0.05, 0.45, 0.02),
                obj_high=(0.35, 0.6, 0.02)
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
register(
    id="SawyerHoverNoTouch-v0",
    entry_point="rl_maml_tf.envs.sawyer.pick_place:pick_place_env",
    # confined to the 2D plane.
    kwargs=dict(frame_skip=5, reward_type="hover_dense",
                # note: No way that this touches the block
                mocap_low=(-0.05, 0.35, 0.1),
                mocap_high=(0.45, 0.7, 0.35),
                hand_low=(-0.05, 0.35, 0.1),
                hand_high=(0.45, 0.7, 0.35),
                obj_low=(0.05, 0.45, 0.02),
                obj_high=(0.35, 0.6, 0.02)
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
register(
    id="SawyerHoverRotated-v0",
    entry_point="rl_maml_tf.envs.sawyer.pick_place:pick_place_env",
    # confined to the 2D plane.
    kwargs=dict(frame_skip=5, reward_type="hover_dense",
                effector_quat=(0.5, -0.5, 0.5, 0.5,),
                mocap_low=(-0.05, 0.35, 0.05),
                mocap_high=(0.45, 0.7, 0.35),
                hand_low=(-0.05, 0.35, 0.05),
                hand_high=(0.45, 0.7, 0.35),
                obj_low=(0.05, 0.45, 0.02),
                obj_high=(0.35, 0.6, 0.02)
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
register(
    id="SawyerTouch-v0",
    entry_point="rl_maml_tf.envs.sawyer.pick_place:pick_place_env",
    # confined to the 2D plane.
    kwargs=dict(frame_skip=5, reward_type="touch_dense",
                mocap_low=(-0.05, 0.35, 0.05),
                mocap_high=(0.45, 0.7, 0.35),
                hand_low=(-0.05, 0.35, 0.05),
                hand_high=(0.45, 0.7, 0.35),
                obj_low=(0.05, 0.45, 0.02),
                obj_high=(0.35, 0.6, 0.02)
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
register(
    id="SawyerPush-v0",
    entry_point="rl_maml_tf.envs.sawyer.pick_place:pick_place_env",
    # confined to the 2D plane.
    kwargs=dict(frame_skip=5, reward_type="push_dense",
                mocap_low=(-0.05, 0.35, 0.05),
                mocap_high=(0.45, 0.7, 0.35),
                hand_low=(-0.05, 0.35, 0.05),
                hand_high=(0.45, 0.7, 0.35),
                obj_low=(0.05, 0.45, 0.02),
                obj_high=(0.35, 0.6, 0.02)
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
register(
    id="SawyerPick-v0",
    entry_point="rl_maml_tf.envs.sawyer.pick_place:pick_place_env",
    # Block goal can be in the air.
    kwargs=dict(frame_skip=5, reward_type="pick_dense",
                mocap_low=(-0.05, 0.35, 0.035),
                mocap_high=(0.45, 0.7, 0.35),
                hand_low=(-0.05, 0.35, 0.05),
                hand_high=(0.45, 0.7, 0.35),
                obj_low=(0.05, 0.45, 0.02),
                obj_high=(0.35, 0.6, 0.02)
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
register(
    id="SawyerPickReach-v0",
    entry_point="rl_maml_tf.envs.sawyer.pick_place:pick_place_env",
    # Block goal can be in the air.
    kwargs=dict(frame_skip=5, reward_type="pick_reach_dense",
                mocap_low=(-0.05, 0.35, 0.035),
                mocap_high=(0.45, 0.7, 0.35),
                hand_low=(-0.05, 0.35, 0.05),
                hand_high=(0.45, 0.7, 0.35),
                obj_low=(0.05, 0.45, 0.1),
                obj_high=(0.35, 0.6, 0.30),
                shaped_init=0.5,
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
register(
    id="SawyerPickAndPlace-v0",
    entry_point="rl_maml_tf.envs.sawyer.pick_place:pick_place_env",
    # Place goal has to be on the surface.
    kwargs=dict(frame_skip=5, reward_type="pick_place_dense",
                mocap_low=(-0.05, 0.35, 0.05),
                mocap_high=(0.45, 0.7, 0.35),
                hand_low=(-0.05, 0.35, 0.05),
                hand_high=(0.45, 0.7, 0.35),
                obj_low=(0.05, 0.45, 0.02),
                obj_high=(0.35, 0.6, 0.02)
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
