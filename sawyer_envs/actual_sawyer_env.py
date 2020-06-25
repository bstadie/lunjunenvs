from sawyer_envs.pick_place import pick_place_env
import numpy as np
from sawyer_envs.door_multitask import door_env
from sawyer_envs.mixed_env import mixed_env
from operator import iadd
from functools import reduce


def get_sawyer_reach_env():
    #  sawyer reach.
    kwargs = dict(frame_skip=5, reward_type="reach_dense",
                  mocap_low=(-0.05, 0.35, 0.05),
                  mocap_high=(0.45, 0.7, 0.35),
                  hand_low=(-0.05, 0.35, 0.05),
                  hand_high=(0.45, 0.7, 0.35),
                  obj_low=(0.05, 0.45, 0.02),
                  obj_high=(0.35, 0.6, 0.02)
                  )

    env = pick_place_env(**kwargs)
    return env



def get_sawyer_door_envs(env_name="default"):
    if env_name == 'default':
        kwargs = dict(frame_skip=5,
                      k_tasks=3,
                      sample_task_on_reset=True,
                      reward_type="door_dense",
                      mocap_low=(-0.5, 0.25, 0.035),
                      mocap_high=(0.5, 0.8, 0.35),
                      hand_low=(-0.35, 0.35, 0.05),
                      hand_high=(0.35, 0.7, 0.35),
                      )

        env = door_env(**kwargs)
        #kk = env.reset()
        #print(len(env.reset()))
        env.obs_dim = 25
        env.action_dim = 8
        return env


def get_sawyer_mixed_envs(env_name="default"):
    obj_goal_low = (-0.05, 0.35, 0.2)
    obj_goal_high = (0.05, 0.35, 0.2)
    obj_init_pos_0 = reduce(iadd, [
                          (-0.2, 0.525, 0.05),
                          (-0.0, 0.525, 0.05),
                          (0.2, 0.525, 0.05)
                      ], [])
    obj_init_pos_1 = reduce(iadd, [
        (-0.15, 0.325, 0.05),
        (-0.0, 0.525, 0.05),
        (0.23, 0.425, 0.05)
    ], [])

    obj_init_pos_2 = reduce(iadd, [
        (-0.15, 0.325, 0.05),
        (-0.15, 0.425, 0.05),
        (-0.1, 0.345, 0.05)
    ], [])
    obj_init_pos_3 = reduce(iadd, [
        (0.2, 0.625, 0.05),
        (0.1, 0.425, 0.05),
        (0.2, 0.525, 0.05)
    ], [])
    if env_name == 'default':
        kwargs = dict(frame_skip=5,
                      k_tasks=6,
                      sample_task_on_reset=False,  # note: this is what shuffles the tasks
                      reward_type="dense",  #"dense"
                      obj_init_pos=obj_init_pos_1,
                      mocap_low=(-0.5, 0.25, 0.035),
                      mocap_high=(0.5, 0.8, 0.35),
                      hand_low=(-0.15, 0.45, 0.2),
                      hand_high=(0.15, 0.65, 0.4),
                      obj_low=obj_goal_low,
                      obj_high=obj_goal_high
                      )

        env = mixed_env(**kwargs)
        print(len(env.reset()))
        #env.render()
        # print(env.reset().shape())
        env.obs_dim = 52
        env.action_dim = 8
        env.horizon = 250
        return env


def get_other_sawyer_envs(env_name='reach'):
        if env_name == 'reach':
            kwargs=dict(frame_skip=5, reward_type="reach_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )
            obs_dim = 11
            act_dim = 8
        elif env_name == 'hover':
            kwargs=dict(frame_skip=5, reward_type="hover_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )
        elif env_name == 'hover_no_touch':
            kwargs=dict(frame_skip=5, reward_type="hover_dense",
                        # note: No way that this touches the block
                        mocap_low=(-0.05, 0.35, 0.1),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.1),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )
        elif env_name == 'hover_hand_rotated':
            kwargs= dict(frame_skip=5, reward_type="hover_dense",
                        effector_quat=(0.5, -0.5, 0.5, 0.5,),
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )

        elif env_name == 'touch':
            kwargs=dict(frame_skip=5, reward_type="touch_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )
        elif env_name == 'push':
            kwargs=dict(frame_skip=5, reward_type="push_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )

        elif env_name == 'pick':
            kwargs=dict(frame_skip=5, reward_type="pick_dense",
                        mocap_low=(-0.05, 0.35, 0.035),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )

        elif env_name == 'pick_reach':
            kwargs=dict(frame_skip=5, reward_type="pick_reach_dense",
                        mocap_low=(-0.05, 0.35, 0.035),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.1),
                        obj_high=(0.35, 0.6, 0.30),
                        shaped_init=0.5,
                        )
            act_dim = 8
            obs_dim = 28
        elif env_name == 'pick_place':
            kwargs = dict(frame_skip=5, reward_type="pick_place_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )

        env = pick_place_env(**kwargs)
        #print(len(env.reset()))
        env.obs_dim = obs_dim
        env.action_dim = act_dim
        return env

def main():
    env = get_sawyer_env()
    env.reset()
    for i in range(100000):
        env.step(np.random.randn(4))
        env.render()


if __name__ == "__main__":
    main()
