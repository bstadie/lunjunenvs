import numpy as np
import tensorflow as tf
import copy
#from kick_ass_utils import get_env
from envs.point import PointEnv
from sawyer_envs.actual_sawyer_env import get_sawyer_mixed_envs, get_other_sawyer_envs, get_sawyer_reach_env

def run_v2():


    env_name = 'point'
    env = PointEnv(sparse_reward=False)
    #env = get_other_sawyer_envs('pick_reach')
    #env = get_sawyer_reach_env()
    #env.horizon = 600
    env.horizon = 100
    horizon = env.horizon
    obs_orig = env.reset()
    print(F"OBS IS: {obs_orig}")
    obs = env.reset()
    print(F"OBS AFTER RESET IS: {obs}")
    goal_orig = env.get_goal()
    print(F"GOAL IS: {goal_orig}")
    env.reset()
    # goal should not change after reset.
    print(F"GOAL AFTER RESET IS: {env.get_goal()}")
    env.set_goal(obs_orig[0:2])
    print(F"GOAL AFTER SETTING GOAL IS: {env.get_goal()}")



    while True:
        #env.reset_model()
        rand_goal = env.get_random_goal()
        env.set_goal(rand_goal)
        obs = env.reset()
        for i in range(horizon):
            #action, _states = model.predict(obs)
            #action += 1.0 * 0.25 * np.random.randn(len(action))
            #action = 0.0 * 0.3 * np.random.randn(len(action))
            #action -= 0.74
            obs, rewards, dones, info = env.step(np.random.uniform(-1.0, 1.0, 2))
            #print(obs.shape)
            #print(action.shape)
            #kkkkk
            env.render()
        #print(F"Final Rew: {rewards}")
        print(F"Resetting GOAL")


def main():
    sess = tf.Session()
    with sess.as_default():
        run_v2()



if __name__ == "__main__":
    main()
