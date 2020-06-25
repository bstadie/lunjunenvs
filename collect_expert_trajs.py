import numpy as np
import tensorflow as tf
import copy
#from kick_ass_utils import get_env
from envs.point import PointEnv
from sawyer_envs.actual_sawyer_env import get_sawyer_mixed_envs, get_other_sawyer_envs, get_sawyer_reach_env

def run_v2():
    import gym
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines import PPO2
    from stable_baselines import SAC as SAC2
    from slightly_modified_sax import SAC
    from stable_baselines import PPO1
    from stable_baselines.sac.policies import MlpPolicy as Spol
    from stable_baselines.common.policies import MlpPolicy as PPOPol

    #horizon = 200

    env_name = 'point'
    reward_threshold = -0.2
    env = PointEnv(sparse_reward=False)
    #env = get_other_sawyer_envs('pick_reach')
    #env = get_sawyer_reach_env()
    #env.horizon = 600
    env.horizon = 100
    horizon = env.horizon
    print(horizon)
    #env.reset = env.reset_fixed_goal
    #kkkk
    model = SAC(Spol, env, verbose=1, learning_starts=500)
    model.learn(total_timesteps=30*1000)
    #model = PPO1(env=env, policy=PPOPol, verbose=0)
    #model.learn(total_timesteps=2*25*40000, log_interval=10000)

    save_trajs = False
    if save_trajs:
        all_states = []
        all_actions = []
        n_trajs_to_save = 100
        while len(all_states) < n_trajs_to_save:
            this_traj_states, this_traj_actions = [], []
            if env_name in ['point_disctractor', 'reacher_distractor']:
                obs = env.reset_fixed_goal(0)
            else:
                obs = env.reset()
            for i in range(horizon):
                action, _states = model.predict(obs)
                this_traj_actions.append(copy.deepcopy(action))
                this_traj_states.append(copy.deepcopy(obs))
                action += 1.1*np.random.randn(len(action))
                obs, rewards, dones, info = env.step(action)
                #env.render()
            #print(F"Final Rew: {rewards}")
            if rewards > reward_threshold:
                #print("Saving good traj")
                this_traj_states = np.array(this_traj_states)
                this_traj_actions = np.array(this_traj_actions)
                all_states.append(this_traj_states)
                all_actions.append(this_traj_actions)
                if len(all_states) % 1000 == 0:
                    print(F"saved {len(all_states)} states")

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        d = dict()
        d['expert_obs'] = all_states
        d['expert_acs'] = all_actions
        np.savez("expert_trajs_dir/expert_trajs_" + env_name + ".npz", **d)

    while True:
        #env.reset_model()
        obs = env.reset()

        for i in range(horizon):
            action, _states = model.predict(obs)
            action += 1.0 * 0.25 * np.random.randn(len(action))
            #action = 0.0 * 0.3 * np.random.randn(len(action))
            #action -= 0.74
            obs, rewards, dones, info = env.step(action)
            #print(obs.shape)
            #print(action.shape)
            #kkkkk
            env.render()
        print(F"Final Rew: {rewards}")


def main():
    sess = tf.Session()
    with sess.as_default():
        run_v2()



if __name__ == "__main__":
    main()
