from policies import RandomPol, BCPol
from prettytable import PrettyTable
from sawyer_envs.actual_sawyer_env import get_other_sawyer_envs, get_sawyer_door_envs, get_sawyer_mixed_envs
from envs.point import PointEnv
import numpy as np
from buffers import ExpertBuffer, HindsightBuffer
from utils import init_tf
import tensorflow as tf
from baselines_bc import learn as max_liklihood_bc

from baselines.gail import mlp_policy


def random_pol_baseline():
    success = 0
    trials = 0
    env = PointEnv()
    obs = env.reset()
    policy = RandomPol(env=env)
    for test_idx in range(1000):
        obs = env.reset()
        for i in range(300):
            act = policy.act(obs)[0]
            obs, _, _, _ = env.step(act)
            #print(obs)
            #env.render()
            #print(env.get_reward())

        rew = env.get_reward()
        if np.abs(rew - 1.0) < 0.01:
            success += 1
        trials += 1
        if test_idx % 10 == 0:
            print(F"{100*success/trials} percent success rate")




def supervised_learning():
    success = 0
    trials = 0


    buffer = ExpertBuffer('expert_trajs_point.npz')
    s_mb, a_mb = buffer.sample()

    policy = BCPol(s_mb, a_mb)
    init_tf()

    env = PointEnv()
    obs = env.reset()


    train_iters = int(1e5)
    for i in range(train_iters):
        s_mb, a_mb = buffer.sample()
        loss = policy.train(s_mb, a_mb)
        if i % 10000 == 0:
            #print(F"Loss at step {i} is: {loss}")
            #pass
            #  test the policy.
            success = 0
            trials = 0
            for test_idx in range(10):
                obs = env.reset()
                for step_idx in range(300):
                    act = policy.act(obs)
                    obs, _, _, _ = env.step(act)
                    # print(obs)
                    # env.render()
                    # print(env.get_reward())

                rew = env.get_reward()
                if np.abs(rew - 1.0) < 0.01:
                    success += 1
                trials += 1
                if test_idx == 9:
                    print(F"{100*success/trials} percent success rate")


    while True:
        for test_idx in range(100):
            obs = env.reset()
            for i in range(300):
                act = policy.act(obs)
                obs, _, _, _ = env.step(act)
                env.render()


def maximum_liklihood():
    success = 0
    trials = 0

    buffer = ExpertBuffer('expert_trajs_point.npz', batch_size=256, number_of_trajs_to_use=20000)
    s_mb, a_mb = buffer.sample()

    env = PointEnv()
    obs = env.reset()

    #policy = BCPol(s_mb, a_mb)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=32, num_hid_layers=2)

    init_tf()
    policy = max_liklihood_bc(env,
                              policy_fn,
                              buffer,
                              max_iters=int(1e5),
                              ckpt_dir=None,
                              log_dir=None,
                              task_name=None,
                              verbose=True)



    for test_idx in range(100):
        obs = env.reset()
        for i in range(250):
            act = policy.act(False, obs)[0]
            obs, _, _, _ = env.step(act)
            # print(obs)
            # env.render()
            # print(env.get_reward())

        rew = env.get_reward()
        if np.abs(rew - 1.0) < 0.01:
            success += 1
        trials += 1
        if test_idx % 10 == 0:
            print(F"{100*success/trials} percent success rate")


    while True:
        obs = env.reset()
        for i in range(250):
            act = policy.act(False, obs)[0] + 0.1*np.random.randn(2)
            obs, _, _, _ = env.step(act)
            env.render()
            # print(obs)
            # env.render()
            # print(env.get_reward())


def hindsight_supervised_learning():
    success = 0
    trials = 0

    buffer = HindsightBuffer(obs_shape=8, acs_shape=2, future_k=1)
    s_mb, a_mb = buffer.sample()

    policy = BCPol(s_mb, a_mb)
    init_tf()

    env = PointEnv(sparse_reward=True)
    obs = env.reset()
    train_iters = 10000
    for i in range(train_iters):

        rollout_obs = []
        rollout_acts = []
        obs = env.reset()
        for j in range(250):
            act = policy.act(obs)
            obs, rew, _, _ = env.step(act)
            rollout_obs.append(obs)
            rollout_acts.append(act)

        buffer.add_traj(rollout_obs, rollout_acts)
        rew = env.get_reward()
        trials += 1
        if np.abs(rew - 1.0) < 0.1:
            success += 1

        #  train policy.
        for optim_step in range(10):
            s_mb, a_mb = buffer.sample()
            loss = policy.train(s_mb, a_mb)

        # evaluation
        if i % 100 == 0:
            #print(F"Loss at step {i} is: {loss}")
            print(F"{100*success/trials} percent success rate at step {i}")
            success = 0
            trials = 0


    while True:
        for test_idx in range(100):
            obs = env.reset()
            for i in range(300):
                act = policy.act(obs)
                obs, _, _, _ = env.step(act)
                env.render()


def hindsight_maximum_liklihood():
    pass


if __name__ == "__main__":
    sess = tf.Session()
    with sess.as_default():
        hindsight_supervised_learning()
