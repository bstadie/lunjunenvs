import numpy as np

class ExpertBuffer:
    def __init__(self, expert_trajs_path, number_of_trajs_to_use=None, batch_size=64):
        trajs = np.load('expert_trajs_dir/' + expert_trajs_path)
        expert_obs = trajs['expert_obs']
        expert_acts = trajs['expert_acs']
        if number_of_trajs_to_use:
            expert_obs = expert_obs[0:number_of_trajs_to_use]
            expert_acts = expert_acts[0:number_of_trajs_to_use]
        print(F"Fed in set of {len(expert_obs)} expert trajectories")
        expert_obs = np.reshape(expert_obs, (expert_obs.shape[0]*expert_obs.shape[1], expert_obs.shape[2]))
        expert_acts = np.reshape(expert_acts, (expert_acts.shape[0]*expert_acts.shape[1], expert_acts.shape[2]))
        self.expert_obs = expert_obs
        self.expert_acts = expert_acts
        self.expert_obs, self.expert_acts = unison_shuffled_copies(self.expert_obs, self.expert_acts)

        self.sample_idx = 0
        self.batch_size = batch_size


    def sample(self):
        lower = self.sample_idx*self.batch_size
        upper = lower + self.batch_size
        if upper > len(self.expert_obs):
            lower = 0
            upper = self.batch_size
            self.sample_idx = 0

        return self.expert_obs[lower:upper], self.expert_acts[lower:upper]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class HindsightBuffer:
    #  buffer only works on point right now!!!
    def __init__(self, obs_shape, acs_shape, buffer_size=30000, traj_length=250, future_k=250, batch_size=256):
        assert buffer_size % traj_length == 0
        self.future_k = future_k
        self.obs_shape = obs_shape
        self.acs_shape = acs_shape
        self.buffer_size = buffer_size
        self.traj_length = traj_length
        self.obs = np.zeros((buffer_size, obs_shape))
        self.acts = np.zeros((buffer_size, acs_shape))
        self.buffer_fill_idx = 0
        self.buffer_full = False
        self.batch_size = batch_size

    def add_traj(self, traj_obs, traj_acs):
        assert len(traj_obs) == len(traj_acs) == self.traj_length

        # goal relabeling.
        for i, traj_ob in enumerate(traj_obs):
            fetch_goal_from_time = min(i + self.future_k, self.traj_length-1)
            future_k_goal = traj_obs[fetch_goal_from_time][2:4]
            traj_ob[2:4] = future_k_goal
            traj_obs[i] = traj_ob

        traj_obs, traj_acs = np.array(traj_obs), np.array(traj_acs)
        lower_buffer_bound = self.buffer_fill_idx*self.traj_length
        upper_buffer_bound = (self.buffer_fill_idx+1)*self.traj_length
        if upper_buffer_bound > self.buffer_size:
            self.buffer_full = True
            self.buffer_fill_idx = 0
            lower_buffer_bound = 0
            upper_buffer_bound = self.traj_length

        traj_obs = np.squeeze(traj_obs)
        traj_acs = np.squeeze(traj_acs)

        self.obs[lower_buffer_bound:upper_buffer_bound] = traj_obs
        self.acts[lower_buffer_bound:upper_buffer_bound] = traj_acs
        self.buffer_fill_idx += 1

    def sample(self):
        if not self.buffer_full:
            lower_buffer_bound = 0
            upper_buffer_bound = self.buffer_fill_idx * self.traj_length
        else:
            lower_buffer_bound = 0
            upper_buffer_bound = self.buffer_size

        if self.buffer_fill_idx == 0 and self.buffer_full == False:
            return np.zeros((self.batch_size, self.obs_shape)), np.zeros((self.buffer_size, self.acs_shape))

        chosen_idxs = np.random.choice(upper_buffer_bound, self.batch_size)
        selected_obs = self.obs[chosen_idxs]
        selected_acts = self.acts[chosen_idxs]
        return selected_obs, selected_acts
