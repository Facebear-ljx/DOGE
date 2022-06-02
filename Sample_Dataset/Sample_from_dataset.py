import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.s0, self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7 = np.zeros(1), np.zeros(1), np.zeros(
            1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6, self.a7 = np.zeros(1), np.zeros(1), np.zeros(
            1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7 = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(
            1), np.zeros(1), np.zeros(1), np.zeros(1)
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.reward1 = np.zeros((max_size, 1))
        self.reward2 = np.zeros((max_size, 1))
        self.reward3 = np.zeros((max_size, 1))
        self.reward4 = np.zeros((max_size, 1))
        self.reward5 = np.zeros((max_size, 1))
        self.reward6 = np.zeros((max_size, 1))
        self.next_reward = np.zeros((max_size, 1))
        self.nn_reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.state_buffer = []
        self.action_buffer = []
        self.next_state_buffer = []
        self.next_action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []

        self.device = torch.device(device)

    # 1. Offline RL add data function: add_data_to_buffer -> convert_buffer_to_numpy_dataset -> cat_new_dataset
    def add_data_to_buffer(self, state, action, reward, done):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

    # 2. Offline RL add data function: add_data_to_buffer -> convert_buffer_to_numpy_dataset -> cat_new_dataset
    def convert_buffer_to_numpy_dataset(self):
        return np.array(self.state_buffer), \
               np.array(self.action_buffer), \
               np.array(self.reward_buffer), \
               np.array(self.done_buffer)

    # 3. Offline RL add data function: add_data_to_buffer -> convert_buffer_to_numpy_dataset -> cat_new_dataset
    def cat_new_dataset(self, dataset):
        new_state, new_action, new_reward, new_done = self.convert_buffer_to_numpy_dataset()

        state = np.concatenate([dataset['observations'], new_state], axis=0)
        action = np.concatenate([dataset['actions'], new_action], axis=0)
        reward = np.concatenate([dataset['rewards'].reshape(-1, 1), new_reward.reshape(-1, 1)], axis=0)
        done = np.concatenate([dataset['terminals'].reshape(-1, 1), new_done.reshape(-1, 1)], axis=0)

        # free the buffer when you have converted the online sample to offline dataset
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        return {
            'observations': state,
            'actions': action,
            'rewards': reward,
            'terminals': done,
        }

    # TD3 add data function
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # Offline and Online sample data from replay buffer function
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)  ###################################

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),  ####################################
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample_multiple(self, batch_size):
        """
        used for convert_D4RL_macro
        :param batch_size:
        :return:
        """
        ind = np.random.randint(0, self.size, size=batch_size)  ###################################

        return (
            torch.FloatTensor(self.s0[ind]).to(self.device),
            torch.FloatTensor(self.s1[ind]).to(self.device),
            torch.FloatTensor(self.s2[ind]).to(self.device),
            torch.FloatTensor(self.s3[ind]).to(self.device),
            torch.FloatTensor(self.s4[ind]).to(self.device),
            torch.FloatTensor(self.s5[ind]).to(self.device),
            torch.FloatTensor(self.s6[ind]).to(self.device),
            torch.FloatTensor(self.s7[ind]).to(self.device),

            torch.FloatTensor(self.a0[ind]).to(self.device),
            torch.FloatTensor(self.a1[ind]).to(self.device),
            torch.FloatTensor(self.a2[ind]).to(self.device),
            torch.FloatTensor(self.a3[ind]).to(self.device),
            torch.FloatTensor(self.a4[ind]).to(self.device),
            torch.FloatTensor(self.a5[ind]).to(self.device),
            torch.FloatTensor(self.a6[ind]).to(self.device),
            torch.FloatTensor(self.a7[ind]).to(self.device),

            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.reward1[ind]).to(self.device),
            torch.FloatTensor(self.reward2[ind]).to(self.device),
            torch.FloatTensor(self.reward3[ind]).to(self.device),
            torch.FloatTensor(self.reward4[ind]).to(self.device),
            torch.FloatTensor(self.reward5[ind]).to(self.device),
            torch.FloatTensor(self.reward6[ind]).to(self.device),
            # torch.FloatTensor(self.d1[ind]).to(self.device),
            # torch.FloatTensor(self.d2[ind]).to(self.device),
            # torch.FloatTensor(self.d3[ind]).to(self.device),
            torch.FloatTensor(self.d7[ind]).to(self.device),
        )

    def sample_lambda(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)  ###################################

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),  ####################################
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def split_dataset(self, env, dataset, terminate_on_end=False, ratio=10, toycase=False, env_name=None):
        """
            Returns datasets formatted for use by standard Q-learning algorithms,
            with observations, actions, next_observations, rewards, and a terminal
            flag.

            Args:
                env: An OfflineEnv object.
                dataset: An optional dataset to pass in for processing. If None,
                    the dataset will default to env.get_dataset()
                terminate_on_end (bool): Set done=True on the last timestep
                    in a trajectory. Default is False, and will discard the
                    last timestep in each trajectory.
                **kwargs: Arguments to pass to env.get_dataset().
                ratio=N: split the dataset into N peaces

            Returns:
                A dictionary containing keys:
                    observations: An N/ratio x dim_obs array of observations.
                    actions: An N/ratio x dim_action array of actions.
                    next_observations: An N/ratio x dim_obs array of next observations.
                    rewards: An N/ratio-dim float array of rewards.
                    terminals: An N/ratio-dim boolean array of "done" or episode termination flags.
            """
        N = dataset['rewards'].shape[0]
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []

        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        for i in range(int(N / ratio) - 1):
            if 'large' in env_name:
                if (0 <= dataset['observations'][i, 0] <= 0 and 15 <= dataset['observations'][i, 1] <= 18) \
                        or (10.5 <= dataset['observations'][i, 0] <= 21 and 7 <= dataset['observations'][i, 1] <= 9) \
                        or (0 <= dataset['observations'][i, 0] <= 0 and 6.5 <= dataset['observations'][i, 1] <= 9.5) \
                        or (19 <= dataset['observations'][i, 0] <= 29.5 and 15 <= dataset['observations'][i, 1] <= 17) \
                        and toycase:
                    # print('find a point')
                    continue
            elif 'antmaze-medium' in env_name:
                if (11.5 <= dataset['observations'][i, 0] <= 20.5 and 11 <= dataset['observations'][i, 1] <= 13) \
                        or (4 <= dataset['observations'][i, 0] <= 13 and 7 <= dataset['observations'][i, 1] <= 9) \
                        and toycase:
                    continue

            obs = dataset['observations'][i].astype(np.float32)
            new_obs = dataset['observations'][i + 1].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])

            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps - 1)
            if (not terminate_on_end) and final_timestep:
                # Skip this transition and don't apply terminals on the last step of an episode
                episode_step = 0
                continue
            if done_bool or final_timestep:
                episode_step = 0

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1

        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
        }

    def convert_D4RL(self, dataset, scale_rewards=False, scale_state=False, scale_action=False):
        """
        convert the D4RL dataset into numpy ndarray, you can select whether normalize the rewards and states
        :param scale_action:
        :param dataset: d4rl dataset, usually comes from env.get_dataset or replay_buffer.split_dataset
        :param scale_rewards: whether scale the reward to [0, 1]
        :param scale_state: whether scale the state to standard gaussian distribution ~ N(0, 1)
        :return: the mean and standard deviation of states
        """
        dataset_size = len(dataset['observations'])
        dataset['terminals'] = np.squeeze(dataset['terminals'])
        dataset['rewards'] = np.squeeze(dataset['rewards'])

        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(dataset['terminals']),
                np.arange(dataset_size) < dataset_size - 1))
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), dataset_size))

        self.state = dataset['observations'][nonterminal_steps]
        self.action = dataset['actions'][nonterminal_steps]
        self.next_state = dataset['observations'][nonterminal_steps + 1]  ####################################
        self.next_action = dataset['actions'][nonterminal_steps + 1]
        self.reward = dataset['rewards'][nonterminal_steps].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'][nonterminal_steps + 1].reshape(-1, 1)
        self.size = self.state.shape[0]

        # min_max normalization
        if scale_rewards:
            self.reward -= 1.
            # r_max = np.max(self.reward)
            # r_min = np.min(self.reward)
            # self.reward = (self.reward - r_min) / (r_max - r_min)

        s_mean = self.state.mean(0, keepdims=True)
        s_std = self.state.std(0, keepdims=True)

        s_min = self.state.min(0, keepdims=True)
        s_max = self.state.max(0, keepdims=True)

        a_mean = self.action.mean(0, keepdims=True)
        a_std = self.action.std(0, keepdims=True)

        if scale_state == 'minmax':
            # min_max normalization
            self.state = (self.state - s_min) / (s_max - s_min)
            self.next_state = (self.next_state - s_min) / (s_max - s_min)
            if scale_action:
                self.action = (self.action - a_mean) / (a_std + 1e-3)
                self.next_action = (self.next_action - a_mean) / (a_std + 1e-3)
                return s_min, s_max, a_mean, a_std
            else:
                return s_min, s_max

        elif scale_state == 'standard':
            # standard normalization
            self.state = (self.state - s_mean) / (s_std + 1e-3)
            self.next_state = (self.next_state - s_mean) / (s_std + 1e-3)
            if scale_action:
                self.action = (self.action - a_mean) / (a_std + 1e-3)
                self.next_action = (self.next_action - a_mean) / (a_std + 1e-3)
                a_max = self.action.max(0, keepdims=True)
                a_min = self.action.min(0, keepdims=True)
                return s_mean, s_std, a_mean, a_std, a_max, a_min
            else:
                return s_mean, s_std

        else:
            if scale_action:
                self.action = (self.action - a_mean) / (a_std + 1e-3)
                self.next_action = (self.next_action - a_mean) / (a_std + 1e-3)
                return s_mean, s_std, a_mean, a_std
            else:
                return s_mean, s_std

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std
