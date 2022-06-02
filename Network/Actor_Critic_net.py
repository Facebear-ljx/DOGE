import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal, Normal

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 10
EPS = 1e-7


class BC(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        """
        Deterministic Behavior cloning policy
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param device: cuda or cpu
        """
        super(BC, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)

    def forward(self, x):
        """
        :param x: state
        :return: action
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu_head = self.mu_head(x)
        return mu_head


class BC_standard(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        """
        Stochastic Behavior cloning policy
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param device: cuda or cpu
        """
        super(BC_standard, self).__init__()
        self.device = device
        # self.bn0 = nn.BatchNorm1d(num_state)
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

        self._action_means = torch.tensor(0, dtype=torch.float32).to(self.device)
        self._action_mags = torch.tensor(1, dtype=torch.float32).to(self.device)

    def get_log_density(self, x, y):
        """
        calculate the log(probability) of the action conditioned on the state
        :param x: state
        :param y: action
        :return: log(P(action|state))
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        # x = self.bn0(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = torch.clamp(y, -1. + EPS, 1. - EPS)
        y = torch.atanh(y)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        logp_pi = a_distribution.log_prob(y).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - y - F.softplus(-2 * y))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)
        return logp_pi

    def get_action(self, x):
        """
        sample an action from the distribution
        :param x: state
        :return: action
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        # x = self.bn0(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        action = torch.tanh(action)
        return action


# Conditional VAE
class BC_VAE(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, num_latent, device):
        """
        Conditional VAE behavior cloning, used for BCQ
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param num_latent: number of the latent variables
        :param device: cuda or cpu
        """
        super(BC_VAE, self).__init__()
        self.latent_dim = num_latent
        self.device = device
        # encode
        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mean = nn.Linear(num_hidden, num_latent)  # mean(z)
        self.sigma = nn.Linear(num_hidden, num_latent)  # sigma(z)

        # decode
        self.fc4 = nn.Linear(num_latent + num_state, num_hidden)
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.action = nn.Linear(num_hidden, num_action)

    def encode(self, state, action):
        """
        encode the state and action into latent space as a gaussian distribution ~ N(mean, sigma)
        :param state:
        :param action:
        :return: Z~N(mean, sigma)
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_sigma = self.sigma(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_sigma

    def reparameterize(self, mu, log_sigma):
        """
        reparameterize trick, sample a latent variable according to N(mu, sigma)
        :param mu:
        :param log_sigma:
        :return: a latent variable:z
        """
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, state, z=None):
        """
        decode the latent variable to action conditioned on state, if the z is missed, z is randomly sampled from
        standard Normal distribution and clipped to [-0.5, 0.5]
        :param state:
        :param z: latent variable
        :return: action
        """
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5], copy from BCQ's implementation
        # https://github.com/sfujim/BCQ/blob/master/continuous_BCQ/BCQ.py

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        x = torch.cat([z, state], dim=1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        action_recon = self.action(x)
        action_recon = torch.tanh(action_recon)
        return action_recon

    def decode_multiple(self, state, num_decode=10, z=None):
        """
        decode multiple actions, used in BEAR to generate 10 actions to do the MMD calculation
        :param num_decode: the number of the decoded action
        :param state:
        :param z:
        :return: num_decode actions
        """
        if z is None:
            z = torch.randn((state.shape[0], num_decode, self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        a = F.relu(self.fc4(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.fc5(a))
        a = self.action(a)
        return torch.tanh(a), a

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float)

        mean, sigma = self.encode(state, action)
        latent_pi = self.reparameterize(mean, sigma)
        action_recon = self.decode(state, latent_pi)
        return action_recon, mean, sigma


class Actor(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        """
        Stochastic Actor network, used for SAC, SBAC, BEAR, CQL
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param device: cuda or cpu
        """
        super(Actor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()

        logp_pi = a_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)

        action = torch.tanh(action)
        return action, logp_pi, a_distribution

    def get_log_density(self, x, y):
        """
        calculate the log probability of the action conditioned on state
        :param x: state
        :param y: action
        :return: log(P(action|state))
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = torch.clip(y, -1. + EPS, 1. - EPS)
        y = torch.atanh(y)

        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        logp_pi = a_distribution.log_prob(y).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - y - F.softplus(-2 * y))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)
        return logp_pi

    def get_action(self, x):
        """
        generate actions according to the state
        :param x: state
        :return: action
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        action = torch.tanh(action)
        return action

    def get_action_multiple(self, x, num_samples=10):
        """
        used in BEAR
        :param x:
        :param num_samples:
        :return:
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = x.unsqueeze(0).repeat(num_samples, 1, 1).permute(1, 0, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        return torch.tanh(action), action


class Actor_deterministic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device, max_action=1.):
        """
        Deterministic Actor network, used for TD3, DDPG, BCQ, TD3_BC
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param device: cuda or cpu
        """
        super(Actor_deterministic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.action = nn.Linear(num_hidden, num_action)
        self.max_action = max_action

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        a = F.relu(self.fc1(x))
        a = F.relu(self.fc2(a))
        a = self.action(a)
        return torch.tanh(a) * self.max_action


class Double_Critic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        """
        Double Q network, used for TD3_BC, BCQ
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param device: cuda or cpu
        """
        super(Double_Critic, self).__init__()
        self.device = device

        # Q1 architecture
        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(num_state + num_action, num_hidden)
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.fc6 = nn.Linear(num_hidden, 1)

    def forward(self, x, y):
        sa = torch.cat([x, y], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


# Ensemble Q_net
class Ensemble_Critic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, num_q, device):
        """
        Ensemble Q network, the number of the Q network is no more than 4.
        Used in Bear and TD3_BC_Unc
        :param num_state: dimension of state
        :param num_action: dimension of action
        :param num_hidden: dimension of the hidden layer
        :param num_q: number of ensembled Q network
        :param device: cuda or cpu
        """
        super(Ensemble_Critic, self).__init__()
        self.device = device
        self.num_q = num_q

        # Q1 architecture
        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(num_state + num_action, num_hidden)
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.fc6 = nn.Linear(num_hidden, 1)

        # Q3 architecture
        self.fc7 = nn.Linear(num_state + num_action, num_hidden)
        self.fc8 = nn.Linear(num_hidden, num_hidden)
        self.fc9 = nn.Linear(num_hidden, 1)

        # Q4 architecture
        self.fc10 = nn.Linear(num_state + num_action, num_hidden)
        self.fc11 = nn.Linear(num_hidden, num_hidden)
        self.fc12 = nn.Linear(num_hidden, 1)

    def forward(self, x, y, with_var=False):
        """
        :param x: state
        :param y: action
        :param with_var: whether output the variance of Q networks
        :return: Q value of all the Q networks. if with_var=True, output the variance too.
        """
        sa = torch.cat([x, y], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        if self.num_q >= 3:
            q3 = F.relu(self.fc7(sa))
            q3 = F.relu(self.fc8(q3))
            q3 = self.fc9(q3)

        if self.num_q >= 4:
            q4 = F.relu(self.fc10(sa))
            q4 = F.relu(self.fc11(q4))
            q4 = self.fc12(q4)

        if self.num_q == 2:
            all_q = torch.cat([q1.unsqueeze(0), q2.unsqueeze(0)], 0)
        elif self.num_q == 3:
            all_q = torch.cat([q1.unsqueeze(0), q2.unsqueeze(0), q3.unsqueeze(0)], 0)
        elif self.num_q == 4:
            all_q = torch.cat([q1.unsqueeze(0), q2.unsqueeze(0), q3.unsqueeze(0), q4.unsqueeze(0)], 0)  # num_q x B x 1
        else:
            print("wrong num_q!!!! should in range[2,4]")

        if with_var:
            std_q = torch.std(all_q, dim=0, keepdim=False, unbiased=False)
            return all_q, std_q

        return all_q

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1

    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q2

    def Q3(self, state, action):
        sa = torch.cat([state, action], 1)

        q3 = F.relu(self.fc7(sa))
        q3 = F.relu(self.fc8(q3))
        q3 = self.fc9(q3)
        return q3

    def Q4(self, state, action):
        sa = torch.cat([state, action], 1)

        q4 = F.relu(self.fc10(sa))
        q4 = F.relu(self.fc11(q4))
        q4 = self.fc12(q4)
        return q4


class V_critic(nn.Module):
    def __init__(self, num_state, num_hidden, device):
        super(V_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.state_value(x)
        return v


class Q_critic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        super(Q_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q


class Alpha(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        super(Alpha, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.alpha = nn.Linear(num_hidden, 1)

    def forward(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.fc1(x))
        alpha = F.softplus(self.alpha(x))
        return alpha


class W(nn.Module):
    def __init__(self, num_state, num_hidden, device):
        super(W, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softplus(self.fc3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, dim_state, dim_action, dim_hidden, device):
        super(Encoder, self).__init__()
        self.device = device
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.en1 = nn.Linear(dim_state+dim_action, dim_hidden)
        self.en2 = nn.Linear(dim_hidden, dim_hidden)
        self.en3 = nn.Linear(dim_hidden, dim_state+dim_action)

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(state, dtype=torch.float).to(self.device)
        state_action = torch.cat([state, action], dim=1)
        x = F.relu(self.en1(state_action))
        x = F.relu(self.en2(x))
        latent = self.en3(x)
        latent_s = latent[:, 0:self.dim_state]
        latent_a = latent[:, self.dim_state:self.dim_state+self.dim_action]
        return latent_s, latent_a


class Distance(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device, batch_size, negative_samples, negative_policy=10, a_max=1, a_min=-1):
        super(Distance, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_action = num_action
        self.num_state = num_state
        self.energy_scale = torch.tensor(100.)
        self.negative_policy = negative_policy
        self.negative_samples = negative_samples
        self.negative_samples_w_policy = int(negative_samples / 2 + negative_policy)

        self.a_min = a_min
        self.a_scale = a_max - a_min

        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, 1)

    def value(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        sa = torch.cat([x, y], dim=1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value

    def linear_distance(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)

        state = x.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        state = state.view(self.batch_size * self.negative_samples, self.num_state)

        action = y.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        action = action.view(self.batch_size * self.negative_samples, self.num_action)

        noise_action = ((torch.rand([self.batch_size * self.negative_samples, self.num_action]) - 0.5) * 3).to(self.device)
        noise = noise_action - action
        norm = torch.norm(noise, dim=1, keepdim=True)

        output = self.value(state, noise_action)
        label = norm
        return output, label

    def distance(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)

        state = x.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        state = state.view(self.batch_size * self.negative_samples, self.num_state)

        action = y.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        action = action.view(self.batch_size * self.negative_samples, self.num_action)

        noise = (torch.randn([self.batch_size * self.negative_samples, self.num_action])*0.5).to(self.device)
        norm = torch.norm(noise, dim=1, keepdim=True)

        noise_action = noise + action
        output = self.value(state, noise_action)
        label = F.softplus(norm * 2) * 10
        return output, label

    def forward(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        Positive_E = -self.value(x, y)
        Positive = torch.exp(Positive_E)

        state = x.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        state = state.view(self.batch_size * self.negative_samples, self.num_state)
        noise_action = ((torch.rand([self.batch_size * self.negative_samples, self.num_action]) - 0.5) * 3).to(self.device)  # wtf, noise scale should be larger than the action range
        # noise_action = (torch.rand([self.batch_size * self.negative_samples, self.num_action]) * self.a_scale * 1.05 + self.a_min).to(
        #     self.device)
        # noise_action = (torch.rand([self.batch_size * self.negative_samples, self.num_action]) * (self.a_max - self.a_min)).to(self.device)  ########################

        # noise_action = (torch.ones([self.batch_size * self.negative_samples, self.num_action])).to(self.device)
        Negative_E = -self.value(state, noise_action)
        Negative = torch.exp(Negative_E).view(self.negative_samples, self.batch_size, 1).sum(0)
        # Negative = torch.sum(Negative, dim=1, keepdim=False)
        # Negative = Negative.sum(0)

        out = Positive / (Positive + Negative)
        return out

    def get_positive_log(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        Positive_E = -self.value(x, y)
        return Positive_E

    def get_negative_log(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)

        state = x.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        state = state.view(self.batch_size * self.negative_samples, self.num_state)
        noise_action = ((torch.rand([self.batch_size * self.negative_samples, self.num_action]) - 0.5) * 3).to(self.device)  # wtf, noise scale should be larger than the action range
        Negative_E = -self.value(state, noise_action)
        Negative = torch.exp(Negative_E).view(self.negative_samples, self.batch_size, 1).sum(0)
        Negative_log = torch.log(Negative + 1e-5)
        return Negative_log
