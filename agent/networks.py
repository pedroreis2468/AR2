"""
Redes Actor-Critic para SAC.
Actor: MLP → distribuição Gaussiana squashed (tanh)
Critic: Twin Q-networks (clipped double Q-learning)
Arquitetura: 2 camadas ocultas de 256 unidades (validada pela equipa Auckland FS).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLP(nn.Module):
    """MLP base com 2 camadas ocultas."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: tuple = (256, 256)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianActor(nn.Module):
    """
    Actor que produz uma distribuição Gaussiana squashed.
    Output: ação em [-1, 1] via tanh.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dims: tuple = (256, 256)):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        """
        Amostra ação com reparametrization trick.
        Retorna: (ação squashed, log_prob, mean)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparametrization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log-probability com correção de Jacobiano para tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class TwinQCritic(nn.Module):
    """
    Twin Q-networks para Clipped Double Q-learning.
    Recebe (obs, action) e produz dois Q-values.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dims: tuple = (256, 256)):
        super().__init__()
        self.q1 = MLP(obs_dim + action_dim, 1, hidden_dims)
        self.q2 = MLP(obs_dim + action_dim, 1, hidden_dims)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)
