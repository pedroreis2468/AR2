"""
Soft Actor-Critic (SAC) com auto-tuning de temperatura.
Referência: Haarnoja et al. (2018) - Soft Actor-Critic Algorithms and Applications

Implementação custom em PyTorch para fins educativos.
Para produção, usar stable-baselines3 (ver train.py).
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import Optional

from .networks import GaussianActor, TwinQCritic
from .replay_buffer import ReplayBuffer


class SACAgent:
    """
    Agente SAC com:
    - Twin Q-critics (clipped double Q-learning)
    - Actor Gaussiano squashed
    - Auto-tuning do coeficiente de entropia (α)
    - Soft target updates (Polyak averaging)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple = (256, 256),
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 5000,
        target_entropy: Optional[float] = None,
        device: str = 'auto',
    ):
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        # --- Redes ---
        self.actor = GaussianActor(obs_dim, action_dim, hidden_dims).to(self.device)
        self.critic = TwinQCritic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = TwinQCritic(obs_dim, action_dim, hidden_dims).to(self.device)

        # Copiar pesos para target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- Otimizadores ---
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # --- Auto-tuning de α (temperatura de entropia) ---
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        # --- Replay Buffer ---
        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)

        # Contadores
        self.total_steps = 0
        self.update_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Seleciona ação dado uma observação."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(obs_t)
            else:
                action, _, _ = self.actor.sample(obs_t)

        return action.cpu().numpy().flatten()

    def store_transition(self, obs, action, reward, next_obs, done):
        """Armazena transição no replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def update(self) -> dict:
        """
        Atualiza actor, critic e α.
        Retorna métricas de treino.
        """
        if len(self.buffer) < self.learning_starts:
            return {}

        # Amostrar batch
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ---- Atualizar Critic ----
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs_t)
            q1_target, q2_target = self.critic_target(next_obs_t, next_actions)
            q_target = torch.min(q1_target, q2_target)
            td_target = rewards_t + (1 - dones_t) * self.gamma * (
                q_target - self.alpha * next_log_probs
            )

        q1, q2 = self.critic(obs_t, actions_t)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- Atualizar Actor ----
        new_actions, log_probs, _ = self.actor.sample(obs_t)
        q1_new, q2_new = self.critic(obs_t, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---- Atualizar α ----
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- Soft update targets ----
        self._soft_update()

        self.update_count += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q_mean': q_new.mean().item(),
        }

    def _soft_update(self):
        """Polyak averaging para target networks."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """Guarda o modelo."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha.data,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
        }, path)
        print(f"[SAC] Modelo guardado em {path}")

    def load(self, path: str):
        """Carrega o modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha.data = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.update_count = checkpoint['update_count']
        print(f"[SAC] Modelo carregado de {path} (step {self.total_steps})")
