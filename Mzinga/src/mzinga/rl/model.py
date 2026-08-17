from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.norm(self.fc(x)))


class HivePolicyValue(nn.Module):
    def __init__(
        self,
        obs_dim: int = 88,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_actions: int = 2048,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.input_norm = nn.LayerNorm(obs_dim)
        self.fc_in = nn.Linear(obs_dim, hidden_dim)

        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc_in(self.input_norm(obs)))
        x = self.blocks(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)

    def act(
        self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits, value = self.forward(obs.unsqueeze(0))
            logits = logits.squeeze(0)
            logits = logits.masked_fill(~mask, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = int(torch.argmax(probs))
            else:
                action = int(torch.multinomial(probs, 1).item())
            log_prob = F.log_softmax(logits, dim=-1)[action]

        return action, log_prob, value

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
