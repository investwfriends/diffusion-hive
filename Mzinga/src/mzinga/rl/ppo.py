from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mzinga.rl.model import HivePolicyValue


MAX_ACTIONS = 2048


class RolloutBuffer:
    def __init__(self, obs_dim: int, capacity: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros((capacity, MAX_ACTIONS), dtype=bool)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.capacity = capacity

    def add(self, obs, action, log_prob, value, reward, mask, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.rewards[self.pos] = reward
        self.masks[self.pos] = mask
        self.dones[self.pos] = float(done)
        self.pos += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        gae = 0.0
        for t in reversed(range(self.pos)):
            next_val = 0.0 if self.dones[t] else self.values[t + 1] if t + 1 < self.pos else last_value
            delta = self.rewards[t] + gamma * next_val - self.values[t]
            gae = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]
        self.advantages[: self.pos] = (self.advantages[: self.pos] - self.advantages[: self.pos].mean()) / (
            self.advantages[: self.pos].std() + 1e-8
        )

    def get_batches(self, batch_size: int, device: torch.device):
        indices = np.random.permutation(self.pos)
        for start in range(0, self.pos, batch_size):
            batch_idx = indices[start: start + batch_size]
            yield (
                torch.as_tensor(self.obs[batch_idx], device=device),
                torch.as_tensor(self.actions[batch_idx], device=device),
                torch.as_tensor(self.log_probs[batch_idx], device=device),
                torch.as_tensor(self.advantages[batch_idx], device=device),
                torch.as_tensor(self.returns[batch_idx], device=device),
                torch.as_tensor(self.masks[batch_idx], device=device),
            )

    def clear(self):
        self.pos = 0

    def __len__(self):
        return self.pos


def ppo_update(
    model: HivePolicyValue,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    batch_size: int,
    n_epochs: int,
    clip_epsilon: float,
    ent_coef: float,
    vf_coef: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_batches = 0

    for _ in range(n_epochs):
        for obs_b, act_b, old_logp_b, adv_b, ret_b, mask_b in buffer.get_batches(batch_size, device):
            logits, values = model(obs_b)

            logits_masked = logits.masked_fill(~mask_b, float("-inf"))
            has_valid = mask_b.any(dim=-1)
            if not has_valid.all():
                logits_masked[~has_valid] = 0.0
            log_probs_all = F.log_softmax(logits_masked, dim=-1)
            probs = F.softmax(logits_masked, dim=-1)

            new_log_prob = log_probs_all[torch.arange(len(act_b)), act_b]

            ratio = torch.exp(new_log_prob - old_logp_b)

            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, ret_b)

            entropy = -(probs * log_probs_all.masked_fill(~mask_b, 0.0)).sum(-1).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_batches += 1

    return {
        "policy_loss": total_policy_loss / n_batches,
        "value_loss": total_value_loss / n_batches,
        "entropy": total_entropy / n_batches,
    }
