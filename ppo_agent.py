import gymnasium as gym
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.utils
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x).squeeze(-1)



class PPOTrainer:
    def __init__(self, 
                 env : gym.Env,
                 agent : Agent,
                 rollout_num : int,
                 rollout_length : int,
                 epsilon : float = 0.2,
                 value_loss_weight : float = 0.2,
                 clip_norm : float = 0.5,
                 norm_adv : bool = True,
                 ppo_epoch : int = 4,
                 gamma : float = 1.0):
        self.rollout_num = rollout_num
        self.rollout_length = rollout_length 
        self.env = env
        self.device = 0
        self.agent = agent.to(self.device)
        self.gamma = gamma
        self.num_env = self.env.num_envs
        self.epsilon = epsilon
        self.ppo_batch_size = 4
        self.indice_dataset = TensorDataset(torch.arange(self.num_env*self.rollout_length))
        self.value_loss_weight = value_loss_weight
        self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=2e-4)
        self.clip_norm = clip_norm
        self.ppo_epoch = ppo_epoch
        self.norm_adv = norm_adv
    @torch.no_grad
    def validate(self):
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, device=self.device)
        rewards = []
        with torch.no_grad():
            self.agent.eval()
            terminateds = []
            truncateds = []
            sampling_length = 503
            for _ in range(0, sampling_length):
                action, action_log_prob, entropy, state_value = self.agent.get_action_and_value(obs, action=None)
                next_obs, reward, terminated, truncated, _ = self.env.step(action.tolist())
                rewards.append(torch.tensor(reward, device=self.device, dtype=torch.float32))
                terminateds.append(torch.tensor(terminated, device=self.device))
                truncateds.append(torch.tensor(truncated, device=self.device))
                obs = next_obs
                obs = torch.tensor(obs, device=self.device)
            G_ts = [-1 for _ in range(sampling_length)]
            G_t = torch.zeros(self.num_env, device=self.device)
            for i in range(sampling_length -1, -1, -1):
                # if truncated or done, we set gamma to zero
                truncated_gamma = torch.where(terminateds[i] | truncateds[i], 0, self.gamma)
                G_t = G_t * truncated_gamma + rewards[i]
                G_ts[i] = G_t
            return G_ts[0].mean()
            
    def train(self):
        writer = SummaryWriter()
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, device=self.device)
        for n_rollout in range(self.rollout_num):
            # n step estimation here
            # we ignore truncation here and treat it as termination
            rollout_obs = []
            rewards = []
            action_log_probs = []
            actions = []
            state_values = []
            
            terminateds = []
            truncateds = []
            with torch.no_grad():
                self.agent.eval()
                for _ in range(0, self.rollout_length):
                    rollout_obs.append(obs)
                    action, action_log_prob, entropy, state_value = self.agent.get_action_and_value(obs, action=None)
                    actions.append(action)
                    action_log_probs.append(action_log_prob)                
                    state_values.append(state_value)
                    next_obs, reward, terminated, truncated, _ = self.env.step(action.tolist())
                    rewards.append(torch.tensor(reward, device=self.device, dtype=torch.float32))
                    terminateds.append(torch.tensor(terminated, device=self.device))
                    truncateds.append(torch.tensor(truncated, device=self.device))
                    obs = next_obs
                    obs = torch.tensor(obs, device=self.device)
            G_t = self.agent.get_action_and_value(obs, action=None)
            G_ts = [-1 for _ in range(self.rollout_length)]
            # rollout_obs[i] -> s_i
            # rewards[i] -> r_i
            # terminated[i] -> whether s_i -> s_i+1 results in termination
            for i in range(self.rollout_length -1, -1, -1):
                # if truncated or done, we set gamma to zero
                truncated_gamma = torch.where(terminateds[i] | truncateds[i], 0, self.gamma)
                G_t = G_t * truncated_gamma + rewards[i]
                G_ts[i] = G_t
            
            rollout_obs = torch.cat(rollout_obs, dim=0)
            actions = torch.cat(actions, dim=0)
            action_log_probs = torch.cat(action_log_probs, dim = 0)
            G_ts = torch.cat(G_ts, dim=0)
            state_values = torch.cat(state_values, dim=0)
            # without TD(lambda)
            assert G_ts.shape == state_values.shape
            A_ts = G_ts - state_values

            dataloader = DataLoader(self.indice_dataset, batch_size = self.ppo_batch_size, shuffle=True)
            self.agent.train()
            for i in range(self.ppo_epoch):
                for indices in dataloader:
                    rollout_obs_batch = rollout_obs[indices]
                    actions_batch = actions[indices]
                    _, new_action_log_prob_batch, _, new_state_value  = self.agent.get_action_and_value(rollout_obs_batch, action=actions_batch)
                    A_ts_batch = A_ts[indices]
                    if self.norm_adv:
                        A_ts_batch = (A_ts_batch - A_ts_batch.mean()) / (A_ts_batch.std() + 1e-8)
                    G_ts_batch = G_ts[indices]
                    action_log_probs_batch = action_log_probs[indices]
                    ratio = torch.exp(new_action_log_prob_batch - action_log_probs_batch)
                    L_pi = torch.min(ratio * A_ts_batch, torch.clip(ratio, 1-self.epsilon, 1+self.epsilon) * A_ts_batch)
                    L_pi = L_pi.mean()
                    # todo : mse clipping
                    # TODO : this is incorrect, we need to fix this
                    L_V = torch.nn.MSELoss()(new_state_value, G_ts_batch)
                    # TODO : entropy loss
                    L = - L_pi + self.value_loss_weight * L_V 
                    self.agent_optimizer.zero_grad()
                    L.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_norm)
                    self.agent_optimizer.step()
            mean_value = self.validate()
            writer.add_scalar("losses/value_loss", L_V.item(), n_rollout)
            writer.add_scalar("losses/policy_loss", L_pi.item(), n_rollout)
            writer.add_scalar("return", mean_value.item(), n_rollout)
            print(mean_value)



            

