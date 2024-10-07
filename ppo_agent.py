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
                 ppo_num_minibatch : int = 4,
                 gamma : float = 1.0,
                 use_gae : bool = True,
                 lam : float = 0.95):
        self.rollout_num = rollout_num
        self.rollout_length = rollout_length 
        self.env = env
        self.device = 0
        self.agent = agent.to(self.device)
        self.gamma = gamma
        self.num_env = self.env.num_envs
        self.epsilon = epsilon
        self.ppo_num_minibatch = ppo_num_minibatch
        self.ppo_batch_size = (self.num_env * self.rollout_length) // self.ppo_num_minibatch
        self.indice_dataset = TensorDataset(torch.arange(self.num_env*self.rollout_length))
        self.value_loss_weight = value_loss_weight
        self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=2e-4)
        self.clip_norm = clip_norm
        self.ppo_epoch = ppo_epoch
        self.norm_adv = norm_adv
        self.use_gae = use_gae
        self.lam = lam
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

        with torch.no_grad():
            # n step estimation here
            # we ignore truncation here and treat it as termination
            rollout_obs = torch.zeros((self.rollout_length, self.num_env) + self.env.single_observation_space.shape, device=self.device)
            rewards = torch.zeros((self.rollout_length, self.num_env), device=self.device)
            action_log_probs = torch.zeros((self.rollout_length, self.num_env), device=self.device)
            actions = torch.zeros((self.rollout_length, self.num_env) + self.env.single_action_space.shape, device=self.device, dtype=torch.long)
            state_values = torch.zeros((self.rollout_length, self.num_env), device=self.device)
            terminateds = torch.zeros((self.rollout_length, self.num_env), device=self.device, dtype=torch.bool)
            truncateds = torch.zeros((self.rollout_length, self.num_env), device=self.device, dtype=torch.bool)
            G_ts = torch.zeros_like(state_values)
            A_ts = torch.zeros_like(G_ts)

        for n_rollout in range(self.rollout_num):
            with torch.no_grad():
                self.agent.eval()
                for i in range(0, self.rollout_length):
                    rollout_obs[i] = obs
                    action, action_log_prob, entropy, state_value = self.agent.get_action_and_value(obs, action=None)
                    actions[i] = action
                    action_log_probs[i] = action_log_prob             
                    state_values[i] = state_value
                    next_obs, reward, terminated, truncated, _ = self.env.step(action.tolist())
                    rewards[i] = torch.tensor(reward, device=self.device)
                    terminateds[i] = torch.tensor(terminated, device=self.device)
                    truncateds[i] = torch.tensor(truncated, device=self.device)
                    obs = next_obs
                    obs = torch.tensor(obs, device=self.device)

                next_state_value = self.agent.get_action_and_value(obs, action=None)[-1]
                
                if self.use_gae:
                    lastgaelam = 0
                    for i in range(self.rollout_length -1, -1, -1):
                        truncated_gamma = torch.where(terminateds[i] | truncateds[i], 0, self.gamma)
                        delta_t = rewards[i] + truncated_gamma * next_state_value - state_values[i]
                        A_ts[i] = lastgaelam = delta_t + truncated_gamma * self.lam * lastgaelam
                        next_state_value = state_values[i]
                    G_ts = A_ts + state_values
                else:
                    # rollout_obs[i] -> s_i
                    # rewards[i] -> r_i
                    # terminated[i] -> whether s_i -> s_i+1 results in termination
                    G_t = next_state_value
                    for i in range(self.rollout_length -1, -1, -1):
                        # if truncated or done, we set gamma to zero
                        truncated_gamma = torch.where(terminateds[i] | truncateds[i], 0, self.gamma)
                        G_ts[i] = G_t = G_t * truncated_gamma + rewards[i]
                    A_ts = G_ts - state_values
                    
            # without TD(lambda)
            assert G_ts.shape == state_values.shape
            dataloader = DataLoader(self.indice_dataset, batch_size = self.ppo_batch_size, shuffle=True)
            self.agent.train()
            for i in range(self.ppo_epoch):
                for indices in dataloader:
                    rollout_obs_batch = rollout_obs.view((self.rollout_length*self.num_env,) + self.env.single_observation_space.shape)[indices]
                    actions_batch = actions.view((self.rollout_length*self.num_env,) + self.env.single_action_space.shape)[indices]
                    G_ts_batch = G_ts.view(-1)[indices]
                    action_log_probs_batch = action_log_probs.view(-1)[indices]
                    A_ts_batch = A_ts.view(-1)[indices]
                    _, new_action_log_prob_batch, _, new_state_value  = self.agent.get_action_and_value(rollout_obs_batch, action=actions_batch)
                    if self.norm_adv:
                        A_ts_batch = (A_ts_batch - A_ts_batch.mean()) / (A_ts_batch.std() + 1e-8)
                    
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



            

