import gymnasium as gym
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.utils
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import numpy as np

# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer

def layer_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
    return m

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Dropout(dropout),
            nn.PReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Dropout(dropout),
            nn.PReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Dropout(dropout),
            nn.PReLU(),
            layer_init(nn.Linear(hidden_dim, output_dim))
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
        
class Agent(nn.Module):
    def __init__(self, env, dropout):
        super().__init__()
        self.critic = MLP(np.array(env.observation_space.shape).prod(), 128, 1, dropout)
        self.actor =  MLP(np.array(env.observation_space.shape).prod(), 128, env.action_space.n, dropout)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.float().flatten(start_dim=1)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x).squeeze(-1)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                 norm_return : bool = False,
                 use_gae : bool = True,
                 lam : float = 0.95):
        self.rollout_num = rollout_num
        #self.rollout_length = rollout_length 
        self.env = env
        self.device = 0
        self.agent = agent.to(self.device)
        self.gamma = gamma
        # only one environment
        self.num_env = 1
        self.epsilon = epsilon
        # self.ppo_num_minibatch = ppo_num_minibatch
        # self.ppo_batch_size = (self.num_env * self.rollout_length) // self.ppo_num_minibatch
        #self.indice_dataset = TensorDataset(torch.arange(self.num_env*self.rollout_length))
        self.value_loss_weight = value_loss_weight
        self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=5e-4)
        self.clip_norm = clip_norm
        self.ppo_epoch = ppo_epoch
        self.norm_adv = norm_adv
        self.use_gae = use_gae
        self.norm_return = norm_return
        self.lam = lam
    @torch.no_grad
    def validate(self):
        # this returns undiscounted reward
        obs, _ = self.env.reset()
        # add a batch dim
        obs = torch.tensor(obs, device=self.device).unsqueeze(0)
        rewards = []
        with torch.no_grad():
            self.agent.eval()
            sampling_length = 1000
            total_reward = 0
            terminated = truncated = False 
            while not (terminated or truncated):
                action, action_log_prob, entropy, state_value = self.agent.get_action_and_value(obs, action=None)
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                # # we disable truncation...
                # assert not np.any(truncated)
                total_reward += reward
                obs = next_obs
                obs = torch.tensor(obs, device=self.device).unsqueeze(0)
            return total_reward
            
    def train(self):
        writer = SummaryWriter()
        seed_everything(1234)
        N_TRIALS = 25
        train_rewards = []
        valid_rewards = []
        for n_rollout in range(self.rollout_num):
            rollout_obs = []
            actions = []
            action_log_probs = []
            state_values = []
            rewards = []
            obs, _ = self.env.reset(seed=n_rollout)
            terminated = truncated = False
            with torch.no_grad():
                self.agent.train()
                # run until the end of episode
                total_reward = 0
                while not (terminated or truncated):
                    obs = torch.tensor(obs, device=self.device)
                    rollout_obs.append(obs)
                    action, action_log_prob, entropy, state_value = self.agent.get_action_and_value(obs.unsqueeze(0), action=None)
                    actions.append(action.item())
                    action_log_probs.append(action_log_prob.item())          
                    state_values.append(state_value.item())
                    obs, reward, terminated, truncated, _ = self.env.step(action.item())
                    total_reward += reward
                    # # we disable truncation...
                    # assert not np.any(truncated)
                    rewards.append(reward)   
                
                train_rewards.append(total_reward)
                
                rollout_obs = torch.stack(rollout_obs, dim=0)
                actions = torch.tensor(actions, device=self.device)
                action_log_probs = torch.tensor(action_log_probs, device=self.device)
                rewards = torch.tensor(rewards, device=self.device)
                state_values = torch.tensor(state_values, device=self.device)
                G_ts = torch.zeros_like(state_values)
                A_ts = torch.zeros_like(G_ts)
                # run to completion so it's zero
                next_state_value = 0
                if self.use_gae:
                    lastgaelam = 0
                    for i in range(rollout_obs.size(0) -1, -1, -1):
                        delta_t = rewards[i] + self.gamma * next_state_value - state_values[i]
                        A_ts[i] = lastgaelam = delta_t +  self.gamma * self.lam * lastgaelam
                        next_state_value = state_values[i]
                    G_ts = A_ts + state_values
                else:
                    # rollout_obs[i] -> s_i
                    # rewards[i] -> r_i
                    # terminated[i] -> whether s_i -> s_i+1 results in termination
                    G_t = next_state_value
                    for i in range(rollout_obs.size(0) -1, -1, -1):
                        G_ts[i] = G_t = G_t * self.gamma + rewards[i]
                    if self.norm_return:
                        G_ts = (G_ts - G_ts.mean()) / (G_ts.std() + 1e-8)
                    A_ts = G_ts - state_values
                    
            self.agent.train()
            for i in range(self.ppo_epoch):
                rollout_obs_batch = rollout_obs.detach()
                actions_batch = actions.detach()
                G_ts_batch = G_ts.detach()
                action_log_probs_batch = action_log_probs.detach()
                A_ts_batch = A_ts.detach()
                _, new_action_log_prob_batch, _, new_state_value  = self.agent.get_action_and_value(rollout_obs_batch, action=actions_batch)
                if self.norm_adv:
                    A_ts_batch = (A_ts_batch - A_ts_batch.mean()) / (A_ts_batch.std() + 1e-8)
                
                ratio = torch.exp(new_action_log_prob_batch - action_log_probs_batch)
                L_pi = torch.min(ratio * A_ts_batch, torch.clip(ratio, 1-self.epsilon, 1+self.epsilon) * A_ts_batch)
                L_pi = L_pi.mean()
                # todo : mse clipping
                # TODO : this is incorrect, we need to fix this
                # use the smooth l1 loss
                L_V = F.smooth_l1_loss(new_state_value, G_ts_batch).mean()
                # TODO : entropy loss
                self.agent_optimizer.zero_grad()
                (-L_pi).backward()
                L_V.backward()
                # torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_norm)
                self.agent_optimizer.step()
            
            writer.add_scalar("losses/value_loss", L_V.item(), n_rollout)
            writer.add_scalar("losses/policy_loss", L_pi.item(), n_rollout)
            #writer.add_scalar("return", mean_value.item(), n_rollout)
            if n_rollout % 10 == 0:
                valid_rewards.append(self.validate())
                print(n_rollout, np.mean(np.array(train_rewards[-N_TRIALS:])), np.mean(np.array(valid_rewards[-N_TRIALS:])))



            

