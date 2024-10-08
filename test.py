from ppo_agent import *
num_envs = 4
env = gym.vector.make("CartPole-v1", num_envs=num_envs)
agent = Agent(envs=env)

trainer = PPOTrainer(
    env =env,
    agent = agent,
    rollout_num = 100000,
    rollout_length = 32,
    epsilon = 0.2,
    value_loss_weight = 0.2,
    clip_norm = 0.5,
    norm_adv = True,
    ppo_epoch = 4,
    gamma = 1.0,
    use_gae = True,
    lam = 0.95
)
trainer.train()