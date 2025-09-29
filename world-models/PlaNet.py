"""
mostly implemented this from this repo: https://github.com/cross32768/PlaNet_PyTorch. 
the original code works with image data, but I decided to implement it with feed-forward networks. 
this is not a functional code as I have not adjusted the hyperparameters and some of the methods have some issues. 
the only reason for implementing this was to understand world models and Dreamer architectures, as PlaNet is the mother of all of them :)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from torch.distributions.kl import kl_divergence
import gymnasium as gym
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, embed_dim, min_stddev=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.min_stddev = min_stddev
        
        self.rnn = nn.GRUCell(state_dim + action_dim, hidden_dim)
        
        # Prior networks
        self.fc_rnn_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_std_prior = nn.Linear(hidden_dim, state_dim)
        
        # Posterior networks  
        self.fc_rnn_hidden_embed = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_std_posterior = nn.Linear(hidden_dim, state_dim)
    
    def initial_state(self, batch_size, device):
        """Return initial state with zeros"""
        return {
            'mean': torch.zeros(batch_size, self.state_dim, device=device),
            'std':  torch.zeros(batch_size, self.state_dim, device=device),
            'stochastic': torch.zeros(batch_size, self.state_dim, device=device),
            'deterministic': torch.zeros(batch_size, self.hidden_dim, device=device),
        }
    
    def prior(self, state, action, hidden):
        """Compute prior distribution p(s_t+1 | h_t+1)"""
        input = torch.cat([state, action], dim=1)
        
        hidden = self.rnn(input, hidden)
        
        state_prior = F.relu(self.fc_rnn_hidden(hidden))
        
        mean = self.fc_state_mean_prior(state_prior)
        std = F.softplus(self.fc_state_std_prior(state_prior)) + self.min_stddev
        
        dist = torch.distributions.Normal(mean, std)
        
        return dist, hidden
    
    def posterior(self, hidden, embed):
        """Compute posterior distribution p(s_t | h_t, o_t)"""        
        combined = torch.cat([embed, hidden], dim=1)
        hidden_embed = F.relu(self.fc_rnn_hidden_embed(combined))
        
        mean = self.fc_state_mean_posterior(hidden_embed)
        std = F.softplus(self.fc_state_std_posterior(hidden_embed)) + self.min_stddev
        
        dist = torch.distributions.Normal(mean, std)
        
        return dist, hidden
    
    def KLD(self, posterior_dist, prior_dist):
        kl = kl_divergence(posterior_dist, prior_dist)
        return kl.sum(dim=1)
    

class Decoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + hidden_dim, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, output_dim)
        
    def forward(self, state, hidden):
        x = torch.cat([state, hidden], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RewardModel(nn.Module):
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=300):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, rnn_hidden):
        hidden = F.relu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.relu(self.fc2(hidden))
        hidden = F.relu(self.fc3(hidden))
        reward = self.fc4(hidden)
        return reward


"""
this is different in original implementaion because they worked with pixel data, although i wrote sth here :))
"""

class Normalizer():
    @staticmethod
    def preprocess_obs(obs):
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)
        return obs



class ReplayBuffer:
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool_)  
        self.index = 0
        self.is_filled = False
        
    def push(self, observation, action, reward, done):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done
        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity
        
    def sample(self, batch_size, chunk_length):
        """Sample sequences of experiences"""
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        
        for _ in range(batch_size): 
            cross_border = True
            attempts = 0
            max_attempts = 100  
            
            while cross_border and attempts < max_attempts:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()
                attempts += 1
                
            if attempts >= max_attempts:
                print(f"Warning: Could not find sequence that doesn't cross episode boundary after {max_attempts} attempts.")
            
            sampled_indexes += list(range(initial_index, final_index + 1))
            
        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:])
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1])
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
            
        return sampled_observations, sampled_actions, sampled_rewards, sampled_done
    
    def __len__(self):
        return self.capacity if self.is_filled else self.index



class CEMPlanner:
    def __init__(self, env, encoder, rssm, reward_model, horizon, iterations, n_candidates, topk):
        self.env = env
        self.encoder = encoder
        self.rssm = rssm
        self.reward_model = reward_model
        self.horizon = horizon
        self.iterations = iterations
        self.n_candidates = n_candidates
        self.topk = topk
        self.device = next(encoder.parameters()).device
        self.action_low = torch.tensor(env.action_space.low).to(self.device)
        self.action_high = torch.tensor(env.action_space.high).to(self.device)
        self.rnn_hidden = None
        
    def reset(self):
        """reset hidden state between episodes"""
        self.rnn_hidden = None
        
    def plan(self, obs):
        obs_tensor = torch.tensor(Normalizer.preprocess_obs(obs)).float().to(self.device)
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        with torch.no_grad():
            if self.rnn_hidden is None:
                self.rnn_hidden = torch.zeros(1, self.rssm.hidden_dim, device=self.device)
                
            embedded_obs = self.encoder(obs_tensor)
            state_posterior, _ = self.rssm.posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample()
            
            action_dim = self.env.action_space.shape[0]
            mean = torch.zeros((self.horizon, action_dim), device=self.device)
            std = torch.ones((self.horizon, action_dim), device=self.device)
            
            for _ in range(self.iterations):
                action_candidates = torch.distributions.Normal(mean, std).sample([self.n_candidates]).transpose(0, 1)
                
                returns = torch.zeros(self.n_candidates, device=self.device)
                current_state = state.repeat(self.n_candidates, 1)
                current_rnn_hidden = self.rnn_hidden.repeat(self.n_candidates, 1)
                
                for h in range(self.horizon):
                    action = action_candidates[h]
                    #action = torch.clamp(action, self.action_low, self.action_high)
                    
                    prior_dist, current_rnn_hidden = self.rssm.prior(
                        current_state, 
                        action, 
                        current_rnn_hidden
                    )
                    
                    current_state = prior_dist.sample()
                    reward = self.reward_model(current_state, current_rnn_hidden)
                    returns += reward.squeeze()
                
                elite_indices = returns.argsort(descending=True)[:self.topk]
                elite_actions = action_candidates[:, elite_indices]
                
                mean = elite_actions.mean(dim=1)
                std = elite_actions.std(dim=1) + 1e-6
            
            action = mean[0]
            
            """
            updating hidden state for the next round of planning
            """
            _, self.rnn_hidden = self.rssm.prior(
                state_posterior.sample(),
                action.unsqueeze(0),
                self.rnn_hidden
            )
            
        return action.cpu().numpy()




def collect_random_episodes(env, replay_buffer, num_episodes):
    total_steps = 0
    total_reward = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(obs, action, reward, done)
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
        
        total_reward += episode_reward

    return total_steps


def collect_planned_episodes(env, planner, replay_buffer, num_episodes):
    total_steps = 0
    returns = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        planner.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = planner.plan(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(obs, action, reward, done)
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
        
        returns.append(episode_reward)
        print(f"planning episode {episode+1}/{num_episodes}: reward = {episode_reward:.2f}")
    
    return total_steps, sum(returns) / len(returns)


def train(models, optimizer, replay_buffer, batch_size, chunk_length, free_nats, kl_weight):
    encoder, rssm, obs_model, reward_model = models
    
    observations, actions, rewards, _ = replay_buffer.sample(batch_size, chunk_length)
    
    observations = torch.tensor(Normalizer.preprocess_obs(observations), device=device).float()
    actions = torch.tensor(actions, device=device).float()
    rewards = torch.tensor(rewards, device=device).float()
    
    state = torch.zeros(batch_size, rssm.state_dim, device=device)
    rnn_hidden = torch.zeros(batch_size, rssm.hidden_dim, device=device)
    
    obs_loss = 0
    reward_loss = 0
    kl_loss = 0
    
    for t in range(chunk_length - 1):
        embedded_obs_t = encoder(observations[:, t])
        
        posterior_dist, rnn_hidden = rssm.posterior(rnn_hidden, embedded_obs_t)
        state = posterior_dist.rsample()
        
        pred_obs = obs_model(state, rnn_hidden)
        pred_reward = reward_model(state, rnn_hidden)
        
        obs_loss += 0.5 * F.mse_loss(pred_obs, observations[:, t])
        reward_loss += 0.5 * F.mse_loss(pred_reward, rewards[:, t])
        
        prior_dist, rnn_hidden = rssm.prior(state, actions[:, t], rnn_hidden)
        
        embedded_next_obs = encoder(observations[:, t+1])
        next_posterior_dist, _ = rssm.posterior(rnn_hidden, embedded_next_obs)
        
        kl_div = rssm.KLD(next_posterior_dist, prior_dist)
        kl_loss += kl_div.clamp(min=free_nats).mean()
        
        state = next_posterior_dist.rsample()
    
    obs_loss /= (chunk_length - 1)
    reward_loss /= (chunk_length - 1)
    kl_loss /= (chunk_length - 1)
    
    loss = obs_loss + reward_loss + kl_weight * kl_loss
    
    optimizer.zero_grad()
    loss.backward()
    
    all_params = itertools.chain(
        encoder.parameters(), 
        rssm.parameters(), 
        obs_model.parameters(), 
        reward_model.parameters()
    )
    torch.nn.utils.clip_grad_norm_(all_params, 1000.0)
    
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'obs_loss': obs_loss.item(),
        'reward_loss': reward_loss.item(),
        'kl_loss': kl_loss.item()
    }


def evaluate(env, planner, episodes=5):
    returns = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        planner.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = planner.plan(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        returns.append(total_reward)
    
    return sum(returns) / len(returns)


def train_planet(
    env_name="Pendulum-v1",
    epochs=50,
    seed_episodes=5,
    batch_size=50,
    sequence_length=50,
    training_iterations=100,
    planning_horizon=12,
    state_dim=30,
    hidden_dim=200,
):
 
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    encoder = Encoder(input_dim=obs_dim).to(device)
    rssm = RSSM(state_dim, action_dim, hidden_dim, embed_dim=1024).to(device)
    obs_model = Decoder(state_dim, hidden_dim, obs_dim).to(device)
    reward_model = RewardModel(state_dim, hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        itertools.chain(
            encoder.parameters(),
            rssm.parameters(),
            obs_model.parameters(),
            reward_model.parameters()
        ),
        lr=1e-3
    )
    
    buffer = ReplayBuffer(100000, env.observation_space.shape, action_dim)
    
    planner = CEMPlanner(
        env, encoder, rssm, reward_model,
        horizon=planning_horizon, 
        iterations=10,
        n_candidates=1000, 
        topk=100
    )
    """
    first let collect random samples
    """
    print("collecting random exp ")
    total_steps = collect_random_episodes(env, buffer, seed_episodes)
    print("finished")    
    """
    now train it
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for i in range(training_iterations):
            losses = train(
                (encoder, rssm, obs_model, reward_model),
                optimizer, buffer, batch_size, sequence_length,
                free_nats=3.0, kl_weight=0.1
            )
            
            if (i+1) % 10 == 0:
                print(f"  Iteration {i+1}/{training_iterations}, "
                      f"Loss: {losses['loss']:.3f}")
        
        
        print("collecting plnanned exp:")
        steps, avg_reward = collect_planned_episodes(env, planner, buffer, 10)
        print(f"average reward: {avg_reward:.2f}")
        print(f"total steps for planning phases: {steps + total_steps}")

        """
        let's see ?
        """

        
        if (epoch+1) % 10 == 0:
            eval_reward = evaluate(env, planner, episodes=3)
            print(f"evaluation: {eval_reward:.2f}")
    
    print("training complete")
    return encoder, rssm, obs_model, reward_model

train_planet()