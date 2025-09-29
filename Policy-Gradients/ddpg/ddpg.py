import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import random
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import torch.optim as optim

LOG_STD_MIN = -20
LOG_STD_MAX = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def initialize_layers(sequential_net):
    f1 = 1./np.sqrt(sequential_net[0].weight.data.size()[0])
    sequential_net[0].weight.data.uniform_(-f1, f1)
    sequential_net[0].bias.data.uniform_(-f1, f1)
    
    f2 = 1./np.sqrt(sequential_net[2].weight.data.size()[0])
    sequential_net[2].weight.data.uniform_(-f2, f2)
    sequential_net[2].bias.data.uniform_(-f2, f2)
    
    f3 = 0.003
    sequential_net[4].weight.data.uniform_(-f3, f3)
    sequential_net[4].bias.data.uniform_(-f3, f3)

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def get_size(self):
        return len(self.buffer)
    
class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_1, hidden_2):
        super(CriticNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_states + n_actions, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1)
        )
        
        self.to(device)
        self.apply(weights_init_)
        initialize_layers(self.net)
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], 1)
        q = self.net(x)
        return q

class ActorNetwork(nn.Module):
    def __init__(self, n_inputs, hidden_1, hidden_2, n_actions):
        super(ActorNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, n_actions),
            nn.Tanh()
        )

        self.to(device)
        self.apply(weights_init_)
        initialize_layers(self.net)
    
    def forward(self, state):
        action = self.net(state)
        return action
    
class OUNoise(object):
    '''Ornstein-Uhlenbeck process for exploration'''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
  
class DDPG(object):
    def __init__(self, env, args):
        self.env = env
        self.gamma = args.gamma
        self.tau = args.tau
        self.memory = ReplayMemory(args.memory_size, args.seed)
        
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        
        self.critic = CriticNetwork(self.n_states, self.n_actions, args.hidden_1, args.hidden_2)
        self.actor = ActorNetwork(self.n_states, args.hidden_1, args.hidden_2, self.n_actions)
        
        self.target_critic = CriticNetwork(self.n_states, self.n_actions, args.hidden_1, args.hidden_2)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.target_actor = ActorNetwork(self.n_states, args.hidden_1, args.hidden_2, self.n_actions)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.noise = OUNoise(env.action_space)
        
        self.batch_size = args.batch_size
        self.update_rate = args.update_rate
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
            
        self.steps_done = 0
        self.episodes_done = 0
        
        self.render_interval = args.render_interval
        self.seed = args.seed
        
        self.video_dir = args.video_dir
        os.makedirs(self.video_dir, exist_ok=True)
        
        self.rewards = []
        self.avg_rewards = []

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
            
            if not evaluate:
                action = self.noise.get_action(action, self.steps_done)
            
            return action

    def step(self, state, action, reward, next_state, done):

        self.memory.push(state, action, reward, next_state, done)
        
        self.steps_done += 1
        
        if self.memory.get_size() > self.batch_size and self.steps_done % self.update_rate == 0:
            for _ in range(self.update_rate):  
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
                self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def update(self, states, actions, rewards, next_states, dones):

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values
            
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, max_episodes):
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
            
        should_render = False
        episode_rewards = []
        
        for episode in tqdm(range(1, max_episodes + 1)):
            self.episodes_done = episode
            frames = []  
            
            should_render = episode % self.render_interval == 0
            
            state, _ = self.env.reset(seed=self.seed)
            self.noise.reset()
            
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):

                action = self.select_action(state)
                
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                if should_render:
                    frames.append(self.env.render())
                
                self.step(state, action, reward, next_state, done or truncated)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            self.rewards.append(episode_reward)
            self.avg_rewards.append(avg_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
            
            if should_render and frames:
                gif_path = os.path.join(self.video_dir, f'episode_{episode}.gif')
                imageio.mimsave(gif_path, frames, duration=0.01)
                print(f"Saved episode {episode} as GIF")
                
                self.save_learning_curve()
        
        self.save_learning_curve()
        
        return self.rewards, self.avg_rewards
    
    def save_learning_curve(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards, label='Episode Reward')
        plt.plot(self.avg_rewards, label='100-Episode Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DDPG Learning Curve')
        plt.legend()
        plt.savefig(os.path.join(self.video_dir, 'learning_curve.png'))
        plt.close()
    
    def evaluate(self, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            state, _ = self.env.reset(seed=self.seed)
            done = False
            truncated = False
            while not (done or truncated):
                action = self.select_action(state, evaluate=True)
                state, reward, done, truncated, _ = self.env.step(action)
                avg_reward += reward
        
        avg_reward /= eval_episodes
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        return avg_reward

def main():
    parser = argparse.ArgumentParser(description='DDPG for HalfCheetah-v4')
    
    parser.add_argument('--env', default='HalfCheetah-v4', help='Gym environment name')
    
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='Critic learning rate')
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='Actor learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--memory_size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--hidden_1', type=int, default=400, help='Hidden layer 1 size')
    parser.add_argument('--hidden_2', type=int, default=300, help='Hidden layer 2 size')
    parser.add_argument('--update_rate', type=int, default=1, help='How often to update networks')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    
    parser.add_argument('--render_interval', type=int, default=100, help='Render every N episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--video_dir', default='video_ddpg', help='Directory to save videos')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    env = gym.make(args.env, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    
    agent = DDPG(env, args)
    
    rewards, avg_rewards = agent.train(args.episodes)
    
    agent.evaluate()
    
    env.close()

if __name__ == "__main__":
    main()
