import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions),\
        np.array(self.probs), np.array(self.values),\
        np.array(self.rewards), np.array(self.dones),\
        batches
    
    def store_memory(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        self.values = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )
        
        self.mean = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
    
    def forward(self, state):
        x = self.actor(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
    
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
    
    def forward(self, state):
        value = self.critic(state)
        return value

class Agent:
    def __init__(self, n_actions, input_dims, gamma, alpha, policy_clip,
                 batch_size, horizon, n_epochs, gae_lambda, fc1_dims=256, fc2_dims=256): 
        
        self.gamma = gamma
        self.policy_clip = policy_clip  
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda  
        self.horizon = horizon  

        self.actor = ActorNetwork(n_actions, input_dims, alpha, fc1_dims, fc2_dims)  
        self.critic = CriticNetwork(input_dims, alpha, fc1_dims, fc2_dims)  
        self.memory = PPOMemory(batch_size)        
    
    def remember(self, state, action, probs, values, reward, done):
        self.memory.store_memory(state, action, probs, values, reward, done)
    
    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(device)
        
        with torch.no_grad():
            action, log_prob = self.actor.sample_action(state)
            value = self.critic(state)
        
        action = action.cpu().numpy().flatten()
        log_prob = log_prob.cpu().item()
        value = value.cpu().item()
        
        return action, log_prob, value

    def compute_advantage(self, rewards, values, dones):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)

        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1-dones[t]) * gae

            returns[t] = gae + values[t]
            advantages[t] = gae
            
        return advantages, returns
        
    def learn(self):
        states, actions, old_probs, values, rewards, dones, batches = self.memory.generate_batches()

        advantages, returns = self.compute_advantage(rewards, values, dones)

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)  
        old_probs = torch.tensor(old_probs, dtype=torch.float).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(device)
        returns = torch.tensor(returns, dtype=torch.float).to(device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            for batch in batches:
                states_batch = states[batch]
                actions_batch = actions[batch]
                old_probs_batch = old_probs[batch]
                advantages_batch = advantages[batch]
                returns_batch = returns[batch]

                mean, std = self.actor.forward(states_batch)
                dist = Normal(mean, std)
                new_probs = dist.log_prob(actions_batch).sum(dim=1)
                
                policy_ratio = torch.exp(new_probs - old_probs_batch)
                weighted_probs = advantages_batch * policy_ratio
                clipped_probs = advantages_batch * torch.clamp(policy_ratio, 1-self.policy_clip, 1+self.policy_clip)

                actor_loss = -torch.min(weighted_probs, clipped_probs).mean()
                critic_value = self.critic(states_batch).squeeze()
                critic_loss = nn.functional.mse_loss(critic_value, returns_batch)

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()
    
    def save_models(self, path='models'):
        os.makedirs(path, exist_ok=True)
        print('... saving models ...')
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
    
    def load_models(self, path='models'):
        print('... loading models ...')
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))

def parse_args():
    parser = argparse.ArgumentParser(description='PPO for HalfCheetah')
    
    # Environment
    parser.add_argument('--env', type=str, default='HalfCheetah-v4', help='Gymnasium environment')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes')
    parser.add_argument('--horizon', type=int, default=2048, help='Steps before update')
    parser.add_argument('--log-interval', type=int, default=10, help='Print interval')
    parser.add_argument('--save-interval', type=int, default=100, help='Model save interval')
    parser.add_argument('--render-interval', type=int, default=100, help='Render GIF interval')
    
    # PPO hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--alpha', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--policy-clip', type=float, default=0.2, help='Policy clip')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='PPO epochs')
    
    # Network parameters
    parser.add_argument('--fc1-dims', type=int, default=256, help='First hidden layer size')
    parser.add_argument('--fc2-dims', type=int, default=256, help='Second hidden layer size')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-gif', action='store_true', help='Disable GIF creation')
    
    return parser.parse_args()

def make_gif(frames, filename):
    """Create a GIF from a list of frames"""
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved GIF to {filename}")

def record_video(env, agent, video_length=500, filename=""):
    """Record a video of the agent's performance"""
    frames = []
    observation, _ = env.reset()
    
    for _ in range(video_length):
        frames.append(env.render())
        action, _, _ = agent.choose_action(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            break
    
    make_gif(frames, filename)
    return frames

def train(args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Create environment (with render_mode for visualizations)
    render_mode = 'rgb_array' if not args.no_gif else None
    env = gym.make(args.env, render_mode=render_mode)
    
    # Create agent
    agent = Agent(
        n_actions=env.action_space.shape[0],
        input_dims=[env.observation_space.shape[0]],
        gamma=args.gamma,
        alpha=args.alpha,
        policy_clip=args.policy_clip,
        batch_size=args.batch_size,
        horizon=args.horizon,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        fc1_dims=args.fc1_dims,
        fc2_dims=args.fc2_dims
    )
    
    # For tracking progress
    best_score = env.reward_range[0]
    score_history = []
    avg_history = []
    learn_steps = 0
    
    # Training loop with progress bar
    pbar = tqdm(range(args.episodes), desc="Training")
    for episode in pbar:
        observation, _ = env.reset(seed=args.seed + episode)
        done = False
        score = 0
        episode_steps = 0
        
        while not done:
            action, log_prob, value = agent.choose_action(observation)
            
            # Take step in environment
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_steps += 1
            
            agent.remember(observation, action, log_prob, value, reward, done)
            
            # Update observation
            observation = next_observation
            score += reward
            
            if len(agent.memory.states) >= args.horizon:
                agent.learn()
                learn_steps += 1
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:]) if len(score_history) >= 100 else np.mean(score_history)
        avg_history.append(avg_score)
        
        pbar.set_postfix({
            'score': f'{score:.1f}',
            'avg_100': f'{avg_score:.1f}',
            'steps': episode_steps,
            'updates': learn_steps
        })
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        # Record video periodically
        if not args.no_gif and episode > 0 and episode % args.render_interval == 0:
            video_env = gym.make(args.env, render_mode='rgb_array')
            video_filename = f'videos/episode_{episode}.gif'
            record_video(video_env, agent, filename=video_filename)
            video_env.close()
    
    if not args.no_gif:
        agent.load_models()
        video_env = gym.make(args.env, render_mode='rgb_array')
        record_video(video_env, agent, filename='videos/final_performance.gif')
        video_env.close()
    
    # Plot learning curve
    plt.figure(figsize=(12, 8))
    plt.plot(score_history, alpha=0.4, color='blue', label='Episode Scores')
    plt.plot(avg_history, linewidth=2, color='red', label='100-episode Average')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'PPO Learning Curve - {args.env}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results/learning_curve.png')
    plt.close()
    
    if not args.no_gif:
        create_learning_progress_gif()
    
    env.close()
    return score_history

def create_learning_progress_gif():
    """Create a GIF showing learning progress over time"""
    video_files = sorted(glob.glob('videos/episode_*.gif'))
    
    if len(video_files) == 0:
        return
    
    if len(video_files) > 5:
        indices = np.linspace(0, len(video_files)-1, 5).astype(int)
        video_files = [video_files[i] for i in indices]
    
    all_frames = []
    for video_file in video_files:
        episode = int(video_file.split('_')[-1].split('.')[0])
        
        reader = imageio.get_reader(video_file)
        frames = []
        for frame in reader:
            frames.append(imageio.core.util.Array(frame))
        
        title_frame = np.ones((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8) * 255
        
        all_frames.extend(frames[:100])  # Limit to first 100 frames to keep GIF manageable
    
    imageio.mimsave('videos/learning_progress.gif', all_frames, fps=30)
    print("Created learning progress GIF")

if __name__ == '__main__':
    args = parse_args()
    train(args)