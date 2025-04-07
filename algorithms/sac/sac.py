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

torch.set_default_dtype(torch.float32)

torch.set_float32_matmul_precision('high')

LOG_STD_MIN = -20
LOG_STD_MAX = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable cuDNN benchmark for faster training (if using CUDA)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print("CUDA optimizations enabled")

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # Convert to float32 for better performance
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        
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
    def __init__(self, n_states, n_actions, hidden_dim):
        super(CriticNetwork, self).__init__()

        self.net_1 = nn.Sequential(
            nn.Linear(n_states + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.net_2 = nn.Sequential(
            nn.Linear(n_states + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.to(device)
        self.apply(weights_init_)
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], 1)
        q1 = self.net_1(x)
        q2 = self.net_2(x)
        return q1, q2

class ActorNetwork(nn.Module):
    def __init__(self, n_inputs, hidden_dim, n_actions, action_space):
        super(ActorNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, n_actions)
        self.log_std = nn.Linear(hidden_dim, n_actions)
        
        # Make sure these are float32
        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2., 
                                         dtype=torch.float32).to(device)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2., 
                                        dtype=torch.float32).to(device)
        
        self.to(device)
        self.apply(weights_init_)
    
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std)

        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        u = dist.rsample()  
        a = torch.tanh(u)
        
        action = a * self.action_scale + self.action_bias

        log_prob = dist.log_prob(u)
        log_prob = (log_prob - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        
        return action, log_prob, mean
    
class SAC(object):
    def __init__(self, env, args):
        self.env = env
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.memory = ReplayMemory(args.memory_size, args.seed)
        
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
        
        self.critic = CriticNetwork(n_states, n_actions, args.hidden_dim)
        self.actor = ActorNetwork(n_states, args.hidden_dim, n_actions, env.action_space)
        
        self.target_critic = CriticNetwork(n_states, n_actions, args.hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.n_episodes = args.episodes
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.update_rate = args.update_rate
        
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.lr)
        
        if args.auto_entropy:
            self.target_entropy = -np.prod(env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device, dtype=torch.float32)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args.lr)
        
        self.auto_entropy = args.auto_entropy
        self.steps_done = 0
        
        self.render_interval = args.render_interval
        self.reward_log_interval = args.reward_log_interval
        self.seed = args.seed
        
        # Updated paths to match repository structure
        self.video_dir = os.path.join('../../results/sac/videos')
        self.model_dir = os.path.join('../../results/sac/models')
        self.log_dir = os.path.join('../../results/sac/logs')
        
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def select_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
            return action.detach().cpu().numpy().flatten()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
        self.steps_done += 1
        
        if self.memory.get_size() > self.batch_size and self.steps_done % self.update_rate == 0:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
            self.learn(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def _calculate_target_q(self, next_states, rewards, dones):
        with torch.no_grad():
            next_actions, next_state_log_prob, _ = self.actor.sample(next_states)
            next_target_1, next_target_2 = self.target_critic(next_states, next_actions)
            min_next_target = torch.min(next_target_1, next_target_2) - self.alpha * next_state_log_prob
            next_q_value = rewards + self.gamma * (1 - dones) * min_next_target
            return next_q_value
    
    def _calculate_critic_loss(self, states, actions, target_q_values):
        qf1, qf2 = self.critic(states, actions)
        qf1_loss = F.mse_loss(qf1, target_q_values)
        qf2_loss = F.mse_loss(qf2, target_q_values)
        return qf1_loss + qf2_loss
    
    def _calculate_actor_loss(self, states):
        pi, log_pi, _ = self.actor.sample(states)
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def learn(self, states, actions, rewards, next_states, dones):
        # Convert numpy arrays to float32 tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # Compute target Q values
        next_q_value = self._calculate_target_q(next_states, rewards, dones)
        
        # Update critic
        qf_loss = self._calculate_critic_loss(states, actions, next_q_value)
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Update actor
        policy_loss, log_pi = self._calculate_actor_loss(states)
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        
        # Soft update target networks
        self.soft_update(self.tau)
        
        # Update entropy coefficient if auto-tuning
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp().item()

    def soft_update(self, tau):
        for target_param, critic_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * critic_param.data + (1.0 - tau) * target_param.data)
    
    def train(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.n_episodes
        
        # For visualization
        all_frames = []
        episode_rewards = []
        best_reward = float('-inf')
        
        # Log file
        reward_log_file = open(os.path.join(self.log_dir, 'reward_log.csv'), 'w')
        reward_log_file.write('episode,reward,avg_reward_100\n')
        
        pbar = tqdm(range(num_episodes))
        for episode in pbar:
            state, _ = self.env.reset(seed=self.seed + episode)
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action = self.select_action(state)
                
                # Take action in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition and update
                self.step(state, action, reward, next_state, done)
                
                # Move to next state
                state = next_state
                episode_reward += reward
            
            # Save episode reward
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            # Log reward to file
            reward_log_file.write(f'{episode},{episode_reward:.2f},{avg_reward:.2f}\n')
            reward_log_file.flush()
            
            # Update progress bar
            pbar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'avg_reward': f'{avg_reward:.2f}'
            })
            
            # Save best model
            if avg_reward > best_reward and episode > 10:
                best_reward = avg_reward
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'target_critic': self.target_critic.state_dict()
                }, os.path.join(self.model_dir, 'sac_best.pth'))
                print(f"New best model saved with avg reward: {avg_reward:.2f}")
            
            # Log rewards and update plots at specified intervals
            if episode % self.reward_log_interval == 0 or episode == num_episodes - 1:
                self.plot_rewards(episode_rewards, episode)
                print(f"Episode {episode}: Avg reward over last 100 episodes: {avg_reward:.2f}")
            
            # Render and save video periodically
            if episode % self.render_interval == 0 or episode == num_episodes - 1:
                frames = self.record_video(os.path.join(self.video_dir, f"episode_{episode}.gif"))
                if frames:
                    all_frames.extend(frames[:100])  # Keep representation frames for final video
                
                # Periodically clear CUDA cache if using GPU
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Close log file
        reward_log_file.close()
        
        # Create final learning progress video
        if all_frames:
            imageio.mimsave(os.path.join(self.video_dir, 'learning_progress.gif'), all_frames, fps=30)
            
        # Final rewards plot
        self.plot_rewards(episode_rewards, num_episodes)
        
        # Final evaluation
        self.record_video(os.path.join(self.video_dir, "final.gif"))
        
        return episode_rewards
    
    def record_video(self, filename, num_frames=500):
        """Record agent performance as a video"""
        eval_env = gym.make('HalfCheetah-v4', render_mode='rgb_array')
        frames = []
        total_reward = 0
        
        state, _ = eval_env.reset(seed=self.seed)
        for _ in range(num_frames):
            # Render
            frame = eval_env.render()
            frames.append(frame)
            
            # Select action
            action = self.select_action(state, evaluate=True)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            if done:
                break
                
            state = next_state
        
        # Save video
        if frames:
            imageio.mimsave(filename, frames, fps=30)
            print(f"Saved video to {filename} (Eval reward: {total_reward:.2f})")
            
        eval_env.close()
        return frames
        
    def plot_rewards(self, rewards, episode):
        """Plot rewards over episodes"""
        plt.figure(figsize=(12, 6))
        
        # Plot episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Episode Rewards (Current: {rewards[-1]:.2f})')
        
        # Plot moving average
        plt.subplot(1, 2, 2)
        moving_avg = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
        plt.plot(moving_avg)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (100 ep)')
        plt.title(f'Moving Average (Current: {moving_avg[-1]:.2f})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'rewards_ep{episode}.png'))
        plt.close()
        
    def load_best_model(self):
        """Load the best saved model"""
        try:
            checkpoint = torch.load(os.path.join(self.model_dir, 'sac_best.pth'))
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_critic.load_state_dict(checkpoint['target_critic'])
            print("Loaded best model for evaluation")
        except:
            print("Could not load best model, using current model")

def parse_args():
    parser = argparse.ArgumentParser(description='SAC for HalfCheetah-v4')
    
    # Environment settings
    parser.add_argument('--env', type=str, default='HalfCheetah-v4', help='Gymnasium environment name')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--render-interval', type=int, default=100, help='Interval for rendering and saving videos')
    parser.add_argument('--reward-log-interval', type=int, default=10, help='Interval for logging and plotting rewards')
    
    # SAC hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy coefficient')
    parser.add_argument('--auto-entropy', action='store_true', dest='auto_entropy', default=True,
                       help='Use automatic entropy tuning (default: True)')
    parser.add_argument('--no-auto-entropy', action='store_false', dest='auto_entropy')
    parser.add_argument('--memory-size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--update-rate', type=int, default=1, help='Update rate (steps per update)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    env = gym.make(args.env)
    
    agent = SAC(env, args)
    
    rewards = agent.train()
    
    env.close()

if __name__ == "__main__":
    main()
