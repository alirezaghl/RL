import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description="REINFORCE algorithm with and without baseline")
    
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    
    parser.add_argument("--input_dim", type=int, default=4, help="Input dimension for policy network")
    parser.add_argument("--output_dim", type=int, default=2, help="Output dimension for policy network")
    parser.add_argument("--hidden_sizes", type=str, default="256,128", help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--value_hidden_sizes", type=str, default="128,64", help="Value network hidden sizes")
    
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--log_interval", type=int, default=10, help="Episodes between logging")
    
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--run_both", action="store_true", default=True, 
                    help="Run both versions (with and without baseline)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def set_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
            
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        policy_logits = self.net(state)
        policy_dist = nn.Softmax(dim=1)(policy_logits)
        return policy_dist, policy_logits
        
def compute_returns(rewards, gamma=0.99):
    discounted_returns = []
    cumulative_return = 0
    for reward in reversed(rewards):
        cumulative_return = gamma * cumulative_return + reward
        discounted_returns.insert(0, cumulative_return)
    return discounted_returns

def train_reinforce(env, policy_net, optimizer, num_episodes, gamma=0.99, log_interval=10):
    episode_rewards_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_log_probs = []
        done = False
        
        while not done:
            # Convert state to tensor and add batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(policy_net.parameters()).device)
            
            # Get action distribution from policy
            policy_dist, _ = policy_net(state_tensor)
            
            # Sample action from distribution
            action = torch.multinomial(policy_dist, 1).item()
            
            # Store log probability of selected action
            log_prob = torch.log(policy_dist[0, action])
            episode_log_probs.append(log_prob)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store reward
            episode_rewards.append(reward)
            
            # Update state
            state = next_state
        
        # Episode complete - calculate total reward and store
        total_episode_reward = sum(episode_rewards)
        episode_rewards_history.append(total_episode_reward)
        
        # Calculate returns at each timestep
        returns = compute_returns(episode_rewards, gamma)
        returns_tensor = torch.FloatTensor(returns).to(next(policy_net.parameters()).device)
        
        # Calculate loss
        policy_loss = 0
        for step_log_prob, step_return in zip(episode_log_probs, returns_tensor):
            policy_loss += -step_log_prob * step_return
        
        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards_history[-log_interval:])
            print(f"Episode {episode + 1}, Average Reward (last {log_interval}): {avg_reward:.2f}")
    
    return episode_rewards_history

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(ValueNetwork, self).__init__()
        
        # Parse hidden sizes from string to list of integers
        layers = []
        prev_size = input_dim
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
            
        layers.append(nn.Linear(prev_size, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, state):
        value = self.net(state)
        return value

def train_reinforce_with_baseline(env, policy_net, value_net, policy_optimizer, value_optimizer, 
                                num_episodes, gamma=0.99, log_interval=10):
    episode_rewards_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_log_probs = []
        episode_values = []
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(policy_net.parameters()).device)
            
            policy_dist, _ = policy_net(state_tensor)
            state_value = value_net(state_tensor)
            
            action = torch.multinomial(policy_dist, 1).item()
            
            log_prob = torch.log(policy_dist[0, action])
            episode_log_probs.append(log_prob)
            episode_values.append(state_value)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            
            state = next_state
        
        total_episode_reward = sum(episode_rewards)
        episode_rewards_history.append(total_episode_reward)
        
        returns = compute_returns(episode_rewards, gamma)
        returns_tensor = torch.FloatTensor(returns).to(next(policy_net.parameters()).device)
        
        values_tensor = torch.cat(episode_values) 
        
        
        returns_tensor = returns_tensor.view(-1, 1)  
        
        # Calculate advantage
        advantages = returns_tensor - values_tensor.detach()
        
        # Calculate policy loss using advantage
        policy_loss = 0
        for step_log_prob, advantage in zip(episode_log_probs, advantages.squeeze()):
            policy_loss += -step_log_prob * advantage
            
        value_loss = nn.MSELoss()(values_tensor.squeeze(), returns_tensor.squeeze())
        
        # Update policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        # Update value network
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards_history[-log_interval:])
            print(f"Episode {episode + 1}, Average Reward (last {log_interval}): {avg_reward:.2f}")
    
    return episode_rewards_history

def exponential_moving_average(data, window_size):
    alpha = 2 / (window_size + 1)
    ema = []
    current_ema = data[0]
    ema.append(current_ema)
    for value in data[1:]:
        current_ema = alpha * value + (1 - alpha) * current_ema
        ema.append(current_ema)
    return ema

def main():
    
    args = parse_arguments()
    
    set_seeds(args.seed)
    
    policy_hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]
    value_hidden_sizes = [int(x) for x in args.value_hidden_sizes.split(',')]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = gym.make(args.env)
    env.reset(seed=args.seed)
        
    # Create results storage
    rewards_no_baseline = None
    rewards_with_baseline = None
    
    if not args.use_baseline or args.run_both:
        print("\nTraining REINFORCE without Baseline:")
        
        policy_net = PolicyNetwork(args.input_dim, args.output_dim, policy_hidden_sizes).to(device)
        optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
        
        rewards_no_baseline = train_reinforce(
            env, policy_net, optimizer, 
            num_episodes=args.episodes, 
            gamma=args.gamma,
            log_interval=args.log_interval
        )
    
    if args.use_baseline or args.run_both:
        print("\nTraining REINFORCE with Baseline:")
        
        policy_net = PolicyNetwork(args.input_dim, args.output_dim, policy_hidden_sizes).to(device)
        value_net = ValueNetwork(args.input_dim, value_hidden_sizes).to(device)
        
        policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
        value_optimizer = optim.Adam(value_net.parameters(), lr=args.lr)
        
        rewards_with_baseline = train_reinforce_with_baseline(
            env, policy_net, value_net, 
            policy_optimizer, value_optimizer,
            num_episodes=args.episodes, 
            gamma=args.gamma,
            log_interval=args.log_interval
        )
    
    if args.run_both and rewards_no_baseline and rewards_with_baseline:
        window_size = 50
        
        plt.figure(figsize=(12, 6))
        plt.plot(rewards_no_baseline, label="Without Baseline", alpha=0.3, color='tab:blue')
        plt.plot(rewards_with_baseline, label="With Baseline", alpha=0.3, color='tab:green')
        
        # Compute EMAs
        ema_no_baseline = exponential_moving_average(rewards_no_baseline, window_size)
        ema_with_baseline = exponential_moving_average(rewards_with_baseline, window_size)
        
        # Plot EMAs
        plt.plot(ema_no_baseline, label="Exponential Moving Avg (No Baseline)", linestyle='--', color='tab:blue')
        plt.plot(ema_with_baseline, label="Exponential Moving Avg (With Baseline)", linestyle='--', color='tab:green')
        
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.title(f"REINFORCE: With vs Without Baseline (Seed: {args.seed})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    elif rewards_no_baseline:
        window_size = 50
        
        plt.figure(figsize=(12, 6))
        plt.plot(rewards_no_baseline, label="REINFORCE", alpha=0.3, color='tab:blue')
        
        # Compute EMA
        ema_no_baseline = exponential_moving_average(rewards_no_baseline, window_size)
        
        # Plot EMA
        plt.plot(ema_no_baseline, label="Exponential Moving Avg", linestyle='--', color='tab:blue')
        
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.title(f"REINFORCE without Baseline (Seed: {args.seed})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    elif rewards_with_baseline:
        window_size = 50
        
        plt.figure(figsize=(12, 6))
        plt.plot(rewards_with_baseline, label="REINFORCE with Baseline", alpha=0.3, color='tab:green')
        
        ema_with_baseline = exponential_moving_average(rewards_with_baseline, window_size)
        
        # Plot EMA
        plt.plot(ema_with_baseline, label="Exponential Moving Avg", linestyle='--', color='tab:green')
        
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.title(f"REINFORCE with Baseline (Seed: {args.seed})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    env.close()

if __name__ == "__main__":
    main()