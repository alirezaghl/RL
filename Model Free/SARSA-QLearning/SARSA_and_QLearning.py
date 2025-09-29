import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='n-step SARSA and Q-learning')
    
    parser.add_argument('--env', type=str, default='CliffWalking-v0')
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.1)
    
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=500)
    
    parser.add_argument('--window', type=int, default=20)
    
    return parser.parse_args()

def greedy_policy(state, Q):
    return np.argmax(Q[state])

def epsilon_greedy(state, Q, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])

def calculate_n_step_Return(n_step_reward, discount=0.95):
    G = 0
    for r in reversed(n_step_reward):
        G = discount * G + r
    return G

def n_step_sarsa(n_step_reward, Q, state, action, next_state, next_action, discount, n, alpha):
    reward = calculate_n_step_Return(n_step_reward, discount)
    
    TD_error = reward + (discount**n) * Q[next_state][next_action] - Q[state][action]
    
    Q[state][action] += alpha * TD_error
    
    return Q

def n_step_q_learning(n_step_reward, Q, state, action, next_state, discount, n, alpha):
    reward = calculate_n_step_Return(n_step_reward, discount)
    
    TD_error = reward + (discount**n) * np.max(Q[next_state]) - Q[state][action]
    
    Q[state][action] += alpha * TD_error
    
    return Q

class Trainer:
    def __init__(self, env, n_steps=1, alpha=0.5, discount=0.95, epsilon=0.1, max_steps=500):
        self.env = env
        self.n_steps = n_steps
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon
        self.max_steps = max_steps        
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    def train_sarsa(self, num_episodes=500, policy_func=epsilon_greedy, update_func=n_step_sarsa):
        rewards_history = []
        
        for episode in range(num_episodes):
            
            state, _ = self.env.reset()
            action = policy_func(state, self.Q, self.epsilon)
            
            states = deque([state])
            actions = deque([action])
            rewards = deque()
            
            episode_reward = 0
            step = 0
            done = False
            
            while step < self.max_steps and not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                rewards.append(reward)
                states.append(next_state)
                
                next_action = policy_func(next_state, self.Q, self.epsilon)
                actions.append(next_action)
                
                if len(rewards) >= self.n_steps:
                    self.Q = update_func(
                        list(rewards),
                        self.Q,
                        states[0], 
                        actions[0], 
                        next_state, 
                        next_action, 
                        self.discount,
                        self.n_steps,
                        self.alpha
                    )
                    
                    # remove oldest experiences
                    states.popleft()
                    actions.popleft()
                    rewards.popleft()
                
               
                state = next_state
                action = next_action
                step += 1
            
            rewards_history.append(episode_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"SARSA - episode {episode+1}/{num_episodes}, avg reward (last 100): {avg_reward:.2f}")
        
        return rewards_history
    
    def train_q_learning(self, num_episodes=500, policy_func=epsilon_greedy, update_func=n_step_q_learning):
        rewards_history = []
        
        for episode in range(num_episodes):
            
            state, _ = self.env.reset()
            
            states = deque([state])
            actions = deque()
            rewards = deque()
            
            episode_reward = 0
            step = 0
            done = False
            
            while step < self.max_steps and not done:

                action = policy_func(state, self.Q, self.epsilon)
                actions.append(action)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                rewards.append(reward)
                
                if not done:
                    states.append(next_state)
                
                if len(rewards) >= self.n_steps and len(states) > 1:
                    self.Q = update_func(
                        list(rewards),
                        self.Q,
                        states[0],
                        actions[0],
                        next_state,
                        self.discount,
                        self.n_steps,
                        self.alpha
                    )
                    
                    states.popleft()
                    actions.popleft()
                    rewards.popleft()
                
                state = next_state
                step += 1
            
            rewards_history.append(episode_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Q-learning - Episode {episode+1}/{num_episodes}, Avg Reward (last 100): {avg_reward:.2f}")
        
        return rewards_history

def plot_average_rewards(sarsa_rewards, q_learning_rewards, window=50):
    
    plt.figure(figsize=(10, 6))
    
    
    if len(sarsa_rewards) >= window:
        sarsa_moving_avg = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
        q_learning_moving_avg = np.convolve(q_learning_rewards, np.ones(window)/window, mode='valid')
        
        episodes = range(window-1, len(sarsa_rewards))
        
        plt.plot(episodes, sarsa_moving_avg, linewidth=2, color='blue', label='SARSA')
        plt.plot(episodes, q_learning_moving_avg, linewidth=2, color='red', label='Q-learning')
        
        plt.legend(loc='lower right')
    
    plt.xlabel('episode')
    plt.ylabel('total Reward')
    plt.title('learning curves: SARSA vs Q-learning')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def main():

    args = parse_arguments()
    
    np.random.seed(args.seed)
    
    env = gym.make(args.env)
    print(f'observation spaces: {env.observation_space}')
    print(f'action spaces: {env.action_space}')
    
    sarsa_agent = Trainer(
        env=env,
        n_steps=args.n_steps,
        alpha=args.alpha,
        discount=args.discount,
        epsilon=args.epsilon,
        max_steps=args.max_steps
    )
    sarsa_rewards = sarsa_agent.train_sarsa(num_episodes=args.num_episodes)
    
    env = gym.make(args.env)
    
    q_learning_agent = Trainer(
        env=env,
        n_steps=args.n_steps,
        alpha=args.alpha,
        discount=args.discount,
        epsilon=args.epsilon,
        max_steps=args.max_steps
    )
    q_learning_rewards = q_learning_agent.train_q_learning(num_episodes=args.num_episodes)
    
    plot_average_rewards(sarsa_rewards, q_learning_rewards, window=args.window)

if __name__ == "__main__":
    main()