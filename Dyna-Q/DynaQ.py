import random
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from collections import defaultdict
import heapq
import matplotlib.pyplot as plt
from tqdm import trange

class ShapedRewardFrozenLake(FrozenLakeEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = self.custom_reward_function(obs, reward, terminated)
        return obs, reward, terminated, truncated, info
    
    def custom_reward_function(self, observation, reward, done):
        custom_reward = -1 if reward == 0 else reward
        custom_reward = 100 * custom_reward if done else custom_reward
        return custom_reward


def greedy_policy(state, q_values):
    action = np.argmax(q_values[state, :])
    return action

def epsilon_greedy_policy(state, q_values, epsilon, action_space):
    action = greedy_policy(state, q_values) if random.random() > epsilon else action_space.sample()
    return action

def dyna_q(n_episodes, env, alpha, gamma, n, initial_epsilon, min_epsilon, epsilon_decay):
    """standard Dyna-Q with epsilon decay."""
    
    reward_sums = np.zeros(n_episodes)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    model = defaultdict(dict)
    epsilon = initial_epsilon
    
    for episode_i in trange(n_episodes, desc="Dyna-Q episodes"):
        state, info = env.reset()
        reward_sum = 0
        terminal = False
        steps = 0
        
        while not terminal:
          
            action = epsilon_greedy_policy(state, q, epsilon, env.action_space)        
            next_state, reward, terminated, truncated, info = env.step(action)
            terminal = terminated or truncated  
            q[state, action] += alpha * (reward + gamma * np.max(q[next_state, :]) - q[state, action])    
            model[state][action] = (next_state, reward)
            
    
            for _ in range(n):
                if not model:
                    break
                plan_state = random.choice(list(model.keys()))
                plan_action = random.choice(list(model[plan_state].keys()))  
                plan_next_state, plan_reward = model[plan_state][plan_action]                    
                q[plan_state, plan_action] += alpha * (plan_reward + gamma * np.max(q[plan_next_state, :]) - 
                                                      q[plan_state, plan_action])
            
        
            state = next_state
            reward_sum += reward
            steps += 1
            
            if steps > 100:
                break
 
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        reward_sums[episode_i] = reward_sum
        
        if reward > 0 and terminated:
            print(f"Episode {episode_i}: Total reward = {reward_sum:.1f}, Steps = {steps}")
    
    return q, reward_sums

def dyna_q_priority(n_episodes, env, alpha, gamma, n, initial_epsilon, min_epsilon, epsilon_decay, theta):
    """Dyna-Q with prioritized sweeping and epsilon decay."""
    
    reward_sums = np.zeros(n_episodes)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    model = defaultdict(dict)
    priorities = []
    epsilon = initial_epsilon
    
    for episode_i in trange(n_episodes, desc="prioritized Dyna-Q episodes"):
        state, info = env.reset()
        reward_sum = 0
        terminal = False
        steps = 0
        
        while not terminal:
       
            action = epsilon_greedy_policy(state, q, epsilon, env.action_space)   
            next_state, reward, terminated, truncated, info = env.step(action)
            terminal = terminated or truncated
                 
            old_q = q[state, action]
            q[state, action] += alpha * (reward + gamma * np.max(q[next_state, :]) - old_q)
            td_error = abs(reward + gamma * np.max(q[next_state, :]) - old_q)
                 
            model[state][action] = (next_state, reward)
            
            # update priority queue if TD error is significant
            if td_error > theta:
                heapq.heappush(priorities, (-td_error, (state, action)))
            
            # perform planning steps with prioritized sweeping
            for _ in range(n):
                if not priorities:
                    break
                
                _, (plan_state, plan_action) = heapq.heappop(priorities)
                
                if plan_state not in model or plan_action not in model[plan_state]:
                    continue
    
                plan_next_state, plan_reward = model[plan_state][plan_action]
                
                plan_old_q = q[plan_state, plan_action]
                q[plan_state, plan_action] += alpha * (plan_reward + gamma * np.max(q[plan_next_state, :]) - plan_old_q)
                
                # update predecessors
                for pred_state in model:
                    for pred_action in model[pred_state]:
                        pred_next_state, pred_reward = model[pred_state][pred_action]
                        if pred_next_state == plan_state:
                            pred_td_error = abs(pred_reward + gamma * np.max(q[plan_state, :]) - q[pred_state, pred_action])
                            if pred_td_error > theta:
                                heapq.heappush(priorities, (-pred_td_error, (pred_state, pred_action)))
            

            state = next_state
            reward_sum += reward
            steps += 1
            
            if steps > 100:
                break
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        reward_sums[episode_i] = reward_sum
        
        if reward > 0 and terminated:
            print(f"Episode {episode_i}: total reward = {reward_sum:.1f}, Steps = {steps}")
    
    return q, reward_sums

if __name__ == "__main__":

    shaped_env = ShapedRewardFrozenLake(map_name="8x8", is_slippery=False)
    original_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    
    params = {
        'alpha': 0.25,            
        'gamma': 0.9,         
        'n': 30,                 
        'initial_epsilon': 1.0,  
        'min_epsilon': 0.01,     
        'epsilon_decay': 0.99    
    }
    
    priority_param = {
        'theta': 5  
    }
    
    n_episodes = 1000
    
    print("running standard Dyna-Q on shaped reward environment")
    q_dyna_shaped, rewards_dyna_shaped = dyna_q(n_episodes, shaped_env, **params)
    
    print("\nrunning Prioritized Sweeping Dyna-Q on shaped reward environment")
    q_priority_shaped, rewards_priority_shaped = dyna_q_priority(n_episodes, shaped_env, **params, **priority_param)
    
    print("\nrunning standard Dyna-Q on original environment")
    q_dyna_original, rewards_dyna_original = dyna_q(n_episodes, original_env, **params)
    
    print("\nrunning Prioritized sweeping Dyna-Q on original environment...")
    q_priority_original, rewards_priority_original = dyna_q_priority(n_episodes, original_env, **params, **priority_param)
    
    window_size = 50
    avg_dyna_shaped = np.convolve(rewards_dyna_shaped, np.ones(window_size)/window_size, mode='valid')
    avg_priority_shaped = np.convolve(rewards_priority_shaped, np.ones(window_size)/window_size, mode='valid')
    avg_dyna_original = np.convolve(rewards_dyna_original, np.ones(window_size)/window_size, mode='valid')
    avg_priority_original = np.convolve(rewards_priority_original, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(window_size-1, n_episodes), avg_dyna_shaped, 'b-', label='Dyna-Q (shaped)')
    plt.plot(range(window_size-1, n_episodes), avg_priority_shaped, 'r-', label='prioritized Dyna-Q (Shaped)')
    plt.plot(range(window_size-1, n_episodes), avg_dyna_original, 'g-', label='Dyna-Q (Original)')
    plt.plot(range(window_size-1, n_episodes), avg_priority_original, 'y-', label='prioritized Dyna-Q (Original)')
    plt.xlabel('episode')
    plt.ylabel('average reward')
    plt.title(f'algorithm comparison (moving avg, window={window_size})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards_dyna_shaped, 'b-', alpha=0.3)
    plt.plot(range(window_size-1, n_episodes), avg_dyna_shaped, 'b-', linewidth=2)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('Dyna-Q (Rewards)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(rewards_priority_shaped, 'r-', alpha=0.3)
    plt.plot(range(window_size-1, n_episodes), avg_priority_shaped, 'r-', linewidth=2)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('prioritized Dyna-Q (Rewards)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(rewards_dyna_original, 'g-', alpha=0.3)
    plt.plot(range(window_size-1, n_episodes), avg_dyna_original, 'g-', linewidth=2)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('Dyna-Q (original environment)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(rewards_priority_original, 'y-', alpha=0.3)
    plt.plot(range(window_size-1, n_episodes), avg_priority_original, 'y-', linewidth=2)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('prioritized Dyna-Q (original environment)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nfinal average reward (last 100 episodes):")
    print(f"  Dyna-Q (Shaped): {np.mean(rewards_dyna_shaped[-100:]):.1f}")
    print(f"  prioritized Dyna-Q (shaped): {np.mean(rewards_priority_shaped[-100:]):.1f}")
    print(f"  Dyna-Q (Original): {np.mean(rewards_dyna_original[-100:]):.1f}")
    print(f"  Prioritized Dyna-Q (original): {np.mean(rewards_priority_original[-100:]):.1f}")
    
    dyna_success_shaped = np.sum(rewards_dyna_shaped > 0)
    priority_success_shaped = np.sum(rewards_priority_shaped > 0)
    dyna_success_original = np.sum(rewards_dyna_original > 0)
    priority_success_original = np.sum(rewards_priority_original > 0)
    
    print(f"\nsuccessful episodes (reached goal):")
    print(f"  Dyna-Q (Shaped): {dyna_success_shaped}/{n_episodes} ({dyna_success_shaped/n_episodes*100:.1f}%)")
    print(f"  prioritized Dyna-Q (Shaped): {priority_success_shaped}/{n_episodes} ({priority_success_shaped/n_episodes*100:.1f}%)")
    print(f"  Dyna-Q (Original): {dyna_success_original}/{n_episodes} ({dyna_success_original/n_episodes*100:.1f}%)")
    print(f"  prioritized Dyna-Q (Original): {priority_success_original}/{n_episodes} ({priority_success_original/n_episodes*100:.1f}%)")