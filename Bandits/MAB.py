import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random
from tqdm import tqdm

class BanditEnv:
    
    def __init__(self, p_arr):
        self.p_arr = np.array(p_arr)
        self.action_space = len(p_arr)
        self.observation_space = 1  # single state
        self.reset()
        
    def reset(self):
        """Reset environment state"""
        self.t = 0
        return 0  # return observation (0 for bandits)
    
    def step(self, action):
        assert 0 <= action < self.action_space, f"Invalid action: {action}"
        
        reward = 1 if np.random.random() < self.p_arr[action] else 0
        
        self.t += 1
        
        next_state = 0
        
        info = {"optimal_action": np.argmax(self.p_arr)}
        
        done = False
        
        return next_state, reward, done, info

class AgentAdapter:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, n_episodes=1, steps_per_episode=500):
        all_rewards = []
        all_actions = []
        
        for _ in range(n_episodes):
            self.agent.reset()
            rewards = []
            actions = []
            env.reset()
            
            for _ in range(steps_per_episode):
                action = self.agent.get_action()
                _, reward, _, _ = env.step(action)
                self.agent.update_Q(action, reward)
                
                rewards.append(reward)
                actions.append(action)
                
            all_rewards.append(rewards)
            all_actions.append(actions)
            
        return np.array(all_rewards), np.array(all_actions)

@dataclass
class RndAg:
    n_act: int
    
    def __post_init__(self):
        self.reset()
        
    def reset(self):
        self.t = 0
        self.act_counts = np.zeros(self.n_act, dtype=int)
        self.Q = np.zeros(self.n_act, dtype=float)
        
    def update_Q(self, act, rew):
        pass  
        
    def get_action(self):
        self.t += 1
        return np.random.randint(0, self.n_act)

@dataclass
class ExpFstAg:
    n_act: int
    max_ex: int
    
    def __post_init__(self):
        self.reset()
        
    def reset(self):
        self.t = 0
        self.act_counts = np.zeros(self.n_act, dtype=int)
        self.Q = np.zeros(self.n_act, dtype=float)
        
    def update_Q(self, act, rew):
        self.act_counts[act] += 1
        self.Q[act] += (rew - self.Q[act]) / self.act_counts[act]
        
    def get_action(self):
        self.t += 1
        return np.random.randint(0, self.n_act) if self.t <= self.max_ex else np.argmax(self.Q)

@dataclass
class UCB_Ag:
    n_act: int
    
    def __post_init__(self):
        self.reset()
        
    def reset(self):
        self.t = 0
        self.act_counts = np.zeros(self.n_act, dtype=int)
        self.Q = np.zeros(self.n_act, dtype=float)
        
    def update_Q(self, act, rew):
        self.act_counts[act] += 1
        self.Q[act] += (rew - self.Q[act]) / self.act_counts[act]
        
    def get_action(self):
        self.t += 1
        
        if np.any(self.act_counts == 0):
            return np.where(self.act_counts == 0)[0][0]
        
        bonus = np.sqrt(np.log(self.t) / self.act_counts)
        Q_explore = self.Q + bonus
        return np.random.choice(np.flatnonzero(Q_explore == Q_explore.max()))

@dataclass
class EpsGdAg:
    n_act: int
    eps: float = 0.1
    
    def __post_init__(self):
        self.reset()
        
    def reset(self):
        self.t = 0
        self.act_counts = np.zeros(self.n_act, dtype=int)
        self.Q = np.zeros(self.n_act, dtype=float)
        
    def update_Q(self, act, rew):
        self.act_counts[act] += 1
        self.Q[act] += (rew - self.Q[act]) / self.act_counts[act]
        
    def get_action(self):
        self.t += 1
        return np.random.randint(0, self.n_act) if random.random() < self.eps else np.argmax(self.Q)

def run_and_plot_individual(agent_name, agent, p_arr, n_episodes=100, steps_per_episode=500):
    env = BanditEnv(p_arr)
    adapter = AgentAdapter(agent)
    
    print(f"running {agent_name}...")
    rewards, actions = adapter.run(env, n_episodes, steps_per_episode)
    
    # individual performance
    plt.figure(figsize=(10, 6))
    mean_rewards = rewards.mean(axis=0)
    plt.plot(mean_rewards, label=agent_name)
    
    best_arm_reward = max(p_arr)
    avg_arm_reward = np.mean(p_arr)
    plt.axhline(y=best_arm_reward, linestyle='--', color='black', label=f'Best Arm ({best_arm_reward:.2f})')
    plt.axhline(y=avg_arm_reward, linestyle=':', color='gray', label=f'Avg Reward ({avg_arm_reward:.2f})')
    
    plt.title(f"{agent_name} performance")
    plt.xlabel("step")
    plt.ylabel("average reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return rewards

def run_all_algorithms(p_arr, n_episodes=100, steps_per_episode=500):
    n_actions = len(p_arr)
    results = {}
    
    agents = {
        "Random": RndAg(n_act=n_actions),
        "ExpFst5": ExpFstAg(n_act=n_actions, max_ex=5),
        "ExpFst20": ExpFstAg(n_act=n_actions, max_ex=20),
        "UCB": UCB_Ag(n_act=n_actions),
        "EpsGd0.1": EpsGdAg(n_act=n_actions, eps=0.1),
        "EpsGd0.2": EpsGdAg(n_act=n_actions, eps=0.2)
    }
    
    for name, agent in agents.items():
        results[name] = run_and_plot_individual(name, agent, p_arr, n_episodes, steps_per_episode)
    
    plt.figure(figsize=(12, 7))
    
    for name, rewards in results.items():
        mean_rewards = rewards.mean(axis=0)
        plt.plot(mean_rewards, label=name)
    
    best_arm_reward = max(p_arr)
    avg_arm_reward = np.mean(p_arr)
    plt.axhline(y=best_arm_reward, linestyle='--', color='black', label=f'Best Arm ({best_arm_reward:.2f})')
    plt.axhline(y=avg_arm_reward, linestyle=':', color='gray', label=f'Avg Reward ({avg_arm_reward:.2f})')
    
    plt.title("comparison of all algorithms")
    plt.xlabel("step")
    plt.ylabel("average reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nfinal average rewards:")
    for name, rewards in results.items():
        final_reward = rewards.mean(axis=0)[-1]
        print(f"{name}: {final_reward:.4f}")

if __name__ == "__main__":
    p_arr = [0.2, 0.3, 0.4, 0.1, 0.5, 0.7, 0.2, 0.1, 0.3, 0.4]
    print("running algorithms on standard probability distribution")
    run_all_algorithms(p_arr, n_episodes=100, steps_per_episode=500)
    
    skewed_p = [0.1, 0.2, 0.15, 0.21, 0.3, 0.05, 0.9, 0.13, 0.17, 0.07,
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    print("\nrunning algorithms on skewed probability distribution")
    run_all_algorithms(skewed_p, n_episodes=100, steps_per_episode=500)