import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="DQN and DDQN for cartpole")
    
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])

    parser.add_argument("--render_mode", type=str, default="rgb_array")

    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 10, 15, 43, 63],)

    parser.add_argument("--eval_threshold", type=int, default=450,)

    parser.add_argument("--moving_avg_window", type=int, default=25)

    parser.add_argument("--eval_episodes", type=int, default=3)
    
    parser.add_argument("--hidden_size1", type=int, default=512)

    parser.add_argument("--hidden_size2", type=int, default=125)
    
    parser.add_argument("--dqn_episodes", type=int, default=500)

    parser.add_argument("--dqn_buffer_size", type=int, default=10000)

    parser.add_argument("--dqn_batch_size", type=int, default=128)

    parser.add_argument("--dqn_gamma", type=float, default=0.99)

    parser.add_argument("--dqn_eps_start", type=float, default=0.8)

    parser.add_argument("--dqn_eps_end", type=float, default=0.01)

    parser.add_argument("--dqn_eps_decay", type=float, default=8000)

    parser.add_argument("--dqn_lr", type=float, default=1e-3)
    
    parser.add_argument("--ddqn_episodes", type=int, default=500)

    parser.add_argument("--ddqn_buffer_size", type=int, default=10000)

    parser.add_argument("--ddqn_batch_size", type=int, default=128)

    parser.add_argument("--ddqn_gamma", type=float, default=0.99)

    parser.add_argument("--ddqn_eps_start", type=float, default=0.8)

    parser.add_argument("--ddqn_eps_end", type=float, default=0.01)

    parser.add_argument("--ddqn_eps_decay", type=float, default=8000)

    parser.add_argument("--ddqn_lr", type=float, default=1e-3)

    parser.add_argument("--ddqn_update_rate", type=int, default=5)

    parser.add_argument("--ddqn_tau", type=float, default=0.05)
    
    parser.add_argument("--print_freq", type=int, default=100,
                      help="print progress during training")
    
    parser.add_argument("--plot_save_prefix", type=str, default="")
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size1=512, hidden_size2=125):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory(object):
    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.experience = namedtuple("experience", field_names=("state", "action", "reward", "next_state", "done"))

    def push(self, state, action, reward, next_state, done):
        
        if isinstance(state, torch.Tensor):
            state = state.clone().detach()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.clone().detach()
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)
        
        states = torch.cat([e.state for e in experiences if e is not None]).to(device)
        actions = torch.tensor([[e.action] for e in experiences if e is not None], 
                             dtype=torch.long, device=device)
        rewards = torch.tensor([[e.reward] for e in experiences if e is not None], 
                             dtype=torch.float, device=device)
        next_states = torch.cat([e.next_state for e in experiences if e is not None]).to(device)
        dones = torch.tensor([[e.done] for e in experiences if e is not None], 
                           dtype=torch.float, device=device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

    def get_size(self):
        return self.__len__()


class DQNAgent(object):
    def __init__(self, q_network, memory, optimizer, criterion, params):
        self.policy_net = q_network
        
        self.reply_buffer = memory
        
        self.criterion = criterion()
        self.optimizer = optimizer(self.policy_net.parameters(), lr=params['LR'], amsgrad=True)
    
        self.gamma = params['GAMMA']
        self.eps = {'START': params['EPS_START'], 'END': params['EPS_END'], 'DECAY': params['EPS_DECAY']}
        self.steps_done = 0
        self.Loss = []

    def step(self, state, action, reward, next_state, done):
        
        self.steps_done += 1
        
        self.reply_buffer.push(state, action, reward, next_state, done)
        
        if self.reply_buffer.get_size() > self.reply_buffer.batch_size:
            states, actions, rewards, next_states, dones = self.reply_buffer.sample()
            self.learn(states, actions, rewards, next_states, dones)

    def act(self, state, greedy=False):
       
        self.eps_threshold = self.eps['END'] + (self.eps['START'] - self.eps['END']) * \
                           np.exp(-1.0 * self.steps_done / self.eps['DECAY'])

        if greedy or random.random() > self.eps_threshold:
            with torch.no_grad():
                Q_values = self.policy_net(state)
                action = torch.argmax(Q_values).item()
                max_Q = torch.max(Q_values).item()
        else:
            action = np.random.randint(0, env.action_space.n)
            max_Q = 0
        
        return action, max_Q

    def learn(self, states, actions, rewards, next_states, dones):
        q_values = self.policy_net(states)
        state_Q = q_values.gather(1, actions)
        
        with torch.no_grad():
            # Compute expected Q-values using the policy network (no target network)
            next_state_Q = self.policy_net(next_states)
            target = rewards + (self.gamma * next_state_Q.max(1, keepdim=True)[0] * (1 - dones))

        loss = self.criterion(state_Q, target)
        self.Loss.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, PATH):
        torch.save(self.policy_net, PATH + '_policy.pt')

    def load(self, PATH):
        self.policy_net = torch.load(PATH + '_policy.pt')


def create_dqn_agent(seed, QNetwork, ReplayMemory, optimizer, criterion, params, args):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    q_network = QNetwork(
        env.observation_space.shape[0], 
        env.action_space.n,
        args.hidden_size1,
        args.hidden_size2
    ).to(device)
    
    memory = ReplayMemory(params['BUFFER_SIZE'], params['BATCH_SIZE'])
    return DQNAgent(q_network, memory, optimizer, criterion, params)


class DDQNAgent(object):
    def __init__(self, q_network, memory, optimizer, criterion, params):
        self.policy_net = q_network
        self.target_net = type(q_network)(
            env.observation_space.shape[0], 
            env.action_space.n,
            q_network.fc1.out_features,
            q_network.fc2.out_features
        ).to(device)
        
        self.reply_buffer = memory
        
        self.criterion = criterion()
        self.optimizer = optimizer(self.policy_net.parameters(), lr=params['LR'], amsgrad=True)
        self.tau = params['TAU']
        self.gamma = params['GAMMA']
        self.update_rate = params['UPDATE_RATE']
        self.eps = {'START': params['EPS_START'], 'END': params['EPS_END'], 'DECAY': params['EPS_DECAY']}
        self.steps_done = 0
        self.Loss = []

        self.soft_update(1.0)

    def step(self, state, action, reward, next_state, done):

        self.steps_done += 1
        
        self.reply_buffer.push(state, action, reward, next_state, done)
        
        if self.reply_buffer.get_size() > self.reply_buffer.batch_size:
            states, actions, rewards, next_states, dones = self.reply_buffer.sample()
            self.learn(states, actions, rewards, next_states, dones)
            
            if self.steps_done % self.update_rate == 0:
                self.soft_update(self.tau)

    def act(self, state, greedy=False, eps_threshold=None):
    
        self.eps_threshold = self.eps['END'] + (self.eps['START'] - self.eps['END']) * \
                           np.exp(-1.0 * self.steps_done / self.eps['DECAY'])

        if greedy or random.random() > self.eps_threshold:
            with torch.no_grad():
                Q_values = self.policy_net(state)
                action = torch.argmax(Q_values).item()
                max_Q = torch.max(Q_values).item()
        else:
            action = np.random.randint(0, env.action_space.n)
            max_Q = 0
            
        return action, max_Q

    def learn(self, states, actions, rewards, next_states, dones):
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states).float().to(device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(actions).long().to(device)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.from_numpy(rewards).float().to(device)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.from_numpy(next_states).float().to(device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.from_numpy(dones).float().to(device)
            
        state_Q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_state_Q = self.policy_net(next_states)
            next_actions = torch.max(next_state_Q, 1)[1].unsqueeze(1)  
                    
            next_state_target_Q = self.target_net(next_states).gather(1, next_actions)
                    
            # TD target
            target = rewards + (self.gamma * next_state_target_Q * (1 - dones))
                    
        loss = self.criterion(state_Q, target)
        self.Loss.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tau):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save(self, PATH):
        torch.save(self.policy_net, PATH + '_policy.pt')
        torch.save(self.target_net, PATH + '_target.pt')

    def load(self, PATH):
        self.policy_net = torch.load(PATH + '_policy.pt')
        self.target_net = torch.load(PATH + '_target.pt')


def create_ddqn_agent(seed, QNetwork, ReplayMemory, optimizer, criterion, params, args):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    q_network = QNetwork(
        env.observation_space.shape[0], 
        env.action_space.n,
        args.hidden_size1,
        args.hidden_size2
    ).to(device)
    
    memory = ReplayMemory(params['BUFFER_SIZE'], params['BATCH_SIZE'])
    
    return DDQNAgent(q_network, memory, optimizer, criterion, params)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_smooth(DDQN_mean_rewards, DDQN_min_rewards, DDQN_max_rewards, 
                DQN_mean_rewards, DQN_min_rewards, DQN_max_rewards, args):
    """comparison between DQN and DDQN performance"""
    plt.figure(figsize=(12,7))

    DDQN, = plt.plot(range(len(DDQN_mean_rewards)), DDQN_mean_rewards, color='blue', label='DDQN')
    plt.fill_between(range(len(DDQN_min_rewards)), DDQN_min_rewards, DDQN_max_rewards, color='blue', alpha=0.2)

    DQN, = plt.plot(range(len(DQN_mean_rewards)), DQN_mean_rewards, color='red', label='DQN')
    plt.fill_between(range(len(DQN_min_rewards)), DQN_min_rewards, DQN_max_rewards, color='red', alpha=0.2)

    plt.legend(handles=[DDQN, DQN])
    plt.xlabel('Episode')
    plt.ylabel('Reward (Moving Average)')
    plt.title('DQN vs DDQN Performance Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filename = f"{args.plot_save_prefix}dqn_ddqn_comparison.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()


def plot_values(values, args, agent_type="Agent"):

    plt.figure(figsize=(15, 9))

    for i, value in enumerate(values):
        for n, data in enumerate(value):
            plt.plot(range(len(data)), data, 
                     label=f"{agent_type} {i+1}, evaluation {n+1}")

    plt.title(f'{agent_type} test episode mean Q values')
    plt.xlabel("steps")
    plt.ylabel("Q value")
    plt.grid(True)
    plt.legend()
    
    filename = f"{args.plot_save_prefix}{agent_type.lower()}_q_values.png"
    plt.savefig(filename)
    plt.show()


def evaluate_policy(env, agent, num_episodes):
    """agent's performance over multiple episodes"""
    total_rewards = []
    episode_q_values = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        episode_reward = 0
        step_q_values = []
        
        while True:
            action, Q = agent.act(state, greedy=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            step_q_values.append(Q)
            
            next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            
            state = next_state
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        
        running_means = [sum(step_q_values[:i+1])/(i+1) for i in range(len(step_q_values))]
        episode_q_values.append(running_means)
    
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    return mean_reward, std_reward, episode_q_values


def train_dqn_agents(DQN_agents, args):

    dqn_sum_of_rewards = [[] for _ in range(len(DQN_agents))]
    
    for i, dqn_Agent in enumerate(DQN_agents):
        print(f"training DQN agent with seed {args.seeds[i]}")
        
        for e in range(1, args.dqn_episodes + 1):
            episode_reward = 0
            state, _ = env.reset()
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
            while True:
                action, _ = dqn_Agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action) 

                next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
                
                dqn_Agent.step(state, action, reward, next_state, done or truncated)
                
                state = next_state
                episode_reward += reward
                
                if done or truncated:
                    break

            dqn_sum_of_rewards[i].append(episode_reward)
            
            if e % args.print_freq == 0:
                avg_reward = np.mean(dqn_sum_of_rewards[i][-100:])
                print(f"DQN Agent {i+1}, Episode {e}/{args.dqn_episodes}, Avg Reward (last 100): {avg_reward:.2f}")
    
    return dqn_sum_of_rewards


def train_ddqn_agents(DDQN_agents, args):
    ddqn_sum_of_rewards = [[] for _ in range(len(DDQN_agents))]
    
    for i, DDQN_agent in enumerate(DDQN_agents):
        print(f"training DDQN agent with seed {args.seeds[i]}")
        
        for e in range(1, args.ddqn_episodes + 1):
            episode_reward = 0
            state, _ = env.reset()
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            while True:
                action, _ = DDQN_agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
                
                DDQN_agent.step(state, action, reward, next_state, done or truncated)
                
                state = next_state
                episode_reward += reward
                
                if done or truncated:
                    break
                    
            ddqn_sum_of_rewards[i].append(episode_reward)
            
            if e % args.print_freq == 0:
                avg_reward = np.mean(ddqn_sum_of_rewards[i][-100:])
                print(f"DDQN Agent {i+1}, Episode {e}/{args.ddqn_episodes}, Avg Reward (last 100): {avg_reward:.2f}")
    
    return ddqn_sum_of_rewards


def main():
    args = parse_args()
    
    global device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    global env
    env = gym.make("CartPole-v1", render_mode=args.render_mode)
    print('Observations:', env.observation_space.shape[0])
    print('Actions:', env.action_space.n)
    
    plt.style.use('ggplot')
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    
    optimizer = torch.optim.Adam
    criterion = nn.MSELoss
    
    dqn_params = {
        'BUFFER_SIZE': args.dqn_buffer_size,
        'BATCH_SIZE': args.dqn_batch_size,
        'GAMMA': args.dqn_gamma,
        'EPS_START': args.dqn_eps_start,
        'EPS_END': args.dqn_eps_end,
        'EPS_DECAY': args.dqn_eps_decay,
        'LR': args.dqn_lr
    }
    
    ddqn_params = {
        'UPDATE_RATE': args.ddqn_update_rate,
        'BUFFER_SIZE': args.ddqn_buffer_size,
        'BATCH_SIZE': args.ddqn_batch_size,
        'GAMMA': args.ddqn_gamma,
        'EPS_START': args.ddqn_eps_start,
        'EPS_END': args.ddqn_eps_end,
        'EPS_DECAY': args.ddqn_eps_decay,
        'TAU': args.ddqn_tau,
        'LR': args.ddqn_lr
    }
    
    print("\nconfig")
    print(f"DQN episodes: {args.dqn_episodes}, DDQN episodes: {args.ddqn_episodes}")
    print(f"DQN params: γ={args.dqn_gamma}, ε={args.dqn_eps_start}→{args.dqn_eps_end} (decay={args.dqn_eps_decay})")
    print(f"DDQN params: γ={args.ddqn_gamma}, ε={args.ddqn_eps_start}→{args.ddqn_eps_end} (decay={args.ddqn_eps_decay})")
    print(f"DDQN update rate: {args.ddqn_update_rate}, τ={args.ddqn_tau}")
    print(f"network: {args.hidden_size1}-{args.hidden_size2}")
    print(f"random seeds: {args.seeds}")
    print("\n")
    
    DQN_agents = []
    DDQN_agents = []
    
    # multiple agents with different seeds
    for seed in args.seeds:
        dqn_agent = create_dqn_agent(seed, QNetwork, ReplayMemory, optimizer, criterion, dqn_params, args)
        ddqn_agent = create_ddqn_agent(seed, QNetwork, ReplayMemory, optimizer, criterion, ddqn_params, args)
        
        DQN_agents.append(dqn_agent)
        DDQN_agents.append(ddqn_agent)
    
    print("\ntraining DQN Agents")
    dqn_sum_of_rewards = train_dqn_agents(DQN_agents, args)
    
    print("\ntraining DDQN Agents")
    ddqn_sum_of_rewards = train_ddqn_agents(DDQN_agents, args)
    
    if len(dqn_sum_of_rewards[0]) > 0 and len(ddqn_sum_of_rewards[0]) > 0:
        DDQN_mean_rewards = moving_average(np.array(ddqn_sum_of_rewards).mean(axis=0), args.moving_avg_window)
        DDQN_min_rewards = moving_average(np.array(ddqn_sum_of_rewards).min(axis=0), args.moving_avg_window)
        DDQN_max_rewards = moving_average(np.array(ddqn_sum_of_rewards).max(axis=0), args.moving_avg_window)

        DQN_mean_rewards = moving_average(np.array(dqn_sum_of_rewards).mean(axis=0), args.moving_avg_window)
        DQN_min_rewards = moving_average(np.array(dqn_sum_of_rewards).min(axis=0), args.moving_avg_window)
        DQN_max_rewards = moving_average(np.array(dqn_sum_of_rewards).max(axis=0), args.moving_avg_window)

        plot_smooth(DDQN_mean_rewards, DDQN_min_rewards, DDQN_max_rewards, 
                    DQN_mean_rewards, DQN_min_rewards, DQN_max_rewards, args)
    
    print("\nevaluating DDQN Agents")
    DDQN_values = []
    for i, agent in enumerate(DDQN_agents):
        mean_reward, std_reward, mean_values = evaluate_policy(env, agent, args.eval_episodes)
        print(f"DDQN Agent {i+1}: mean_reward = {mean_reward:.2f} ± {std_reward:.2f}")
        if mean_reward >= args.eval_threshold:
            DDQN_values.append(mean_values)

    if DDQN_values:
        plot_values(DDQN_values, args, agent_type="DDQN")
    else:
        print(f'[Info] No DDQN agent passed the {args.eval_threshold} reward threshold. train more.')

    print("\nevaluating DQN Agents")
    DQN_values = []
    for i, agent in enumerate(DQN_agents):
        mean_reward, std_reward, mean_values = evaluate_policy(env, agent, args.eval_episodes)
        print(f"DQN Agent {i+1}: mean_reward = {mean_reward:.2f} ± {std_reward:.2f}")
        if mean_reward >= args.eval_threshold:
            DQN_values.append(mean_values)

    if DQN_values:
        print(f"Found {len(DQN_values)} DQN agents with reward >= {args.eval_threshold}")
        plot_values(DQN_values, args, agent_type="DQN")
    else:
        print(f'[Info] No DQN agent passed the {args.eval_threshold} reward threshold. train more.')

    env.close()
    

if __name__ == "__main__":
    main()