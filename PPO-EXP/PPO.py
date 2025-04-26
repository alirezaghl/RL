import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.distributions import Normal
import itertools
import torch.optim
import tqdm
import time
import os
import csv
import random
from datetime import datetime
import imageio
from einops import rearrange


@dataclass
class config():
    env_name = "Walker2d-v4"  
    seed = 0
    max_step_size = 2048  
    
    fc_dim_actor = 64
    fc_dim_critic = 128
    
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    vf_coef = 0.5
    clip_vloss = True
    ent_coef = 0.0
    episodes = 20000  
    lr = 3e-4
    initial_lr = 3e-4
    end_lr = 3e-5
    batch_size = max_step_size
    minibatch_size = 64
    total_training_steps = 100000
    max_grad_norm = 0.5
    update_epochs = 10
    use_kl_early_stopping = False  
    target_kl = 0.02
    verbose = 1 
    record_gif_every = 1000  
    gif_fps = 30  
    log_kl_divergence = True  
    log_interval = 1  
    verify_first_update_ratio = True  

def set_seed_everywhere(seed):
    """
    Set seed for all random number generators to ensure reproducibility.
    This includes Python's random, NumPy, PyTorch CPU & GPU.
    """
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds set to: {seed}")

class KLDivergenceTracker:
    """Class to track and analyze KL divergence during training"""
    def __init__(self, log_dir):
        self.kl_values = []
        self.update_indices = []
        self.epoch_indices = []
        self.batch_indices = []
        self.episode_indices = []
        self.csv_path = os.path.join(log_dir, "kl_divergence.csv")
        
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Update', 'Epoch', 'Batch', 'KL_Divergence'])
    
    def log_kl(self, kl_value, episode, update, epoch, batch):
        """Log a KL divergence value with its corresponding indices"""
        self.kl_values.append(kl_value)
        self.episode_indices.append(episode)
        self.update_indices.append(update)
        self.epoch_indices.append(epoch)
        self.batch_indices.append(batch)
        
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([episode, update, epoch, batch, kl_value])
    
    def get_stats(self):
        """Get basic statistics about KL divergence"""
        if not self.kl_values:
            return {"mean": 0, "min": 0, "max": 0, "latest": 0}
        
        return {
            "mean": np.mean(self.kl_values),
            "min": np.min(self.kl_values),
            "max": np.max(self.kl_values),
            "latest": self.kl_values[-1]
        }
    
    def plot_kl_evolution(self, save_path):
        """evolution of KL divergence throughout training"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.kl_values)), self.kl_values, alpha=0.6)
        
        if len(self.kl_values) > 10:
            window_size = min(len(self.kl_values) // 10, 100)
            if window_size > 0:
                smoothed = np.convolve(self.kl_values, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(self.kl_values)), smoothed, color='red', linewidth=2)
        
        plt.xlabel('Optimization Steps')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence Evolution During Training')
        
        if len(self.kl_values) > 0:
            plt.axhline(y=0.02, color='g', linestyle='--', label='Target KL')
            plt.axhline(y=0.03, color='r', linestyle='--', label='Early Stopping Threshold (1.5x)')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  
        plt.savefig(save_path)
        plt.close()
    
    def plot_kl_by_episode(self, save_path):
        """Plot average KL divergence per episode"""
        if not self.kl_values:
            return
            
        episode_kl = {}
        for i, ep in enumerate(self.episode_indices):
            if ep not in episode_kl:
                episode_kl[ep] = []
            episode_kl[ep].append(self.kl_values[i])
        
        episodes = sorted(episode_kl.keys())
        avg_kls = [np.mean(episode_kl[ep]) for ep in episodes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, avg_kls)
        plt.xlabel('Episode')
        plt.ylabel('Average KL Divergence')
        plt.title('KL Divergence by Episode')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig(save_path)
        plt.close()

class RatioTracker:
    """
    purpose based on 37 details -> Check if ratio=1: Check if the ratio are always 1s during the first epoch and first mini-batch update, 
    when new and old policies are the same and 
    """
  
    def __init__(self, log_dir):
        self.csv_path = os.path.join(log_dir, "policy_ratios.csv")
        
        # Create CSV file with header
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Update', 'Epoch', 'Batch', 'Min_Ratio', 'Max_Ratio', 'Mean_Ratio', 'Std_Ratio'])
    
    def log_ratios(self, ratios, update, epoch, batch):
        if isinstance(ratios, torch.Tensor):
            ratios = ratios.detach().cpu().numpy()
            
        min_ratio = np.min(ratios)
        max_ratio = np.max(ratios)
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([update, epoch, batch, min_ratio, max_ratio, mean_ratio, std_ratio])
        
        return min_ratio, max_ratio, mean_ratio, std_ratio

class Initialization():
    @staticmethod
    def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.init = Initialization.layer_init
        self.actor = nn.Sequential(
            self.init(nn.Linear(config.num_observations, config.fc_dim_actor)),
            nn.Tanh(),
            self.init(nn.Linear(config.fc_dim_actor, config.fc_dim_actor)),
            nn.Tanh(),
            self.init(nn.Linear(config.fc_dim_actor, config.num_actions), std=0.01)
        )
        torch.manual_seed(config.seed)
        self.policy_logstd = nn.Parameter(torch.zeros(1, config.num_actions))
    
    def forward(self, state):
        mu = self.actor(state)
        return mu

class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.init = Initialization.layer_init
        self.critic = nn.Sequential(
            self.init(nn.Linear(config.num_observations, config.fc_dim_critic)),
            nn.Tanh(),
            self.init(nn.Linear(config.fc_dim_critic, config.fc_dim_critic)),
            nn.Tanh(),
            self.init(nn.Linear(config.fc_dim_critic, 1), std=1.0)
        )
    
    def forward(self, state):
        value = self.critic(state)
        return value.squeeze(-1)

class Policy():
    @staticmethod
    def act(actor, critic, state, deterministic=False):
        """
        Get action from policy with option for deterministic evaluation
        
        Args:
            actor: Actor network
            critic: Critic network
            state: Environment state
            deterministic: If True, return mean action without sampling (for evaluation)
        """
        policy_mean = actor(state)
        
        if deterministic:
            # In deterministic mode, return mean without sampling
            # Useful for evaluation and gif creation
            action = policy_mean
            # We still need a placeholder log_prob
            log_prob = torch.zeros(action.shape[0])
        else:
            # Normal stochastic sampling for training
            policy_logstd = actor.policy_logstd.expand_as(policy_mean)
            policy_std = torch.exp(policy_logstd)
            
            dist = Normal(policy_mean, policy_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        
        value = critic(state)
        
        policy_logstd = actor.policy_logstd.expand_as(policy_mean)
        policy_std = torch.exp(policy_logstd)
        dist = Normal(policy_mean, policy_std)
        
        return log_prob, action, value, dist
    
    @staticmethod
    def evaluate(actor, critic, state, action):
        policy_mean = actor(state)
        policy_logstd = actor.policy_logstd.expand_as(policy_mean)
        policy_std = torch.exp(policy_logstd)
        
        dist = Normal(policy_mean, policy_std)
        log_prob = dist.log_prob(action).sum(-1)
        value = critic(state)
        
        return log_prob, value, dist

class Memory:
    def __init__(self, seed):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.rng = np.random.RandomState(seed)  
    
    def generate_minibatches(self, batch_size, minibatch_size, advantages=None, returns=None):
        """
        purpose based on 37 details -> ppo is an on-policy model and like every other one is sample-inefficient
        using minibathces makes the model to use the data more efficiently. 
        go over experience multiple times.
        """
        batch_size = min(len(self.states), batch_size)
        states = self.states[-batch_size:]
        actions = self.actions[-batch_size:]
        logprobs = self.logprobs[-batch_size:]
        rewards = self.rewards[-batch_size:]
        values = self.values[-batch_size:]
        dones = self.dones[-batch_size:]

        states_array = np.array(states)
        actions_array = np.array(actions)
        logprobs_array = np.array(logprobs)
        rewards_array = np.array(rewards)
        values_array = np.array(values)
        dones_array = np.array(dones)

        assert batch_size % minibatch_size == 0
        
        indices = np.arange(batch_size)
        self.rng.shuffle(indices)
        
        num_minibatches = batch_size // minibatch_size
        indices = indices.reshape(num_minibatches, minibatch_size)

        minibatches = []
        for i in range(num_minibatches):
            batch_indices = indices[i]
            
            batch_dict = {
                'states': states_array[batch_indices],
                'actions': actions_array[batch_indices],
                'logprobs': logprobs_array[batch_indices],
                'rewards': rewards_array[batch_indices],
                'values': values_array[batch_indices],
                'dones': dones_array[batch_indices]
            }
            
            if advantages is not None:
                batch_dict['advantages'] = advantages[batch_indices]
            
            if returns is not None:
                batch_dict['returns'] = returns[batch_indices]
                
            minibatches.append(batch_dict)
        
        return minibatches

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]
    
    def push(self, state, action, logprob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def __len__(self):
        return len(self.states)

"""
based on Arena ppo implementation 
https://arena-chapter2-rl.streamlit.app/%5B2.3%5D_PPO
"""
class GAE:
    @staticmethod
    @torch.inference_mode()
    def compute_gae(values, next_value, dones, gamma, rewards, gae_lambda):
        if isinstance(next_value, torch.Tensor):
            next_value = next_value.reshape(-1)[0]
        
        values_extended = torch.cat([values, torch.tensor([next_value])])
        
        deltas = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        next_advantage = 0
        
        for t in reversed(range(len(rewards))):
            deltas[t] = rewards[t] + gamma * values_extended[t+1] * (1-dones[t]) - values_extended[t]
            advantages[t] = deltas[t] + gamma * gae_lambda * (1-dones[t]) * next_advantage
            next_advantage = advantages[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + values
        
        return advantages, returns

"""
based on Arena ppo implementation 
https://arena-chapter2-rl.streamlit.app/%5B2.3%5D_PPO
"""

class PPOScheduler:
    def __init__(self, optimizer, initial_lr, end_lr, total_phases):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_phases = total_phases
        self.n_step_calls = 0

    def step(self):
        self.n_step_calls += 1
        frac = min(1.0, self.n_step_calls / self.total_phases)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)

def make_optimizer(network, total_phases, initial_lr, end_lr=0.0):
    """Create an optimizer for a single network"""
    optimizer = torch.optim.Adam(
        network.parameters(), 
        lr=initial_lr, 
        eps=1e-5
    )
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_phases)
    return optimizer, scheduler

class PPO(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        set_seed_everywhere(config.seed)
        
        self.config = config
        self.memory = Memory(config.seed)
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.policy = Policy()

        """
        for this i have to go with Andrychowicz 2021 (https://openreview.net/pdf?id=nIAxjsniDzg)
        unlike 37 details i have decided to not to share value and policy networks parameters,
        no layers shared with the policy

        """
        
        self.actor_optimizer, self.actor_scheduler = make_optimizer(
            self.actor, 
            config.total_training_steps, 
            config.lr, 
            config.end_lr
        )
        
        self.critic_optimizer, self.critic_scheduler = make_optimizer(
            self.critic, 
            config.total_training_steps, 
            config.lr, 
            config.end_lr
        )
        
        self.use_kl_early_stopping = config.use_kl_early_stopping
        self.target_kl = config.target_kl
        self.verbose = config.verbose
        self.log_kl_divergence = config.log_kl_divergence
        self.verify_first_update_ratio = config.verify_first_update_ratio
        
        self.episode_rewards = []
        self.avg_rewards = []
        self.update_count = 0
        
        self.kl_tracker = None
        self.ratio_tracker = None
    
    def remember(self, state, action, logprob, reward, value, done):
        self.memory.push(state, action, logprob, reward, value, done)
    
    def prepare_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            tensor = torch.FloatTensor(data)
        else:
            tensor = data.float()  
            
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def rollout(self, advantages=None, returns=None):
        return self.memory.generate_minibatches(
            min(len(self.memory), self.config.batch_size), 
            self.config.minibatch_size,
            advantages,
            returns
        )
    
    def clipped_surrogate_objective(self, logprob_new, mb_logprob, mb_advantages):
        logits_diff = logprob_new - mb_logprob
        prob_ratio = torch.exp(logits_diff)
        non_clipped = prob_ratio * mb_advantages
        clipped = torch.clamp(prob_ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef) * mb_advantages
        
        return torch.min(clipped, non_clipped).mean(), prob_ratio
    
    """
    based on 37 details implementation
    """

    def value_objective(self, values, mb_returns, old_values=None):
        """
        Compute value loss with optional clipping as in the PPO paper.
        Uses einops for consistent tensor handling.
        """
        # Ensure consistent tensor dimensions with einops
        values = rearrange(values, '... -> (...)') 
        mb_returns = rearrange(mb_returns, '... -> (...)')
        
        if self.config.clip_vloss and old_values is not None:
            old_values = rearrange(old_values, '... -> (...)')
            
            v_loss_unclipped = (values - mb_returns) ** 2
            v_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.clip_coef,
                self.config.clip_coef
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            loss = 0.5 * v_loss_max.mean()
        else:
            loss = 0.5 * ((values - mb_returns) ** 2).mean()
        
        return self.config.vf_coef * loss
    
    def entropy(self, dist):
        return self.config.ent_coef * -dist.entropy().mean()
    
    def calculate_kl_divergence(self, new_logprobs, old_logprobs):
        """
        calculate approximate KL divergence, based on original ppo implementation by Schulman et.al.
        """
        log_ratio = new_logprobs - old_logprobs
        return torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

    def advantages(self, states, actions, logprobs, rewards, values, dones):
        states = self.prepare_tensor(states)
        actions = self.prepare_tensor(actions)
        logprobs = self.prepare_tensor(logprobs)
        rewards = self.prepare_tensor(rewards)
        values = self.prepare_tensor(values)
        dones = self.prepare_tensor(dones)
        
        with torch.no_grad():
            next_value = self.critic(states[-1])
        
        advantages, returns = GAE.compute_gae(
            values, next_value, dones, 
            self.config.gamma, rewards, 
            self.config.gae_lambda
        )
        
        return advantages, returns
    
    def create_gif(self, episode, directory):
        """Record a GIF of the agent's performance with improved evaluation"""
        try:
            env = gym.make(self.config.env_name, render_mode="rgb_array")
            
            n_eval_episodes = 5
            best_reward = -float('inf')
            best_frames = []
            
            print("\nRunning evaluation episodes:")
            for eval_episode in range(n_eval_episodes):
                eval_seed = self.config.seed + episode * 100 + eval_episode
                
                frames = []
                state, _ = env.reset(seed=eval_seed)
                done = False
                total_reward = 0
                
                torch.manual_seed(eval_seed)
                
                while not done:
                    frame = env.render()
                    frames.append(frame)
                    
                    """
                    getting deterministic action for agent performance viz, its a better option compare 
                    to use stochastic policies
                    """
                    state_tensor = self.prepare_tensor(state)
                    with torch.no_grad():
                        _, action, _, _ = self.policy.act(self.actor, self.critic, state_tensor, deterministic=True)
                    
                    action_np = action.cpu().numpy().flatten()
                    state, reward, terminated, truncated, _ = env.step(action_np)
                    done = terminated or truncated
                    total_reward += reward
                
                """
                running evaluation mode several times to pick the best one for viz.
                """
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_frames = frames
                    
                print(f"  Eval episode {eval_episode+1}/{n_eval_episodes} (seed={eval_seed}): reward={total_reward:.1f}")
            
            env.close()
            
            torch.manual_seed(self.config.seed)
            
            gif_path = f"{directory}/episode_{episode}_reward_{best_reward:.1f}.gif"
            imageio.mimsave(gif_path, best_frames, fps=self.config.gif_fps)
            print(f"GIF saved to {gif_path} (best of {n_eval_episodes} evaluation episodes, reward={best_reward:.1f})")
            
            return best_reward
        except Exception as e:
            print(f"Error creating GIF: {str(e)}")
            return None
    
    def train(self):
        pbar = tqdm.tqdm(range(self.config.episodes), desc="Training")
        score_history = []
        avg_history = []
        continue_training = True
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{self.config.env_name}_seed{self.config.seed}_{timestamp}"
        checkpoint_dir = f"checkpoints/{run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        with open(f"{checkpoint_dir}/config.txt", 'w') as f:
            for key, value in vars(self.config).items():
                f.write(f"{key}: {value}\n")
        
        if self.log_kl_divergence:
            self.kl_tracker = KLDivergenceTracker(checkpoint_dir)
        
        if self.verify_first_update_ratio:
            self.ratio_tracker = RatioTracker(checkpoint_dir)

        for episode in pbar:
            episode_seed = self.config.seed + episode
            state, _ = self.config.env.reset(seed=episode_seed)
            
            done = False
            score = 0
            episode_steps = 0

            while not done and episode_steps < self.config.max_step_size and continue_training:
                episode_steps += 1
                state_tensor = self.prepare_tensor(state)

                with torch.no_grad():
                    log_prob, action, value, _ = self.policy.act(self.actor, self.critic, state_tensor, deterministic=False)
                
                action_np = action.cpu().numpy().flatten()
                
                next_state, reward, terminated, truncated, _ = self.config.env.step(action_np)
                done = terminated or truncated
                
                self.remember(
                    state, 
                    action_np, 
                    log_prob.item(), 
                    reward, 
                    value.item(), 
                    done
                )
                
                state = next_state
                score += reward
                
                if len(self.memory) >= self.config.batch_size:
                    if self.use_kl_early_stopping:
                        continue_training = self.update_policy(continue_training, episode)
                        
                        if not continue_training:
                            if self.verbose >= 1:
                                print(f"Stopping training due to KL threshold at episode {episode}")
                            break
                    else:
                        self.update_policy(continue_training=None, current_episode=episode)
            
            if not continue_training and self.use_kl_early_stopping:
                break
                
            score_history.append(score)
            avg_score = np.mean(score_history[-100:]) if len(score_history) >= 100 else np.mean(score_history)
            avg_history.append(avg_score)
            
            postfix_dict = {
                'episode': episode, 
                'score': f"{score:.1f}", 
                'avg_score': f"{avg_score:.1f}",
                'seed': episode_seed
            }
            
            if self.log_kl_divergence and self.kl_tracker and len(self.kl_tracker.kl_values) > 0:
                kl_stats = self.kl_tracker.get_stats()
                postfix_dict['kl_latest'] = f"{kl_stats['latest']:.4f}"
                postfix_dict['kl_mean'] = f"{kl_stats['mean']:.4f}"
            
            pbar.set_postfix(postfix_dict)
            
            if episode > 0 and episode % 50 == 0:
                self.save_model(checkpoint_dir, f"model_ep{episode}")
                
                if self.log_kl_divergence and self.kl_tracker and len(self.kl_tracker.kl_values) > 0:
                    self.kl_tracker.plot_kl_evolution(f"{checkpoint_dir}/kl_evolution_ep{episode}.png")
                    self.kl_tracker.plot_kl_by_episode(f"{checkpoint_dir}/kl_by_episode_ep{episode}.png")
            
            if episode > 0 and episode % self.config.record_gif_every == 0:
                try:
                    print(f"\nCreating GIF for episode {episode}...")
                    gif_score = self.create_gif(episode, checkpoint_dir)
                    if gif_score is not None:
                        print(f"GIF created with score {gif_score:.1f}")
                except Exception as e:
                    print(f"Error creating GIF: {e}")
        
        self.save_model(checkpoint_dir, "final_model")
        
        try:
            print("\nCreating final GIF...")
            self.create_gif(episode, checkpoint_dir)
        except Exception as e:
            print(f"Error creating final GIF: {e}")
        
        if self.log_kl_divergence and self.kl_tracker and len(self.kl_tracker.kl_values) > 0:
            self.kl_tracker.plot_kl_evolution(f"{checkpoint_dir}/kl_evolution_final.png")
            self.kl_tracker.plot_kl_by_episode(f"{checkpoint_dir}/kl_by_episode_final.png")
        
        self.plot_results(score_history, avg_history, checkpoint_dir)
        
        return score_history, avg_history
    
    def update_policy(self, continue_training=None, current_episode=0):
        self.update_count += 1
        is_first_update = (self.update_count == 1)
        
        states = torch.FloatTensor(np.array(self.memory.states))
        actions = torch.FloatTensor(np.array(self.memory.actions))
        old_logprobs = torch.FloatTensor(np.array(self.memory.logprobs))
        rewards = torch.FloatTensor(np.array(self.memory.rewards))
        values = torch.FloatTensor(np.array(self.memory.values))
        dones = torch.FloatTensor(np.array(self.memory.dones))
        
        """
        taking final state value for bootstrapping
        """
        with torch.no_grad():
            next_value = self.critic(self.prepare_tensor(self.memory.states[-1]))
        
        advantages, returns = GAE.compute_gae(
            values, next_value, dones,
            self.config.gamma, rewards,
            self.config.gae_lambda
        )
        
        for epoch in range(self.config.update_epochs):
            is_first_epoch = (epoch == 0 and is_first_update)
            
            if self.use_kl_early_stopping and continue_training is not None and not continue_training:
                break
                
            minibatches = self.rollout(advantages, returns)
            
            for batch_idx, batch in enumerate(minibatches):
                is_first_batch = (batch_idx == 0 and is_first_epoch)
                
                if self.use_kl_early_stopping and continue_training is not None and not continue_training:
                    break
                    
                mb_states = self.prepare_tensor(batch['states'])
                mb_actions = self.prepare_tensor(batch['actions'])
                mb_old_logprobs = self.prepare_tensor(batch['logprobs'])
                mb_advantages = self.prepare_tensor(batch['advantages'])
                mb_returns = self.prepare_tensor(batch['returns'])
                mb_old_values = self.prepare_tensor(batch['values'])
                
                new_logprobs, new_values, dist = self.policy.evaluate(
                    self.actor, self.critic, mb_states, mb_actions
                )
                
                policy_loss, prob_ratio = self.clipped_surrogate_objective(new_logprobs, mb_old_logprobs, mb_advantages)
                policy_loss = -policy_loss  
                
                entropy_loss = self.entropy(dist)
                actor_loss = policy_loss + entropy_loss
                
                value_loss = self.value_objective(new_values, mb_returns, mb_old_values)
                
                if is_first_batch and self.verify_first_update_ratio:
                    ratio_stats = self.ratio_tracker.log_ratios(prob_ratio, self.update_count, epoch, batch_idx)
                    min_ratio, max_ratio, mean_ratio, std_ratio = ratio_stats
                    
                    """
                    Verify ratios are close to 1.0 in first update
                    """
                    if is_first_epoch:
                        epsilon = 1e-5  
                        if abs(mean_ratio - 1.0) > epsilon or std_ratio > epsilon:
                            print("\n===== WARNING: POLICY RATIO SANITY CHECK FAILED =====")
                            print(f"Policy ratios in first update should be 1.0, but found:")
                            print(f"Min: {min_ratio:.6f}, Max: {max_ratio:.6f}")
                            print(f"Mean: {mean_ratio:.6f}, Std: {std_ratio:.6f}")
                            print("This indicates a potential bug in the PPO implementation.")
                            print("The new policy should equal the old policy in the first update.")
                            print("Check your action probability calculation and storage.")
                            print("=====================================================\n")
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                
                with torch.no_grad():
                    kl_div = self.calculate_kl_divergence(new_logprobs, mb_old_logprobs)
                    
                    if self.log_kl_divergence and self.kl_tracker:
                        self.kl_tracker.log_kl(kl_div, current_episode, self.update_count, epoch, batch_idx)
                    
                    """
                    check for early stopping based on KL, implementation is based on Schulman et.al.
                    """
                    if self.use_kl_early_stopping and continue_training is not None:
                        if self.target_kl is not None and kl_div > 1.5 * self.target_kl:
                            continue_training = False
                            if self.verbose >= 1:
                                print(f"Early stopping at KL divergence: {kl_div:.4f} (threshold: {1.5 * self.target_kl:.4f})")
                            break
        
        self.memory.clear()
        if self.use_kl_early_stopping and continue_training is not None:
            return continue_training
        return None
    
    def save_model(self, directory, name):
        path = f"{directory}/{name}.pt"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'random_state': random.getstate(),
            'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(),
            'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'random_state' in checkpoint:
            random.setstate(checkpoint['random_state'])
        if 'numpy_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_state'])
        if 'torch_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_state'])
        if 'cuda_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_state'])
            
        print(f"Model loaded from {path} with full random state restoration")
        
        return checkpoint.get('config', None)
    
    def plot_results(self, scores, avg_scores, directory):
        plt.figure(figsize=(10, 6))
        plt.plot(scores, label='Episode Reward')
        plt.plot(avg_scores, label='Average Reward (100 ep)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'PPO Training on {self.config.env_name} (seed={self.config.seed})')
        plt.legend()
        plt.savefig(f"{directory}/training_results.png")
        plt.close()

def main():
    cfg = config()
    
    set_seed_everywhere(cfg.seed)
    
    env = gym.make(cfg.env_name)
    cfg.num_observations = env.observation_space.shape[0]
    cfg.num_actions = env.action_space.shape[0]
    cfg.env = env
    
    print(f"Training on {cfg.env_name}")
    print(f"Observation space: {cfg.num_observations}")
    print(f"Action space: {cfg.num_actions}")
    print(f"Training for {cfg.episodes} episodes")
    print(f"KL early stopping: {'Enabled' if cfg.use_kl_early_stopping else 'Disabled'}")
    if cfg.use_kl_early_stopping:
        print(f"KL threshold: {cfg.target_kl} (stopping at {1.5 * cfg.target_kl})")
    print(f"KL divergence logging: {'Enabled' if cfg.log_kl_divergence else 'Disabled'}")
    print(f"First update ratio check: {'Enabled' if cfg.verify_first_update_ratio else 'Disabled'}")
    print(f"Creating GIFs every {cfg.record_gif_every} episodes")
    
    agent = PPO(cfg)
    scores, avg_scores = agent.train()
    
    env.close()

if __name__ == "__main__": 
    main()