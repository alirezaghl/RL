# HalfCheetah Reinforcement Learning

This repository contains implementations of three policy gradient algorithms applied to the HalfCheetah-v4 environment:

- **DDPG** (Deep Deterministic Policy Gradient)
- **SAC** (Soft Actor-Critic)
- **PPO** (Proximal Policy Optimization)



## Algorithms

### DDPG (Deep Deterministic Policy Gradient)


#### Configuration
```
gamma = 0.99          # Discount factor
tau = 0.005           # Target network update rate
critic_lr = 0.001     # Critic learning rate
actor_lr = 0.0001     # Actor learning rate
batch_size = 100      # Batch size
memory_size = 1000000 # Replay buffer size
hidden_1 = 400        # Hidden layer 1 size
hidden_2 = 300        # Hidden layer 2 size
update_rate = 1       # How often to update networks
episodes = 1000       # Number of episodes
```

#### Performance
![DDPG HalfCheetah](results/ddpg/videos/episode_1000.gif)

### SAC 


#### Configuration
```
hidden_dim = 256      # Hidden dimension
lr = 1e-3             # Learning rate
gamma = 0.99          # Discount factor
tau = 0.005           # Target network update rate
alpha = 0.2           # Entropy coefficient
auto_entropy = True   # Use automatic entropy tuning
memory_size = 1000000 # Replay buffer size
batch_size = 256      # Batch size
update_rate = 1       # Update rate (steps per update)
episodes = 1000       # Number of episodes
```

#### Performance
![SAC HalfCheetah](results/sac/videos/episode_1000.gif)

### PPO


#### Configuration
```
gamma = 0.99          # Discount factor
alpha = 3e-4          # Learning rate
gae_lambda = 0.95     # GAE lambda
policy_clip = 0.2     # Policy clip
batch_size = 64       # Batch size
n_epochs = 10         # PPO epochs
horizon = 2048        # Steps before update
fc1_dims = 256        # First hidden layer size
fc2_dims = 256        # Second hidden layer size
episodes = 2000       # Number of episodes
```

#### Performance
![PPO HalfCheetah](results/ppo/videos/episode_2000.gif)

## Usage

To train an agent using one of the implemented algorithms:

```bash
# Train with DDPG
python algorithms/ddpg/ddpg.py --episodes 1000

# Train with SAC
python algorithms/sac/sac.py --episodes 1000 

# Train with PPO
python algorithms/ppo/ppo.py --episodes 2000
```


## Requirements

- Python 3.7+
- PyTorch 2.0+
- Gymnasium 0.28+
- NumPy
- Matplotlib
- tqdm
- imageio
