## Overview

DDPG is an off-policy algorithm that combines the ideas of DQN and policy gradients. It's designed for continuous action spaces and uses a deterministic policy with exploration noise.



## Usage

```bash
# Run DDPG with default parameters
python ddpg.py

# Run with custom parameters
python ddpg.py --episodes 1000 --gamma 0.99
```

### Key Parameters

```bash
# Adjust network architecture
python ddpg.py --hidden_1 512 --hidden_2 256

# Modify learning rates
python ddpg.py --critic_lr 0.001 --actor_lr 0.0001
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--env` | Gym environment name | HalfCheetah-v4 |
| `--gamma` | Discount factor | 0.99 |
| `--tau` | Target network update rate | 0.005 |
| `--critic_lr` | Critic learning rate | 0.001 |
| `--actor_lr` | Actor learning rate | 0.0001 |
| `--batch_size` | Batch size | 100 |
| `--memory_size` | Replay buffer size | 1000000 |
| `--hidden_1` | Hidden layer 1 size | 400 |
| `--hidden_2` | Hidden layer 2 size | 300 |
| `--update_rate` | How often to update networks | 1 |
| `--episodes` | Number of episodes | 1000 |
| `--render_interval` | Render every N episodes | 100 |
| `--seed` | Random seed | 42 |




## Visualization

The implementation records GIFs of agent performance and generates learning curves to visualize progress over time.
