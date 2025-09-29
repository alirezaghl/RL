## Overview

DDPG is an off-policy algorithm that combines the ideas of DQN and policy gradients. It's designed for continuous action spaces and uses a deterministic policy with exploration noise.


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
