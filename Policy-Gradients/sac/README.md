## Overview

SAC is an off-policy actor-critic algorithm that optimizes a stochastic policy using entropy regularization.



## Usage

```bash
# Run SAC with default parameters
python sac.py

# Run with custom parameters
python sac.py --episodes 1000 --gamma 0.99 --hidden-dim 256
```

### Key Parameters

```bash
# Adjust learning parameters
python sac.py --lr 1e-3 --batch-size 512 --memory-size 2000000

# Control entropy regularization
python sac.py --alpha 0.1 --no-auto-entropy
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--env` | Gymnasium environment | HalfCheetah-v4 |
| `--episodes` | Number of training episodes | 1000 |
| `--hidden-dim` | Hidden dimension | 256 |
| `--lr` | Learning rate | 1e-3 |
| `--gamma` | Discount factor | 0.99 |
| `--tau` | Target network update rate | 0.005 |
| `--alpha` | Entropy coefficient | 0.2 |
| `--auto-entropy` | Use automatic entropy tuning | True |
| `--memory-size` | Replay buffer size | 1000000 |
| `--batch-size` | Batch size | 256 |
| `--update-rate` | Update rate (steps per update) | 1 |
| `--render-interval` | Interval for rendering | 100 |
| `--reward-log-interval` | Interval for logging rewards | 10 |
| `--seed` | Random seed | 42 |


## Visualization

The implementation periodically records training videos and generates learning curve plots to track progress.
