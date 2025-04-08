## Overview

PPO is an on-policy algorithm that uses clipped surrogate objectives to improve training stability.


## Usage


```bash
# Run PPO with default parameters
python ppo.py

# Run with custom parameters
python ppo.py --episodes 2000 --horizon 2048 --gamma 0.99
```

### Key Parameters

```bash
# Change network architecture
python ppo.py --fc1-dims 512 --fc2-dims 256

# Adjust learning rate and batch size
python ppo.py --alpha 3e-4 --batch-size 128
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--env` | Gymnasium environment | HalfCheetah-v4 |
| `--episodes` | Number of training episodes | 2000 |
| `--horizon` | Steps before update | 2048 |
| `--gamma` | Discount factor | 0.99 |
| `--alpha` | Learning rate | 3e-4 |
| `--gae-lambda` | GAE lambda parameter | 0.95 |
| `--policy-clip` | Policy clip range | 0.2 |
| `--batch-size` | Batch size | 64 |
| `--n-epochs` | PPO epochs per update | 10 |
| `--fc1-dims` | First hidden layer size | 256 |
| `--fc2-dims` | Second hidden layer size | 256 |
| `--seed` | Random seed | 42 |


## Visualization

The implementation saves performance videos periodically and generates learning curve plots to track progress.
