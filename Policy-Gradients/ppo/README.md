## Overview

PPO is an on-policy algorithm that uses clipped surrogate objectives to improve training stability.


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
