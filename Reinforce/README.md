## Overview

REINFORCE is a policy gradient method that directly optimizes the policy without using a value function. This implementation includes:

1. **Standard REINFORCE**: Vanilla implementation using Monte Carlo returns
2. **REINFORCE with Baseline**: An improved version that uses a value function to reduce variance

The baseline version typically converges faster and more stably because it reduces the variance of the policy gradient.


## Usage

```bash
# Run both versions (with and without baseline) for comparison
python reinforce.py --episodes 1000

# Run only the baseline version
python reinforce.py --use_baseline --episodes 1000
```


## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--env` | Gym environment name | CartPole-v1 |
| `--input_dim` | Input dimension for networks | 4 |
| `--output_dim` | Output dimension (actions) | 2 |
| `--hidden_sizes` | Policy network hidden layer sizes | 256,128 |
| `--value_hidden_sizes` | Value network hidden layer sizes | 128,64 |
| `--lr` | Learning rate | 3e-4 |
| `--gamma` | Discount factor | 0.99 |
| `--episodes` | Number of training episodes | 1000 |
| `--log_interval` | Episodes between logging | 10 |
| `--use_baseline` | Use value function baseline | False |
| `--run_both` | Run both with and without baseline | True |
| `--seed` | Random seed | 42 |
