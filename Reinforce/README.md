REINFORCE is a policy gradient method that directly optimizes the policy without using a value function. This implementation includes:

1. **Standard REINFORCE**: vanilla implementation using Monte Carlo returns
2. **REINFORCE with Baseline**: an improved version that uses a value function to reduce variance

the baseline version typically converges faster and more stably because it reduces the variance of the policy gradient.

## results
![REINFORCE Results](https://github.com/alirezaghl/RL/blob/main/Reinforce/results/reinforce.png)

## parameters

| parameter | description | default |
|-----------|-------------|---------|
| `--env` | gym environment name | CartPole-v1 |
| `--input_dim` | input dimension for networks | 4 |
| `--output_dim` | output dimension (actions) | 2 |
| `--hidden_sizes` | policy network hidden layer sizes | 256,128 |
| `--value_hidden_sizes` | value network hidden layer sizes | 128,64 |
| `--lr` | learning rate | 3e-4 |
| `--gamma` | discount factor | 0.99 |
| `--episodes` | number of training episodes | 1000 |
| `--log_interval` | episodes between logging | 10 |
| `--use_baseline` | use value function baseline | False |
| `--run_both` | run both with and without baseline | True |
| `--seed` | random seed | 42 |
