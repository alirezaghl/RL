This repository contains a comprehensive implementation of Proximal Policy Optimization (PPO) for training agents in continuous control environments. 

![Walker2d Training Performance](https://github.com/alirezaghl/RL/blob/main/PPO-EXP/episode_19999_reward_4346.5.gif)


### Hyperparameters

The implementation uses the following key hyperparameters:

```python
gamma = 0.99         # Discount factor
gae_lambda = 0.95    # GAE lambda parameter
clip_coef = 0.2      # PPO clipping coefficient
vf_coef = 0.5        # Value function loss coefficient
ent_coef = 0.0       # Entropy coefficient
lr = 3e-4            # Learning rate
batch_size = 2048    # Steps collected per update
minibatch_size = 64  # Size of minibatches for updates
```


1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.

2. "Implementation Matters in Deep RL: A Case Study on PPO and TRPO." by Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry.

3. [Arena Chapter 2: PPO Implementation](https://arena-chapter2-rl.streamlit.app/[2.3]_PPO)

