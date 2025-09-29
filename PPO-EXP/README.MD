While I had previously created a simpler PPO implementation, the growing popularity of this algorithm in developing reasoning models (although differently than in robotic control applications) motivated me to explore its details further. This investigation led me to the valuable resources listed in the references section, which I then used directly to design and implement this better version of PPO.
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


1. [Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)

2. [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

3. [Arena Chapter 2: PPO Implementation](https://arena-chapter2-rl.streamlit.app/[2.3]_PPO)

4. [What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study](https://openreview.net/forum?id=nIAxjsniDzg)

