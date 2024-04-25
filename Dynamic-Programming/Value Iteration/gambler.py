import numpy as np
import matplotlib.pyplot as plt
from cfg import n, nepochs

v = np.zeros(n+1) #Creating dummy vector for saving value states
v[100] = 1 #Based on the text 'The reward is zero on all transitions except those on which the gambler reaches his goal, when it is +1.'
q = np.zeros((n+1, n+1)) #Creating action-state matrix
pi_star = np.ones(n+1)   #Creating dummy vector for saving optimal action in each state

def value_iteration(p_head):
  for i in range(nepochs):
    for s in range(1, n):
     v[s] = np.max(q[s,:]) #According to Bellman Optimality Equation for v∗
     pi_star[s] = np.argmax(q[s,:]) #An optimal policy can be found by maximising over q∗(s, a)
     for a in range(1, min(s, n - s) + 1):
      q[s,a] = p_head * v[s+a] + (1-p_head) * v[s-a] #Calculate the value for all action in a given state





  x = range(n+1)
  plt.plot(x,v)
  plt.grid(True)
  plt.xlabel('Capital')
  plt.ylabel('Value Estimates')
  plt.show()

  x = range(n+1)
  y = pi_star
  plt.bar(x, y, align='center', alpha=0.5)
  plt.xlabel('Capital')
  plt.ylabel('Final policy (stake)')
  plt.show()

  return q[s,a], pi_star
