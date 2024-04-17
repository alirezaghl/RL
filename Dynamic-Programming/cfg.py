n = 100
v = np.zeros(n+1) #Creating dummy vector for saving value states
v[100] = 1 #Based on the text 'The reward is zero on all transitions except those on which the gambler reaches his goal, when it is +1.'
q = np.zeros((n+1, n+1)) #Creating action-state matrix
pi_star = np.ones(n+1)   #Creating dummy vector for saving optimal action in each state
nepochs = 100