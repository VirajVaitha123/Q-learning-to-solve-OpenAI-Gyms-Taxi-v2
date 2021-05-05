import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA       : number of actions available to the agent
        - Q        : Q table storing rewards for each state action pair
        - eps      : starting value for the epsilon greedy policy
        - eps_decay: magnitude to decrease epilson value on each iteration
        - eps_min  : lowest value eps will reach
        - alpha    : learning rate determining the influence of new experiences on previous Q-table values
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 1
        self.eps_decay = 0.99
        self.eps_min = 0.005
        self.alpha = 0.90
        self.gamma = 1


    def select_action(self, state):
        """ Given the state, select an action based on the epsilon greedy policy.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.eps = max(self.eps*self.eps_decay, self.eps_min)   #decay eps over time to encourage exploitation over exploration
        if random.random() > self.eps:                          #Random generator - if this is more than eps, exploit the best strategy
            return np.argmax(self.Q[state])                     #Provide the action with the highest reward
        else:
            return random.choice(np.arange(self.nA))            #select a random action encouraging exploration of states
    

    