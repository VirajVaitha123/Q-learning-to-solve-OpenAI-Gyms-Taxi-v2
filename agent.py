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