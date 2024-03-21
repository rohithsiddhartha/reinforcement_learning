import numpy as np
import random
class GridWorld687:
    def __init__(self, gamma=0.9):

        self.obstacles = [(2,2), (3,2)]
        self.water = [(4,2)]
        self.terminal = [(4,4)]
        self.actions = {'AU': (-1, 0), 'AD': (1, 0), 'AL': (0, -1), 'AR': (0, 1)}
        self.gamma = gamma
        self.states = [(r, c) for r in range(5) for c in range(5)]
        self.prob_specified = 0.8
        self.prob_specified_right = 0.05
        self.prob_specified_left = 0.05
        self.prob_stay = 0.1
        self.movement = {"prob_specified":0.8, "prob_specified_right":0.05,"prob_specified_left":0.05,"prob_stay":0.1}

        self.optimal_policy = [['AR','AR','AR','AD','AD'],
            ['AR','AR' ,'AR' ,'AD' ,'AD'],
            ['AU' ,'AU' ,'' ,'AD' ,'AD'],
            ['AU' ,'AU', '', 'AD', 'AD'],
            ['AU' ,'AU' ,'AR' ,'AR' ,'']]
        
        self.optimal_values = [[4.0187,4.5548,5.1575,5.8336,6.4553],
            [4.3716,5.0324,5.8013,6.6473,7.3907],
            [3.8672,4.3900 ,0.0000 ,7.5769 ,8.4637],
            [3.4182 ,3.8319 ,0.0000 ,8.5738 ,9.6946],
            [2.9976 ,2.9309, 6.0733, 9.6946, 0.0000]]


    def reward_fn(self, s_next):
        if s_next in self.terminal:
            return 10
        elif s_next in self.water:
            return -10
        return 0

    def is_valid_state(self, s):
        i,j = s
        if (i==2 and j==2) or (i==3 and j==2):
            return False
        #hitting boundary walls is also an obstacle
        elif j>=5 or i>=5:
            return False
        elif i<0 or j<0:
            return False
        return True
    
    def get_initial_state(self):
        valid_states = [s for s in self.grid.states if s not in self.grid.obstacles and s not in self.grid.terminal]
        return random.choice(valid_states)
    
    def get_action(self, s):
        r, c = s
        return self.optimal_policy[r][c]

    def get_movement(self):
        return np.random.choice(list(self.movement.keys()), p=list(self.movement.values()))
    

    def get_next_state(self, s, a):
        i,j = s
        movement = self.get_movement()
        if movement=="prob_specified":
            s_next = (i + self.actions[a][0], j +  self.actions[a][1])
        elif movement=="prob_specified_right":
            s_next = (i + self.actions[a][0], j - self.actions[a][1])
        elif movement=="prob_specified_left":
            s_next = (i - self.actions[a][0], j + self.actions[a][1])
        elif movement=="prob_stay":
            s_next = (i,j)
        if self.is_valid_state(s_next):
            return s_next
        return s
    
    def get_initial_state(self):
        valid_states = [s for s in self.states if s not in self.obstacles and s not in self.terminal]
        # print(random.choice(valid_states))
        return random.choice(valid_states)
    
    def get_possible_actions(self, s):
        return [action for action in self.actions.keys() if self.is_valid_state(self.get_next_state(s, action))]
