import numpy as np
import random
from collections import defaultdict
from grid_world import GridWorld687

class ActorCriticAgent:
    def __init__(self, gridworld, alpha_theta=0.01, alpha_w=0.01):
        self.gridworld = gridworld
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.theta = defaultdict(lambda: np.zeros(len(self.gridworld.actions)))

        # When you create a defaultdict with float as its argument, like defaultdict(float), 
        # it means that any time you try to access a key that doesn't exist in the dictionary, 
        # the defaultdict will automatically create a new key with a default value. 
        # In this case, the default value will be the result of calling float() with no arguments, which is 0.0.


        self.w = defaultdict(float)
        self.I = 1

    def softmax_policy(self, state):
        h = np.array([self.theta[state][a] for a in self.gridworld.actions.keys()])
        exp_h = np.exp(h - np.max(h))  # for numerical stability
        policy = exp_h / exp_h.sum()
        return policy

    def choose_action(self, state):
        policy = self.softmax_policy(state)
        actions = list(self.gridworld.actions.keys())
        return np.random.choice(actions, p=policy)

    def update(self, state, action, reward, next_state, done):
        # Critic update (TD error and value function weights)
        V = self.w[state]
        V_next = self.w[next_state] if not done else 0
        td_error = reward + self.gridworld.gamma * V_next - V
        self.w[state] += self.alpha_w * td_error * self.I

        # Actor update (policy parameter)
        actions = list(self.gridworld.actions.keys())
        policy = self.softmax_policy(state)
        for a in actions:
            if a == action:
                self.theta[state][a] += self.alpha_theta * td_error * self.I * (1 - policy[actions.index(a)])
            else:
                self.theta[state][a] -= self.alpha_theta * td_error * self.I * policy[actions.index(a)]

        self.I *= self.gridworld.gamma

    def train_episode(self):
        state = self.gridworld.get_initial_state()
        done = False
        while not done:
            action = self.choose_action(state)
            next_state = self.gridworld.get_next_state(state, action)
            reward = self.gridworld.reward_fn(next_state)
            done = next_state in self.gridworld.terminal
            self.update(state, action, reward, next_state, done)
            state = next_state

# Initialize the GridWorld
gridworld = GridWorld687()

# Initialize the Actor-Critic Agent
agent = ActorCriticAgent(gridworld)

# Train the agent for a certain number of episodes
n_episodes = 1000
for episode in range(n_episodes):
    agent.train_episode()

# Let's check the learned policy from the agent
learned_policy = {s: agent.choose_action(s) for s in gridworld.states}
learned_policy