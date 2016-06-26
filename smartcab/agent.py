import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import exp

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.params = {'epsilon': 1.0, 'alpha': 0.4, 'gamma' : 0.8}
        self.state = None
        self.Q_map = dict()      # Q_map is the mapping for Q values
        self.penalties = 0       # penalties is to record how many time the car break the traffic law
        self.initial_deadline = self.env.get_deadline(self)       # how many steps the agent has traveled before reaching the goal
        self.trial = 0    # the number of trials the agent has gones through
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.penalties = 0
        self.initial_deadline = self.env.get_deadline(self)
        self.trial += 1
    # Q_learn funciton is to calculate the Q_map elements    
    def Q_learn(self, action, next_state, reward):
        Qvalue = self.Q_map.get((self.state, action))
        if Qvalue == None:
            Qvalue = reward
        else:
            qs = [self.Q_map.get((next_state, i)) for i in Environment.valid_actions]
            next_q = max(qs)
            if next_q == None:
                next_q = 0.0

            Qvalue = (1 - self.params['alpha']) * Qvalue + self.params['alpha'] * (reward + self.params['gamma'] * next_q)

        self.Q_map[(self.state, action)] = Qvalue

    # choose_action function is to implement exploration and exploitation
    def choose_action(self, state):
        if random.random() <= self.params['epsilon'] * self.get_decay_rate():
            action = random.choice(Environment.valid_actions)
        else:
            qs = [self.Q_map.get((state, i)) for i in Environment.valid_actions]
            if qs[0] == None:
                return None
            q_max = max(qs)
            possible_actions = [key[-1] for key , Qvalue in self.Q_map.items() if Qvalue == q_max]
            if len(possible_actions) > 1:
                action = random.choice(list(set(possible_actions)))
            else:
                action = possible_actions[0]

        return action

    def get_decay_rate(self):
        return 1.0 / self.trial

    def is_close_to_end(self, deadline):
        if deadline > self.initial_deadline / 2.0:
            return 0
        else:
            return 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        #heading = self.env.agent_states[self]['heading']     
        close_to_end = self.is_close_to_end(deadline)

        self.state = (inputs['light'], close_to_end, self.next_waypoint)
        # TODO: Select action according to your policy
        action = self.choose_action(self.state)
        # to see the result of random action, uncomment the following line and comment the above line
        #action = random.choice(Environment.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)
        if(reward < 0):
            self.penalties += 1

        # TODO: Learn policy based on state, action, reward
        inputs = self.env.sense(self)
        close_to_end = self.is_close_to_end(deadline - 1)
        next_state = (inputs['light'], close_to_end, self.planner.next_waypoint)
        self.Q_learn(action, next_state, reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()

    data = pd.read_csv('output.csv', sep = '\t', header = None, names = ['deadline_ratio', 'penalties','success'])
    success_data = data[data['success'] > 0]
    not_success_data = data[data['success'] < 1]

    fig = plt.figure(1)
    plt.subplot(311)
    plt.plot(not_success_data.index.values, not_success_data['success'], 'ro', success_data.index.values,success_data['success'], 'bo')
    plt.yticks([-0.5, 0, 1, 1.5])
    plt.xlim([0, 100])
    plt.ylabel('Has reached?')

    plt.subplot(312)
    plt.plot(success_data.index.values, success_data['penalties'], 'bo')
    plt.ylabel('Penalties')
    plt.xlim([0, 100])

    plt.subplot(313)
    plt.plot(success_data.index.values, success_data['deadline_ratio'], 'bo')
    plt.ylabel('Deadline ratio')
    plt.xlim([0, 100])
    plt.ylim([-0.2, 1.2])

    fig.suptitle('epsilon: 1.0, alpha: 0.4, gamma: 0.8, reaches: {}'.format(data['success'].sum()), fontsize=14)

    plt.show()

    os.remove('output.csv')


