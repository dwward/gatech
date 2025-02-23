"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: dward45 (replace with your User ID)
GT ID: 903271210 (replace with your GT ID)
"""

import numpy as np


class QLearner(object):

    def __init__(self,
                 num_states=100,  # 10x10 square
                 num_actions=4,  # up,down,left,right
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.state = 0
        self.action = 0
        self.q = np.zeros((num_states, num_actions))
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.rar = rar  # random action rate
        self.radr = radr  # random action decay rate
        self.dyna = dyna  # dyna trips
        self.t_c = np.zeros((num_states, num_actions, num_states))  # dyna transformation matrix
        self.r = np.zeros((num_states, num_actions))  # dyna reward matrix

    def author(self):
        return 'dward45'  # replace tb34 with your Georgia Tech username.

    # set the state and get first action
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.state = s
        action = np.random.randint(0, self.num_actions)
        if self.verbose:
            print "s =", s, "a =", action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state you're in
        @param r: The reward for the last action you took
        @returns: The selected action
        """
        q = self.q
        s = self.state
        a = self.action
        g = self.gamma
        alph = self.alpha

        # DETERMINE ACTION W/ RANDOM CHANCE
        # -----------------------------------
        chance = np.random.uniform()
        if chance < self.rar:
            a_prime = np.random.randint(0, self.num_actions)
        else:
            a_prime = np.argmax(q[s_prime])

        self.rar = self.rar * self.radr

        # UPDATE Q TABLE:
        # Q'[s, a] = (1-alpha)*Q[s, a] + alpha*(r + (gamma * Q[s', argmaxa'(Q[s', a'])])
        # -----------------------------------
        previous_estimate = q[s, a]
        improved_estimate = (r + (g * q[s_prime, a_prime]))
        self.q[s, a] = (1 - alph) * previous_estimate + alph * improved_estimate

        # DYNA-Q ALGORITHM
        # -----------------------------------
        if self.dyna > 0:

            # Increment transformation matrix counter
            self.t_c[s, a, s_prime] = self.t_c[s, a, s_prime] + 1

            # Update reward
            self.r[s, a] = (1 - alph) * self.r[s, a] + alph * r

            random_state_list = np.random.randint(low=0, high=self.num_states, size=self.dyna)
            random_action_list = np.random.randint(low=0, high=self.num_actions, size=self.dyna)

            # EAT MUSHROOMS
            # -----------------------------------
            for i in range(0, self.dyna):

                # Choose a state and action randomly
                rand_s = random_state_list[i]
                rand_a = random_action_list[i]
                rand_s_primes = self.t_c[rand_s, rand_a, :]

                if np.sum(rand_s_primes) > 0:

                    # Determine the most picked s, a and the reward
                    rand_s_prime = np.argmax(rand_s_primes)
                    rand_a_prime = np.argmax(self.q[rand_s_prime, :])
                    rand_r = self.r[rand_s, rand_a]

                    # Update our q-table and reinforce the pathway
                    # Q'[s, a] = (1-alpha)*Q[s, a] + alpha*(r + (gamma * Q[s', argmaxa'(Q[s', a'])])
                    previous_estimate = self.q[rand_s, rand_a]
                    improved_estimate = (rand_r + (g * self.q[rand_s_prime, rand_a_prime]))
                    self.q[rand_s, rand_a] = (1 - alph) * previous_estimate + alph * improved_estimate

        # REMEMBER CURRENT STATE/ACTION
        # -----------------------------------
        self.state = s_prime
        self.action = a_prime

        if self.verbose:
            print "s =", s_prime, "a =", a_prime, "r =", r

        return a_prime


if __name__ == "__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
