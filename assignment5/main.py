# Curso de maestria: Inteligencia artificial (TEC), I Semestre 2019
# Author: Jafet Chaves Barrantes <jafet.a15@gmail.com>

# Reinforcement Learning - Policy Iteration

import numpy as np
import math
from scipy.stats import poisson

# Probability mass function
poisson_pmf = {}
# 1 - cumulative distribution function
poisson_sf = {}

#Helper function, to compute poisson probabilities
def compute_poisson (lam, cutoff):
    poisson_pmf[lam] = poisson.pmf(np.arange(25 + 1), lam)
    poisson_sf[lam] = poisson.sf(np.arange(25 + 1), lam)
    pmf = np.copy(poisson_pmf[lam][:cutoff+1])
    pmf[-1] += poisson_sf[lam][cutoff]
    return pmf

#Model parameters
CAPACITY = 20
RENTAL_REWARD = 10.
MOVING_COST = 2
MAX_TRANSFERS = 5
FORBID_ACT_PUNISH = 1000
REQ_EXPECT_1 = 3
REQ_EXPECT_2 = 4
RET_EXPECT_1 = 3
RET_EXPECT_2 = 2
DISCOUNT = 0.9

POLICY_ACCURACY = 0.01

class PolicyIteration():

    policy = None
    value = None

    def __init__(self):
        self.policy = np.zeros([CAPACITY + 1]*2, int)
        self.funct_value = np.zeros([CAPACITY + 1]*2)

        self._reward1 = self.expected_rental_reward(REQ_EXPECT_1)
        self._reward2 = self.expected_rental_reward(REQ_EXPECT_2)

    def estimate_action_value(self, action, s1, s2):
        transition_prob1 = self.transition_probability(s1, REQ_EXPECT_1, RET_EXPECT_1, -action)
        transition_prob2 = self.transition_probability(s2, REQ_EXPECT_2, RET_EXPECT_2, action)
        transition_prob = np.outer(transition_prob1, transition_prob2)

        # Total expected reward
        total_reward = self._reward1[s1] + self._reward2[s2]
        estimated_value = total_reward - self.estimate_transfer_cost(s1, s2, action)
        estimated_value += DISCOUNT * sum((transition_prob * self.funct_value).flat)

        return estimated_value

    # policy evaluation
    def policy_evaluation(self):
        while True:
            delta = 0
            policy_iterator = np.nditer([self.policy], flags=['multi_index'])

            while not policy_iterator.finished:
                action = policy_iterator[0]
                i, j = policy_iterator.multi_index

                prev_funct_value = self.funct_value[i, j]

                self.funct_value[i, j] = self.estimate_action_value(action, i, j)

                delta = max(delta, abs(self.funct_value[i, j] - prev_funct_value))

                policy_iterator.iternext()

            print("Policy Evaluated, delta:", delta)
            if delta < POLICY_ACCURACY:
                break

    # policy improvement
    def policy_improve(self):
        is_policy_changed = False

        print("Improving Policy")

        policy_iterator = np.nditer([self.policy], flags=['multi_index'])

        action_array = np.arange(-MAX_TRANSFERS, MAX_TRANSFERS + 1)

        while not policy_iterator.finished:
            val = []
            i, j = policy_iterator.multi_index

            for action in action_array:
                estimated_value = self.estimate_action_value(action, i, j)
                val.append(estimated_value)

            pi_s =  action_array[np.argmax(val)]

            if self.policy[i, j] != pi_s:
                is_policy_changed = True
                self.policy[i, j] = pi_s

            policy_iterator.iternext()

        return is_policy_changed

    def estimate_transfer_cost(self, s1, s2, action):

        to_rental1 = None

        if action == 0:
            return 0.

        # Transfer cars to to rental 2
        if action > 0:
            probability = self.transition_probability(s1, REQ_EXPECT_1, RET_EXPECT_1)

        # Transfer cars to to rental 1
        if action < 0:
            probability = self.transition_probability(s2, REQ_EXPECT_2, RET_EXPECT_2)
            to_rental1 = True

        abs_action = abs(action)

        # forbidden action (moving more cars than permitted) is punished
        # +1 because one employee is happy to return one car for free to rental 1
        if to_rental1 == True:
            cost = np.asarray(
                [FORBID_ACT_PUNISH if ii < abs_action+1
                 else abs_action for ii in range(CAPACITY + 1)]
            ) + MOVING_COST
        else:
            cost = np.asarray(
                [FORBID_ACT_PUNISH if ii < abs_action else abs_action for ii in range(CAPACITY + 1)]
            ) + MOVING_COST

        return cost.dot(probability)

    def expected_rental_reward(self, expected_request):
        return np.asarray([self.state_reward(s, expected_request) for s in range(CAPACITY + 1)])

    def state_reward(self, s, lam):
        rewards = RENTAL_REWARD * np.arange(s + 1)
        probability = compute_poisson(lam, s)
        return rewards.dot(probability)

    def transition_probability(self, s, req, ret, action=0):

        car_returns_size = MAX_TRANSFERS + CAPACITY

        p_req = compute_poisson(req, s)
        p_ret = compute_poisson(ret, car_returns_size)
        probability = np.outer(p_req, p_ret)

        transition_prob = np.asarray([probability.trace(offset) for offset in range(-s, car_returns_size + 1)])

        # No cars are being moved
        if action == 0:
            transition_prob[20] += sum(transition_prob[21:])
            return transition_prob[:21]

        # Move cars from rental 1 to rental 2
        if action > 0:
            transition_prob[CAPACITY-action] += sum(transition_prob[CAPACITY-action+1:])
            transition_prob[CAPACITY-action+1:] = 0

            return np.roll(transition_prob, shift=action)[:CAPACITY+1]

        # Move cars from rental 2 to rental 1
        action = -action
        transition_prob[action] += sum(transition_prob[:action])
        transition_prob[:action] = 0

        transition_prob[action+CAPACITY] += sum(transition_prob[action+CAPACITY+1:])
        transition_prob[action+CAPACITY+1:] = 0

        return np.roll(transition_prob, shift=-action)[:CAPACITY+1]

    def policy_iteration(self):

        self.policy_evaluation()
        while self.policy_improve():
            print("****************")
            print("Policy Improved")
            print("****************")
            self.policy_evaluation()

def demo():

    car_rental_policy = PolicyIteration()

    car_rental_policy.policy_iteration()

    print("\n\n Policy Result =\n\n", car_rental_policy.policy)

if __name__ == '__main__':
    demo()

