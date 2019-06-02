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

class PolicyIteration():

    capacity = 20
    rental_reward = 10.
    moving_cost = 1.
    max_moving = 5

    bad_action_cost = 1000

    REQ_EXPECT_1 = 3
    REQ_EXPECT_2 = 4
    RET_EXPECT_1 = 3
    RET_EXPECT_2 = 2

    discount = 0.9

    policy_evaluation_error = 0.01

    policy = None
    value = None

    def __init__(self):
        self.policy = np.zeros([self.capacity + 1]*2, int)
        self.funct_value = np.zeros([self.capacity + 1]*2)

        self._reward1 = self.expected_rental_reward(self.REQ_EXPECT_1)
        self._reward2 = self.expected_rental_reward(self.REQ_EXPECT_2)

        assert self.bad_action_cost >= 0

    def estimate_action_value(self, action, s1, s2):
        transition_prob1 = self.transition_probability(s1, self.REQ_EXPECT_1, self.RET_EXPECT_1, -action)
        transition_prob2 = self.transition_probability(s2, self.REQ_EXPECT_2, self.RET_EXPECT_2, action)
        transition_prob = np.outer(transition_prob1, transition_prob2)

        estimated_value = self._reward1[s1] + self._reward2[s2] - self.expected_moving_cost(s1, s2, action) + \
               self.discount * sum((transition_prob * self.funct_value).flat)

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
            if delta < self.policy_evaluation_error:
                break

    # policy improvement
    def policy_improve(self):
        is_policy_changed = False

        print("Improving Policy")

        policy_iterator = np.nditer([self.policy], flags=['multi_index'])

        action_array = np.arange(-self.max_moving, self.max_moving + 1)

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

    def expected_moving_cost(self, s1, s2, action):
        if action == 0:
            return 0.

        # moving from state s1 into state s2
        if action > 0:
            probability = self.transition_probability(s1, self.REQ_EXPECT_1, self.RET_EXPECT_1)
            cost = self.gen_move_cost_array(action)
            return cost.dot(probability)

        # moving from state s2 into state s1
        probability = self.transition_probability(s2, self.REQ_EXPECT_2, self.RET_EXPECT_2)
        to_rental1 = True
        cost = self.gen_move_cost_array(action, to_rental1)
        return cost.dot(probability)

    def gen_move_cost_array(self, action, return_flag=False):

        abs_action = abs(action)

        # bad action is punished
        # +1 because one employee is happy to return one car for free to rental 1
        if return_flag == True:
            cost = np.asarray(
                [self.bad_action_cost if ii < abs_action+1
                 else abs_action for ii in range(self.capacity + 1)]
            ) + self.moving_cost
        else:
            cost = np.asarray(
                [self.bad_action_cost if ii < abs_action else abs_action for ii in range(self.capacity + 1)]
            ) + self.moving_cost

        return cost

    def expected_rental_reward(self, expected_request):
        return np.asarray([self.state_reward(s, expected_request) for s in range(self.capacity + 1)])

    def state_reward(self, s, lam):
        rewards = self.rental_reward * np.arange(s + 1)
        probability = compute_poisson(lam, s)
        return rewards.dot(probability)

    def transition_probability(self, s, req, ret, action=0):

        car_returns_size = self.max_moving + self.capacity

        p_req = compute_poisson(req, s)
        p_ret = compute_poisson(ret, car_returns_size)
        probability = np.outer(p_req, p_ret)

        transition_prob = np.asarray([probability.trace(offset) for offset in range(-s, car_returns_size + 1)])

        assert abs(action) <= self.max_moving, "action can be large than %s." % self.max_moving

        # No cars are being moved
        if action == 0:
            transition_prob[20] += sum(transition_prob[21:])
            return transition_prob[:21]

        # Move cars from rental 1 to rental 2
        if action > 0:
            transition_prob[self.capacity-action] += sum(transition_prob[self.capacity-action+1:])
            transition_prob[self.capacity-action+1:] = 0

            return np.roll(transition_prob, shift=action)[:self.capacity+1]

        # Move cars from rental 2 to rental 1
        action = -action
        transition_prob[action] += sum(transition_prob[:action])
        transition_prob[:action] = 0

        transition_prob[action+self.capacity] += sum(transition_prob[action+self.capacity+1:])
        transition_prob[action+self.capacity+1:] = 0

        return np.roll(transition_prob, shift=-action)[:self.capacity+1]

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

