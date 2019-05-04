# Curso de maestria: Inteligencia artificial (TEC), I Semestre 2019
# Author: Jafet Chaves Barrantes <jafet.a15@gmail.com>

# Back-Propagation Neural Networks
#
import math
import random
import numpy as np

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# The sigmoid function is the standard 1/(1+e^-x)
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of the sigmoid function, in terms of the output variable
def dsigmoid(y):
    return y*(1-y)

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, eta):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output layer
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden layer
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + eta*change
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + eta*change
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def writeOutput(self,output_data):
        with open(output_data,'w') as f:
            for line in self.wi:
                np.savetxt(f, line, fmt='%.sf')

    def train(self, input_data, output_data, max_iterations=1000, eta=0.5, min_error=1.0):
        for i in range(max_iterations):
            error = 0.0
            # Randomly shuffle the training data
            np.take(input_data,np.random.permutation(input_data.shape[0]),axis=0,out=input_data)
            for p in input_data:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, eta)
            if i % 100 == 0: #Print error only iterations/100 times
                print('error %-.5f' % error)
            if error < min_error:
                print ("Reached minimum error criteria")
                break
        print ("Reached max iterations")
        # Create output file with the weights
        # ~ self.writeOutput(output_data)


def demo():

    data_base = [
        [[0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,
        0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,
        0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,
        0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0], [1,0,0,0,0,0,0,0,0,0]],
        [[0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,0,
        0,1,1,0,0,0,1,1,0,0,0,0,0,1,1,0,
        0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,
        0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1], [0,1,0,0,0,0,0,0,0,0]],
        [[0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
        0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,
        0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,1,0,0,1,1,1,1,1,0], [0,0,1,0,0,0,0,0,0,0]],
        [[0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,
        0,1,1,0,1,1,0,0,0,1,1,1,1,1,0,0,
        0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,
        0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0], [0,0,0,1,0,0,0,0,0,0]],
        [[0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,
        0,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,
        0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,
        0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0], [0,0,0,0,1,0,0,0,0,0]],
        [[0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,
        0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,
        0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,
        0,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0], [0,0,0,0,0,1,0,0,0,0]],
        [[0,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,
        0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0], [0,0,0,0,0,0,1,0,0,0]],
        [[0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,
        0,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,
        0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,
        0,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0], [0,0,0,0,0,0,0,1,0,0]],
        [[0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,
        0,1,1,0,0,0,1,1,0,0,1,1,1,1,1,1,
        0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,
        0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0], [0,0,0,0,0,0,0,0,1,0]],
        [[0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,
        0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,
        0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,
        0,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0], [0,0,0,0,0,0,0,0,0,1]]
    ]

    #Convert to NumPy array
    data_base = np.array(data_base)

    test_case = np.copy(data_base)
    np.take(test_case,np.random.permutation(test_case.shape[0]),axis=0,out=test_case)

    # Create a basic NN (3 layers)
    # NN(self, #nodos de entrada, #nodos de la capa oculta, #nodos de salida)
    n = NN(len(data_base[0][0]), len(data_base[0][0]), len(data_base[0][1]))
    # Train it with some patterns
    n.train(data_base, "output.txt", 1000, 0.5, 0.001)
    # Test it
    n.test(test_case)

if __name__ == '__main__':
    demo()
