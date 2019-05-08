# Curso de maestria: Inteligencia artificial (TEC), I Semestre 2019
# Author: Jafet Chaves Barrantes <jafet.a15@gmail.com>

# Back-Propagation Neural Networks
#
import math
import random
import operator
import time
import sys
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

# Find the length of a given text file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    f.close()
    return i + 1

def map_index(argument):
    switcher = {
        0: "one",
        1: "two",
        2: "three",
        3: "four",
        4: "five",
        5: "six",
        6: "seven",
        7: "eight",
        8: "nine",
        9: "zero",
    }
    return switcher.get(argument)

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

    def evaluate(self, inputs, weights_file):

        weights = self.readWeights(weights_file)
        in_weights = weights[0]
        out_weights = weights[1]

        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * in_weights[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * out_weights[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

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


    def test(self, input_data, weights_file):
        test_case = self.readInput(input_data)
        np.take(test_case,np.random.permutation(test_case.shape[0]),axis=0,out=test_case)
        for p in test_case:
            index, value = max(enumerate(self.evaluate(p[0], weights_file)), key=operator.itemgetter(1))
            number = map_index(index)
            print ("The number is: %s with a certainty of: %f" % (number,value))

    def readWeights(self,weights_file):

        weights = []
        header = []
        header2 = []
        content = []
        content2 = []
        matrices = []
        matrices2 = []

        input_file = open(weights_file,'r')

        lines=[1]

        for i in range(1,file_len(weights_file)+1):
            line = input_file.readline()
            if i not in lines:
                content.append(line)
            elif i in lines:
                header.append(line)

        content = [x.strip() for x in content]
        header = [x.strip() for x in header]
        for a in content:
            if a:
                content2.append(a)

        for a in header:
            if a:
                header2.append(a)

        matrices.append(content2[0:65])
        matrices.append(content2[65:129])

        for j in range(2):
            data_converted = []
            # Every row
            for text in matrices[j]:
                data_splited = []
                data_splited = text.split(',')
                for data in data_splited:
                    numbers = []
                    numbers_converted = []
                    numbers = data_splited[0].split(' ')
                    for number in numbers:
                        number_converted=float(number)
                        numbers_converted.append(number_converted)
                data_converted.append(numbers_converted)
            weights.append(data_converted)

        return weights

    def readInput(self,input_data):
        data_base = []
        reference=[]
        reference2=[]
        content = []
        digits = []
        digits2 = []
        tags = []
        content2 = []

        input_file = open(input_data,'r')

        lines=[10,21,32,43,54,65,76,87,98,109]

        for i in range(1,file_len(input_data)+1):
            line = input_file.readline()
            if i not in lines:
                content.append(line)
            elif i in lines:
                reference.append(line)

        #Parse the file
        content = [x.strip() for x in content]
        reference = [x.strip() for x in reference]
        for a in content:
            if a:
                content2.append(a)

        for a in reference:
            if a:
                reference2.append(a)

        digits.append(content2[0:8])
        digits.append(content2[8:16])
        digits.append(content2[16:24])
        digits.append(content2[24:32])
        digits.append(content2[32:40])
        digits.append(content2[40:48])
        digits.append(content2[48:56])
        digits.append(content2[56:64])
        digits.append(content2[64:72])
        digits.append(content2[72:80])

        for j in range(10):
            digits[j]="".join(digits[j])
            data2 = []
            for element in digits[j]:
                if element != ',':
                    element=int(element)
                    data2.append(element)
            digits2.append(data2)

        digits2.append(data2)

        for m in range(10):
            sequence = []
            for element in reference2[m]:
                if element != ',':
                    element=int(element)
                    sequence.append(element)
            tags.append(sequence)

        for k in range(10):
            a = [digits2[k],tags[k]]
            data_base.append(a)

        #Convert to NumPy array
        data_base = np.array(data_base)

        return data_base

    def writeOutput(self,output_data):
        output_file = open(output_data,'w')
        L = [str(len(self.wi)-1)+" ",str(len(self.wi[0]))+" ",str(len(self.wo[0]))+"\n\n" ]
        output_file.writelines(L)
        output_file.close()

        output_file = open(output_data,'ab')
        np.savetxt(output_file,self.wi,fmt='%.10f')
        output_file.close()

        output_file = open(output_data,'a+')
        output_file.write("\n")
        output_file.close()

        output_file = open(output_data,'ab')
        np.savetxt(output_file,self.wo,fmt='%.10f')
        output_file.close()

    def train(self, input_data, output_data, max_iterations=1000, eta=0.5, min_error=1.0):
        start = time.time()
        training_data = self.readInput(input_data)
        for i in range(max_iterations):
            error = 0.0
            # Randomly shuffle the training data
            np.take(training_data,np.random.permutation(training_data.shape[0]),axis=0,out=training_data)
            for p in training_data:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, eta)
            # ~ if i % 100 == 0: #Print error only iterations/100 times
            print("Training the neural network, %d/%d iterations..." % (i,max_iterations), end="\r")
            if error < min_error:
                print ("Reached minimum error criteria")
                break
        print ("Reached max iterations")
        # Create output file with the weights
        self.writeOutput(output_data)
        end = time.time()
        print("*****************************************")
        print("Time elapsed during training: %f" %(end - start))
        print("Error: %f" % error)
        print("*****************************************")


def demo():
    # Create a basic NN (3 layers)
    # NN(self, #nodos de entrada, #nodos de la capa oculta, #nodos de salida)
    n = NN(64, 64, 10)
    input_data="data.txt"
    output_data="output.txt"
    # Train it with some patterns
    n.train(input_data,output_data,1000,0.5,0.001)
    # Test it
    n.readWeights(output_data)
    n.test(input_data,output_data)

if __name__ == '__main__':
    demo()
