import numpy as np

class Perceptron(object):

    def __init__(self, inputs, bias, maxIter = 1000, rate = 1):
        self.maxIter = maxIter
        self.rate = rate
        self.weights = np.array(inputs)
        self.bias = bias

    def predict(self, inputs):
        output = np.dot(inputs, self.weights) + self.bias
        if output >= 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, train_inputs, labels):
        for _ in range(self.maxIter):
            for inputs, label in zip(train_inputs, labels):
                prediction = self.predict(inputs)
                # np.add(self.weights, self.rate * (label - prediction) * inputs, out = self.weights, casting = "unsafe")
                self.weights = self.weights + self.rate * (label - prediction) * inputs
                self.bias = self.bias + self.rate * (label - prediction)
                # print(self.weights, self.bias)
            # print(self.weights, self.bias)