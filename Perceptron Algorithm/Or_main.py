import numpy as np
from Perceptron import Perceptron

training_input = []
training_input.append(np.array([0,0]))
training_input.append(np.array([0,1]))
training_input.append(np.array([1,0]))
training_input.append(np.array([1,1]))

output = np.array([0,1,1,1])

percept = Perceptron(2)
percept.train(training_input, output)

for _ in training_input:
    print(percept.predict(_))