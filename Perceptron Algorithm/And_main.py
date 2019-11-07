import numpy as np
from Perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([0,0,0,0]))
training_inputs.append(np.array([0,0,0,1]))
training_inputs.append(np.array([0,0,1,0]))
training_inputs.append(np.array([0,0,1,1]))
training_inputs.append(np.array([0,1,0,0]))
training_inputs.append(np.array([0,1,0,1]))
training_inputs.append(np.array([0,1,1,0]))
training_inputs.append(np.array([0,1,1,1]))
training_inputs.append(np.array([1,0,0,0]))
training_inputs.append(np.array([1,0,0,1]))
training_inputs.append(np.array([1,0,1,0]))
training_inputs.append(np.array([1,0,1,1]))
training_inputs.append(np.array([1,1,0,0]))
training_inputs.append(np.array([1,1,0,1]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([1,1,1,1]))

output = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])

percept = Perceptron(4)
percept.train(training_inputs, output)

for _ in training_inputs:
    print(percept.predict(_))