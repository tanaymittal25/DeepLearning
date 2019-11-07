import numpy as np
from New_Perceptron import Perceptron

training_input = []
training_input.append(np.array([0.05, 0.7]))
training_input.append(np.array([0.1, 1.0]))
training_input.append(np.array([0.25, 0.55]))
training_input.append(np.array([0.3, 0.95]))
training_input.append(np.array([0.45, 0.15]))
training_input.append(np.array([0.6, 0.3]))
training_input.append(np.array([0.7, 0.65]))
training_input.append(np.array([0.9, 0.4]))

output = np.array([1,0,1,0,1,1,0,0])

percept = Perceptron([1, 0], 0.2)
percept.train(training_input, output)

for _ in training_input:
    print(percept.predict(_))