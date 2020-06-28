import pickle, random
import numpy as np
import sys

sys.path.append('..')
from neural import NeuralNetwork
from visualization import Plot

with open("numbers.pickle", 'rb') as handle:
    name, weights = pickle.load(handle)

with np.load("numbers.npz") as data:
    INPUTS = data['training_images']
    OUTPUTS = data['training_labels']

nn = NeuralNetwork(name)
nn.load(weights)

#"""
plot = Plot(nn, INPUTS[0])
while True:
    current = random.randint(0, len(INPUTS))
    nn.predict(INPUTS[current])
    plot.plotNetwork(INPUTS[current])
#"""

"""
nn.backpropagate(INPUTS, OUTPUTS, 0.25, 1)
weights = [ [ neuron.weights for neuron in layer.neurons ] for layer in nn.layers ]
with open("numbers.pickle", 'wb') as handle:
    pickle.dump(("numbers", weights), handle)
#"""

"""
nn.accuracy(INPUTS, OUTPUTS, 100)
#"""
