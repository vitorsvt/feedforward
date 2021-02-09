from visualization import Plot
import math, random, pickle, time
import numpy as np


class Neuron:
    def __init__(self, pos, is_out=False):
        self.pos = pos # Position in layer
        self.weights = [] # Weights for each connection
        self.output = None # Last output
        self.is_out = is_out # Is output neuron

    def attach(self, neurons):
        self.outputs = neurons # Next neurons

    def initialize(self, n_input):
        for i in range(n_input):
            self.weights.append(random.uniform(0,1))

    def update(self):
        self.weights = [new for new in self.updated_weights]

    def forward(self, row):
        self.inputs = [] # Input values
        activation = 0
        for weight, feature in zip(self.weights, row):
            self.inputs.append(feature)
            activation += weight*feature
        activation /= len(self.inputs)
        self.output = self.sigmoid(activation)
        return self.output

    def backward(self, rate, expected):
        if self.is_out:
            self.delta = (self.output - expected[self.pos]) * self.derSigmoid(self.output)
        else:
            delta_sum = 0
            cur_weight_index = self.pos
            for output_neuron in self.outputs:
                delta_sum += output_neuron.delta * output_neuron.weights[cur_weight_index]
            self.delta = delta_sum * self.derSigmoid(self.output)
        self.updated_weights = [] # New weight values
        for cur_weight, cur_input in zip(self.weights, self.inputs):
            gradient = self.delta * cur_input
            new_weight = cur_weight - rate*gradient
            self.updated_weights.append(new_weight)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def derSigmoid(x):
        return x * ( 1 - x )


class Layer:
    def __init__(self, n_neuron, is_out=False):
        self.is_out = is_out
        self.neurons = []
        for i in range(n_neuron):
            self.neurons.append( Neuron(i, is_out) )

    def attach(self, layer):
        for neuron in self.neurons:
            neuron.attach(layer.neurons)

    def initialize(self, n_input):
        for neuron in self.neurons:
            neuron.forward(n_input)

    def forward(self, row):
        activations = [neuron.forward(row) for neuron in self.neurons]
        return activations


class NeuralNetwork:
    def __init__(self, name):
        self.name = name
        self.layers = []

    def load(self, weights):
        self.output(len(weights[-1]))
        for neuron, weight in zip(self.layers[0].neurons, weights[-1]):
            neuron.weights = weight
        for layer in reversed(weights[:-1]):
            self.hidden(len(layer))
            for neuron, weight in zip(self.layers[0].neurons, layer):
                neuron.weights = weight

    def output(self, n_neuron):
        self.layers.insert(0, Layer(n_neuron, True))

    def hidden(self, n_neuron):
        hidden = Layer(n_neuron)
        hidden.attach(self.layers[0])
        self.layers.insert(0, hidden)

    def update(self, expected):
        for layer in reversed(self.layers):
            for neuron in layer.neurons:
                neuron.backward(self.rate, expected)
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update()

    def initialize(self, inputs):
        self.layers[0].initWeights(len(inputs))
        for i in range(1, len(self.layers)):
            n_input = len(self.layers[i-1].neurons)
            self.layers[i].initWeights(n_input)

    def predict(self, row):
        prediction = self.layers[0].forward(row)
        for i in range(1, len(self.layers)):
            prediction = self.layers[i].forward(prediction)
        return prediction

    def backpropagate(self, inputs, outputs, rate, iterations):
        self.rate = rate
        n_row = len(inputs)
        t0 = time.perf_counter()
        for i in range(iterations):
            r_index = random.randint(0, n_row-1)
            row = inputs[r_index]
            out = self.predict(row)
            expected = outputs[r_index]
            self.update(expected)
            print(f"{i}/{iterations} ITERATIONS")
            if i % 10000 == 0 and i != 0:
                with open(f"{self.name}.pickle", 'wb') as handle:
                    pickle.dump(self, handle)
                self.accuracy(inputs, outputs, 1000)
        print(f"DONE, TIME ELAPSED: {time.perf_counter() - t0}")

    def accuracy(self, inputs, outputs, sample_size):
        n_row = len(inputs)
        samples = random.sample(range(0, n_row), sample_size)
        t0 = time.perf_counter()
        errors = np.zeros(len(outputs[0]))
        answers = []
        predictions = []
        for r in samples:
            row = inputs[r]
            answers.append(outputs[r].reshape(10))
            predictions.append(self.predict(row))
        correct = sum([np.argmax(a) == np.argmax(b) for a, b in zip(answers, predictions)])
        print("="*20)
        print(f"TEST DONE, TIME ELAPSED: {time.perf_counter() - t0}.")
        print("="*20)
        print(f"TESTED {len(samples)} SAMPLES.")
        print(f"CORRECT ANSWERS: {correct}/{len(samples)}")
        print(f"CORRECT RATIO: {(correct/len(samples))*100}%")
        print("="*20)

def main():
    with open("mnist/numbers.pickle", 'rb') as handle:
        name, weights = pickle.load(handle)

    with np.load("mnist/numbers.npz") as data:
        INPUTS = data['training_images']
        OUTPUTS = data['training_labels']

    nn = NeuralNetwork(name)
    nn.load(weights)

    plot = Plot(nn, INPUTS[0])
    while True:
        current = random.randint(0, len(INPUTS))
        nn.predict(INPUTS[current])
        plot.plotNetwork(INPUTS[current])

if __name__ == "__main__":
    main()
