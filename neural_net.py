from random import random

from engine import Value


class Neuron:
    def __init__(self, n_in):
        self.weight = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.bais = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.weight, x)), self.bais)
        out = act.tanh()
        return out
    
class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    """
    Multi-Layer Perceptron
    """
    def __init__(self, n_in, n_out):
        sz = [n_in] + n_out
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_out))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
        
        