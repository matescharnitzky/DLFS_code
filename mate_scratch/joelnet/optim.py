
import numpy as np

from joelnet.nn import NeuralNet


class Optimizer:    
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError
        
class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.learning_rate * grad
            
class SGDMomentum(Optimizer):
    def __init__(self, 
                 learning_rate: float = 0.01,
                 momentum: float = 0.9):
        self.first = True
        self.learning_rate = learning_rate 
        self.momentum = momentum
        
    def step(self, net: NeuralNet) -> None:
        
        if self.first:
            self.velocities = [np.zeros_like(param) for param in net.params()]
            self.first = False
        
        for (velocity, param, grad) in zip(self.velocities, net.params(), net.grads()):
            velocity *= self.momentum
            velocity += self.learning_rate * grad
            param -= velocity