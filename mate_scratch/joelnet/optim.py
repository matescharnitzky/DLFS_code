
import numpy as np

from joelnet.nn import NeuralNet


class Optimizer:
    def __init__(self,
                 learning_rate: float = 0.01,
                 final_learning_rate: float = 0,
                 decay_type: str = None) -> None:
        self.learning_rate = learning_rate
        self.final_learning_rate = final_learning_rate
        self.decay_type = decay_type
    
    def setup_decay(self) -> None:

        if not self.decay_type:
            return
        
        elif self.decay_type == 'exponential':
            self.decay_per_epoch = np.power(self.final_learning_rate / self.learning_rate, 1.0 / (self.max_epochs - 1))
        
        elif self.decay_type == 'linear':
            self.decay_per_epoch = (self.learning_rate - self.final_learning_rate) / (self.max_epochs - 1)

    def decay_learning_rate(self) -> None:

        if not self.decay_type:
            return
        
        if self.decay_type == 'exponential':
            self.learning_rate *= self.decay_per_epoch
        
        elif self.decay_type == 'linear':
            self.learning_rate -= self.decay_per_epoch
    
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError
        
class SGD(Optimizer):
    def __init__(self, 
                 learning_rate: float = 0.01,
                 final_learning_rate: float = 0,
                 decay_type: str = None) -> None:
        super().__init__(learning_rate, final_learning_rate, decay_type)

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.learning_rate * grad
            
class SGDMomentum(Optimizer):
    def __init__(self, 
                 learning_rate: float = 0.01,
                 final_learning_rate: float = 0,
                 decay_type: str = None,
                 momentum: float = 0.9):
        super().__init__(learning_rate, final_learning_rate, decay_type)
        self.first = True
        self.momentum = momentum
        
    def step(self, net: NeuralNet) -> None:
        
        if self.first:
            self.velocities = [np.zeros_like(param) for param in net.params()]
            self.first = False
        
        for (velocity, param, grad) in zip(self.velocities, net.params(), net.grads()):
            velocity *= self.momentum
            velocity += self.learning_rate * grad
            param -= velocity