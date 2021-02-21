

from matenn.nn import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.learning_rate * grad
