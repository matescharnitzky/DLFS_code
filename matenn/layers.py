
import numpy as np

from typing import Dict, Callable

from matenn.utils import *


class Layer:
    """
    Base class for layers. Each layer should have a forward a backward pass.
    """

    def __init__(self) -> None:
        self.params: Dict[str, ndarray] = {}
        self.grads: Dict[str, ndarray] = {}

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Produce the forward pass in an NN
        """
        raise NotImplementedError

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Produce the backward pass in an NN
        """
        raise NotImplementedError


class Linear(Layer):
    """
    A linear layer takes the linear combination of inputs and weights.
    """

    def __init__(self, input_size: ndarray, output_size: ndarray, seed: int) -> None:
        super().__init__()
        np.random.seed(seed)
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(1, output_size)

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Example NN with 1 hidden layer:
        Z1 = X @ W1 + b1
        A1 = sigma(Z1)
        P = A1 @ W2 + b2
        L = np.mean((P - Y)**2)
        """

        self.inputs = inputs
        output = inputs @ self.params["w"] + self.params["b"]
        return output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Example NN with 2 hidden layer:
        output_grad: dLdP = 2 * (P - Y)
        param_grad: dLdW2 = A1.T @ dLdP
        param_grad: dLdB2 = np.sum(dLdP, axis=0)
        input_grad: dLdA1 = dLdP @ W2.T

        dA1dZ1 = sigma(Z1) * (1 - sigma(Z1))

        output_grad: dPdZ1 = dLdP @ W2.T * sigma(Z1) * (1 - sigma(Z1))
        param_grad: dPdW1 = X.T @ dPdZ1
        param_grad: dPdB1 = np.sum(dPdZ1, axis=0)
        input_grad: dPdX = dPdZ1 @ W1.T
        """

        self.grads["w"] = self.inputs.T @ output_grad
        self.grads["b"] = np.sum(output_grad, axis=0)
        input_grad = output_grad @ self.params["w"].T

        return input_grad


class Activation(Layer):
    """
    An activation layer applies a function on the outputs.
    """

    def __init__(self, f: Callable[[ndarray], ndarray], f_prime: Callable[[ndarray], ndarray]) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: ndarray) -> ndarray:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, output_grad: ndarray) -> ndarray:
        return output_grad * self.f_prime(self.inputs)


def sigmoid(x: ndarray) -> ndarray:
    return 1.0/(1.0 + np.exp(-1.0 * x))


def sigmoid_prime(x: ndarray) -> ndarray:
    s = sigmoid(x)
    return s * (1 - s)


class Sigmoid(Activation):
    """
    A sigmoid activation layer
    """

    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)