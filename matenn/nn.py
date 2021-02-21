

from typing import List, Iterator, Tuple

from matenn.layers import Layer

from numpy import ndarray


class NeuralNet:
    """
    A Neural Net is a collection of layers.
    """

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: ndarray) -> ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, output_grad: ndarray) -> ndarray:
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
        return output_grad

    def params_and_grads(self) -> Iterator[Tuple[ndarray, ndarray]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
