
from numpy import ndarray
import numpy as np


class Loss:
    def loss(self, prediction: ndarray, actual: ndarray) -> float:
        raise NotImplementedError

    def grad(self, prediction: ndarray, actual: ndarray) -> ndarray:
        raise NotImplementedError


class MSE(Loss):
    def loss(self, prediction: ndarray, actual: ndarray) -> float:
        return np.sum((prediction - actual) ** 2) / prediction.shape[0]

    def grad(self, prediction: ndarray, actual: ndarray) -> ndarray:
        return 2 * (prediction - actual) / prediction.shape[0]