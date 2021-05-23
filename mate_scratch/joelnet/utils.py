
from numpy import ndarray
import numpy as np

def to_2d_np(a: ndarray, type: str="col") -> ndarray:
    """
    Turns a 1D Tensor into 2D
    """

    assert a.ndim == 1, \
        "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)

def permute_data(X: ndarray, y: ndarray) -> (ndarray, ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def sigmoid(x: ndarray) -> ndarray:
    return 1.0/(1.0 + np.exp(-1.0 * x))


def sigmoid_prime(x: ndarray) -> ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: ndarray) -> ndarray:
    return np.tanh(x)


def tanh_prime(x: ndarray) -> ndarray:
    return 1 - tanh(x)**2


def relu(x: ndarray) -> ndarray:
    return np.maximum(0, x)


def relu_prime(x: ndarray) -> ndarray:
    x[x<=0] = 0
    x[x>0] = 1
    return x    



    