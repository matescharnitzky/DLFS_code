
from numpy import ndarray
import numpy as np

from matenn.nn import NeuralNet


def assert_same_shape(array: ndarray,
                      array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        """
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        """.format(tuple(array_grad.shape), tuple(array.shape))
    return None


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


def mae(y_true: ndarray, y_pred: ndarray):
    """
    Compute mean absolute error for a neural network.
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: ndarray, y_pred: ndarray):
    """
    Compute root mean squared error for a neural network.
    """
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def eval_regression_model(model: NeuralNet,
                          X_test: ndarray,
                          y_test: ndarray):
    """
    Compute mae and rmse for a neural network.
    """

    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print()
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))


def permute_data(X: ndarray, y: ndarray) -> (ndarray, ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]