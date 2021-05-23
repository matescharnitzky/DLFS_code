
import numpy as np
from numpy import ndarray

from copy import deepcopy

from joelnet.utils import permute_data
from joelnet.data import DataIterator
from joelnet.nn import NeuralNet
from joelnet.loss import Loss
from joelnet.optim import Optimizer


class Trainer:
    def __init__(self,
                 iterator: DataIterator,
                 net: NeuralNet,
                 loss: Loss,
                 optimizer: Optimizer) -> None:

        self.iterator = iterator
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.best_loss = 1e9

    def fit(self,
            X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int,
            eval_every: int,
            seed: int = 1) -> None:

        # permute
        np.random.seed(seed)
        X_train, y_train = permute_data(X_train, y_train)

        # fit
        for epoch in range(epochs):
            loss_train = 0.0
            
            if (epoch + 1) % eval_every == 0:
                last_model = deepcopy(self.net)
            
            for batch in self.iterator(X_train, y_train):
                # 1. forward pass from layer to layer to get the predictions
                pred_train = self.net.forward(batch.inputs)
                # 2. training loss
                loss_train += self.loss.loss(pred_train, batch.targets)
                # 3. gradient of the loss function
                grad = self.loss.grad(pred_train, batch.targets)
                # 4. backward pass from layer to layer
                self.net.backward(grad)
                # 5. update parameters
                self.optimizer.step(self.net)
            # 6. validation
            if (epoch + 1) % eval_every == 0:
                pred_test = self.net.forward(X_test)
                loss_test = self.loss.loss(pred_test, y_test)
                
                if loss_test < self.best_loss:
                    print(f"Epoch: {epoch + 1} |  Train loss: {loss_train:.3f} | Validation loss: {loss_test:.3f}")
                    self.best_loss = loss_test
                else:
                    print(f"Validaton loss increased after epoch: {epoch + 1}")
                    self.net = last_model
                    break
