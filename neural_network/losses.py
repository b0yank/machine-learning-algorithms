import numpy as np
import warnings
from abc import ABCMeta, abstractmethod

class Loss(metaclass = ABCMeta):
    """Abstract loss base class.
    """
    @abstractmethod
    def output_deriv(self, y, t):
        pass

    @abstractmethod
    def get_loss(self, true, predictions):
        pass
        
class MeanSquaredLoss(Loss):
    """Mean squared loss.
    """
    def output_deriv(self, y, t):
        return (y - t)

    def get_loss(self, true, predictions):
        N = len(true)
        return (1/N) * np.sum((true - predictions) ** 2)
        
class CrossEntropyLoss(Loss):
    """Cross-entropy loss.
    """
    def output_deriv(self, y, t):
        if t.shape[1] > 1:
            return -(t / y)

        return (y - t) / (y * (1 - y))

    def get_loss(self, true, predictions):
        N = len(true)

        if true.shape[1] > 1:
            return -(1/N) * np.sum(true * np.log(predictions))

        return -(1/N) * np.sum(true * np.log(predictions) + (1 - true) * np.log(1 - predictions))






