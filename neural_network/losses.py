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
        n = len(true)
        return (1 / n) * np.sum((true - predictions) ** 2)
        
class CrossEntropyLoss(Loss):
    """Cross-entropy loss.
    """
    def output_deriv(self, y, t):
        return -(t / y)

    def get_loss(self, true, predictions):
        n = len(true)
        return -(1 / n) * np.sum(true * np.log(predictions))






