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

       Args:
            label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
              meaning the confidence on label values are relaxed. e.g.
              `label_smoothing=0.2` means that we will use a value of `0.1` for label
              `0` and `0.9` for label `1`" (for binary classification)
    """
    def __init__(self, label_smoothing = 0):
        self.label_smoothing = label_smoothing

    def output_deriv(self, y, t):
        if t.shape[-1] > 1:
            deriv = -(t / y)
        else:
            deriv = (y - t) / (y * (1 - y))

        return deriv * (1 - self.label_smoothing)

    def get_loss(self, true, predictions):
        # true is expected to be a matrix of onehot encoded vectors
        N, K = true.shape

        predictions_smoothed = (1 - self.label_smoothing) * predictions + self.label_smoothing / K

        if true.shape[-1] > 1:
            return -(1/N) * np.sum(true * np.log(predictions_smoothed))

        return -(1/N) * np.sum(true * np.log(predictions_smoothed) + (1 - true) * np.log(1 - predictions_smoothed))







