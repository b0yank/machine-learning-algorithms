import numpy as np
from abc import ABCMeta, abstractmethod

class Optimizer(metaclass = ABCMeta):
    """Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.
    """
    def __init__(self, lr, decay):
        self.lr = lr
        self._lr_initial = lr
        self.decay = decay
        self._epoch = 1
        
    def decay_lr(self):
        self.lr = self._lr_initial / (1 + self.decay * self._epoch)
        self._epoch += 1
      
    @abstractmethod
    def update_weights(self, layer):
        pass
        
class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    """
    def __init__(self, lr = 0.01, decay = 0.0, momentum = 0.0):
        super().__init__(lr, decay)
        self.momentum = momentum
        self.__past_updates = dict()
        self.__past_bias = dict()
        
    def update_weights(self, layer):
        key = layer._name
        gradients = layer.dW
        past_updates = self.__past_updates[key] if key in self.__past_updates else 0

        V_dW = self.momentum * past_updates + (1 - self.momentum) * gradients
        self.__past_updates[key] = gradients
        layer.weights = layer.weights - self.lr * V_dW - layer.kernel_reg_l2 * layer.weights

        if layer.use_bias:
            past_bias = self.__past_bias[key] if key in self.__past_bias else 0

            V_dB = self.momentum * past_bias + (1 - self.momentum) * layer.dB
            self.__past_bias[key] = V_dB
            layer.bias = layer.bias - self.lr * V_dB
        
class Adam(Optimizer):
    """Adam optimizer.

    Default parameters follow those provided in the original paper.
    """
    def __init__(self, lr = 0.001, decay = 0.0, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        super().__init__(lr, decay)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        # dictionaries to keep past values of the m and b variables (m, v for weights and m_b, v_b for bias')
        self.__m, self.__v, self.__m_bias, self.__v_bias = dict(), dict(), dict(), dict()
        # dictionary which will hold the timestep terms of each layer for bias correction of 'm' and 'v'
        self.__t = dict()
        
    def update_weights(self, layer):
        key = layer._name
        self.__t[key] = self.__t[key] + 1 if key in self.__t else 1

        gradients = layer.dW
        m = self.__m[key] if key in self.__m else 0
        v = self.__v[key] if key in self.__v else 0

        self.__m[key] = self.beta_1 * m + (1 - self.beta_1) * gradients
        self.__v[key] = self.beta_2 * v + (1 - self.beta_2) * (gradients ** 2)
        m_hat = self.__m[key] / (1 - self.beta_1 ** self.__t[key])
        v_hat = self.__v[key] / (1 - self.beta_2 ** self.__t[key])

        layer.weights = layer.weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon) - layer.kernel_reg_l2 * layer.weights

        if layer.use_bias:
            m_bias = self.__m_bias[key] if key in self.__m_bias else 0
            v_bias = self.__v_bias[key] if key in self.__v_bias else 0

            self.__m_bias[key] = self.beta_1 * m_bias + (1 - self.beta_1) * layer.dB
            self.__v_bias[key] = self.beta_2  * v_bias + (1 - self.beta_2) * (layer.dB ** 2)
            m_bias_hat = self.__m_bias[key] / (1 - self.beta_1 ** self.__t[key])
            v_bias_hat = self.__v_bias[key] / (1 - self.beta_2 ** self.__t[key])

            layer.bias = layer.bias - self.lr * m_bias_hat / (np.sqrt(v_bias_hat) + self.epsilon)


