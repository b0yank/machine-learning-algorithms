import numpy as np
from scipy import sparse

from neural_network.utils import TIMESTEP_AXIS

# ACTIVATION FUNCTION TYPES
LINEAR = 'linear'
SOFTMAX = 'softmax'
SIGMOID = 'sigmoid'
HARD_SIGMOID = 'hard_sigmoid'
TANH = 'tanh'
RELU = 'relu'

# used in Transformer when you have both a batch_size and a sequence_length dimension
SEQUENCE_SOFTMAX = 'sequence_softmax'

def _softmax(z):
    x = z.reshape(len(z), -1)

    x_exp = np.exp(x)
    sum_ = np.sum(x_exp, axis = 1).reshape(-1, 1)

    softmax = (x_exp / sum_).reshape(z.shape)

    return softmax

def _sequence_softmax(z):
    # moving axes to accomodate for different possible values of TIMESTEP_AXIS
    z_reshaped = np.moveaxis(z, TIMESTEP_AXIS, 0)
    return np.moveaxis(np.array([_softmax(z_reshaped[idx]) for idx in range(z.shape[TIMESTEP_AXIS])]), 0, TIMESTEP_AXIS)

def _softmax_deriv(z):
    s = _softmax(z)

    orig_shape = s.shape

    s = s.reshape(len(s), -1)
    m, n = s.shape

    diags = np.zeros((m, n, n))
    diags[:, np.arange(n), np.arange(n)] = s
    dots = np.einsum('ij,ik->ijk', s, s)

    derivs = diags - dots
    return derivs

def _sequence_softmax_deriv(z):
    # moving axes to accomodate for different possible values of TIMESTEP_AXIS
    z_reshaped = np.moveaxis(z, TIMESTEP_AXIS, 0)
    return np.moveaxis(np.array([_softmax_deriv(z_reshaped[idx]) for idx in range(z.shape[TIMESTEP_AXIS])]), 0, TIMESTEP_AXIS)

def _hard_sigmoid(z):
    mid_mask = (z >= -2.5)&(z <= 2.5)
    large_mask = z > 2.5

    return (z * mid_mask * 0.2) + large_mask + mid_mask * 0.5

def _hard_sigmoid_deriv(z):
    mid_mask = (z >= -2.5)&(z <= 2.5)
    return mid_mask * 0.2

_ACTIVATION_FUNCTIONS = {
    LINEAR: lambda z: z,
    SOFTMAX: _softmax,
    SEQUENCE_SOFTMAX: _sequence_softmax,
    SIGMOID: lambda z: 1 / (1 + np.exp(-z)),
    HARD_SIGMOID: _hard_sigmoid,
    TANH: lambda z: np.tanh(z),
    RELU: lambda z: np.maximum(z, 0)
}

_ACTIVATION_DERIVATIVES = {
    LINEAR: lambda z: np.ones(z.shape),
    SOFTMAX: _softmax_deriv,
    SEQUENCE_SOFTMAX: _sequence_softmax_deriv,
    SIGMOID: lambda z: _ACTIVATION_FUNCTIONS[SIGMOID](z) * (1 - _ACTIVATION_FUNCTIONS[SIGMOID](z)),
    HARD_SIGMOID: _hard_sigmoid_deriv,
    TANH: lambda z: 1 - np.tanh(z) ** 2,
    RELU: lambda z: np.array(z > 0, dtype='int')
}

def get(identifier):
    return Activation(identifier)

class Activation:
    def __init__(self, identifier):
        if identifier is None:
            identifier = LINEAR

        self.identifier = identifier
        self.__activ_func = _ACTIVATION_FUNCTIONS[identifier]
        self.__deriv_func = _ACTIVATION_DERIVATIVES[identifier]


    def get_activation(self, z):
        """
        Returns unit activation.

        Parameters:
        z - unit before activation function is applied
        """
        return self.__activ_func(z)

    def get_derivative(self, z):
        """
        Returns derivative of activation function.

        Parameters:
        z - unit before activation function is applied
        Notes:
        For derivative of an activation function of a layer, please use get_delta(identifier, z, delta)
        and get the derivative from the error (dE/dz) directly.
        """
        return self.__deriv_func(z)

    def get_delta(self, z, delta):
        """
        Returns derivative of error with respect to unit z

        Parameters:
        z - unit before activation function is applied
        delta - error derivative up to activation function
        """
        deriv = self.get_derivative(z)

        if self.identifier == SOFTMAX:
            # summing is required because softmax derivative outputs a Jacobian matrix
            out = np.einsum('i...j,ij->i...', deriv, delta)
            return out
        elif self.identifier == SEQUENCE_SOFTMAX:
            # moving axes to accomodate for different possible values of TIMESTEP_AXIS
            deriv_reshaped = np.moveaxis(deriv, TIMESTEP_AXIS, 0)
            delta_reshaped = np.moveaxis(delta, TIMESTEP_AXIS, 0)
            out = np.moveaxis(np.array([np.einsum('i...j,ij->i...', deriv_reshaped[idx], delta_reshaped[idx]) for idx in range(deriv.shape[TIMESTEP_AXIS])]), 0, TIMESTEP_AXIS)
            return out

        return deriv * delta


