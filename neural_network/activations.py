import numpy as np

# ACTIVATION FUNCTION TYPES
LINEAR = 'linear'
SOFTMAX = 'softmax'
SIGMOID = 'sigmoid'
HARD_SIGMOID = 'hard_sigmoid'
TANH = 'tanh'
RELU = 'relu'

def _softmax(z):
    x = z.reshape(len(z), -1)
    x_exp = np.exp(x)
    sum_ = np.sum(x_exp, axis = 1).reshape(-1, 1)
    return (x_exp / sum_).reshape(z.shape)

def _softmax_deriv(z):
    s = _softmax(z)
    orig_shape = s.shape
    s = s.reshape(len(s), -1)
    m, n = s.shape
    diags = np.zeros((m, n, n))
    diags[:, np.arange(n), np.arange(n)] = s
    dots = np.einsum('ij,ik->ijk', s, s)

    return diags - dots

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
    SIGMOID: lambda z: 1 / (1 + np.exp(-z)),
    HARD_SIGMOID: _hard_sigmoid,
    TANH: lambda z: np.tanh(z),
    RELU: lambda z: np.maximum(z, 0)
}

_ACTIVATION_DERIVATIVES = {
    LINEAR: lambda z: np.ones(z.shape),
    SOFTMAX: _softmax_deriv,
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

        self.__identifier = identifier
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

        if self.__identifier == SOFTMAX:
            # summing is required because softmax derivative outputs a Jacobian matrix
            return np.einsum('i...j,ij->i...', deriv, delta)

        return deriv * delta
