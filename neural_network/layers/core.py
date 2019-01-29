import numpy as np
from abc import ABCMeta, abstractmethod

from neural_network.utils import Mode

# ACTIVATION FUNCTION TYPES
LINEAR = 'linear'
SOFTMAX = 'softmax'
TANH = 'tanh'
RELU = 'relu'

def softmax(z):
    x = z.reshape(len(z), -1)
    sum_ = np.sum(np.exp(x), axis = 1).reshape(-1, 1)
    return (np.exp(x) / sum_).reshape(z.shape)

def softmax_deriv(z):
    s = softmax(z)
    orig_shape = s.shape
    s = s.reshape(len(s), -1)
    m, n = s.shape
    diags = np.zeros((m, n, n))
    diags[:, np.arange(n), np.arange(n)] = s
    dots = np.einsum('ij,ik->ijk', s, s)
    
    return diags - dots

ACTIVATION_FUNCTIONS = {
    LINEAR: lambda z: z,
    SOFTMAX: softmax,
    TANH: lambda z: np.tanh(z),
    RELU: lambda z: np.maximum(z, 0)
}

ACTIVATION_DERIVATIVES = {
    LINEAR: lambda z: np.ones(z.shape),
    SOFTMAX: softmax_deriv,
    TANH: lambda z: 1 - np.tanh(z) ** 2,
    RELU: lambda z: np.array(z > 0, dtype='int')
}

UNITS_NDIMS = 2

class Layer(metaclass = ABCMeta):
    """Abstract class acting as an interface for all layer types and providing some
       common initialization
    """
    def __init__(self,
                 activation,
                 use_bias,
                 ndims,
                 input_shape = None,
                 kernel_reg_l2 = 0.0):
        self.trainable = True
        self.input_shape = input_shape
        self._name = None
        self.activation = activation
        if activation == None:
            activation = LINEAR
        self._activation_func = ACTIVATION_FUNCTIONS[activation]
        self._activation_deriv = ACTIVATION_DERIVATIVES[activation]
        self.use_bias = use_bias
        self._ndims = ndims
        self._output_shape = self.output_shape(input_shape) if input_shape else None
        self.kernel_reg_l2 = kernel_reg_l2
        
    @property
    def weights(self): raise NotImplementedError
    @weights.setter
    def weights(self, weights): raise NotImplementedError
    @property
    def dW(self): raise NotImplementedError
    @property
    def bias(self): return self._bias
    @bias.setter
    def bias(self, bias): self._bias = bias
    @property
    def dB(self): return self._dB
    
    @abstractmethod
    def forward(self, prev_activations, mode = Mode.TRAIN):
        pass
    @abstractmethod
    def backward(self, prev_activations, delta):
        pass
    @abstractmethod
    def output_shape(self, input_shape):
        pass

class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.
    """
    def __init__(self):
        self.input_shape = None
        self._output_shape = None
        self.trainable = False
        
    def forward(self, prev_activations, mode = Mode.TRAIN):
        self.input_shape = prev_activations.shape
        self.activations = prev_activations.reshape(len(prev_activations), -1)
    
    def backward(self, prev_activations, delta):
        return delta.reshape(self.input_shape)
        
    def output_shape(self, input_shape):
        if input_shape is None:
            if self.input_shape is None:
                return None

            input_shape = self.input_shape

        self._output_shape = (input_shape[0], np.prod(input_shape[1:]))
        return self._output_shape

class BatchNormalization(Layer):
    """Batch normalization layer (Ioffe and Szegedy, 2014).

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    """
    def __init__(self,
                 axis = 1,
                 momentum = 0.99,
                 epsilon = 1e-3,
                 center = True,
                 scale = True,
                 beta_initializer = 'zeros',
                 gamma_initializer = 'ones',
                 beta_regularization = 0.0,
                 gamma_regularization = 0.0):
        self.axis = axis
        self.__axis = None
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = scale
        self.center = center
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.beta_regularization = beta_regularization
        self.gamma_regularization = gamma_regularization
        self.kernel_reg_l2 = np.array([gamma_regularization, beta_regularization]).reshape(-1, 1)
        self.use_bias = False
        self.__beta = None
        self.__gamma = None
        self.__dB = 0
        self.__dG = 0
        self.__mu_avg = None
        self.__sigma_avg = None
        self.__N = 0

        self.trainable = True if self.scale or self.center else False

    @property
    def weights(self): return np.array([self.__gamma, self.__beta])
    @weights.setter
    def weights(self, weights): self.__gamma = weights[0]; self.__beta = weights[1]
    @property
    def dW(self): return np.array([self.__dgamma, self.__dbeta])

    def forward(self, prev_activations, mode = Mode.TRAIN):
        ndims = len(prev_activations.shape)
        # correct for negative axis input, e.g. axis=-1
        self.__actual_axis = self.axis if self.axis >= 0 else ndims + self.axis

        if mode == Mode.TEST:
            sigma_denom = 1 / np.sqrt(self.__sigma_avg + self.epsilon)
            X_hat = (np.moveaxis(prev_activations, self.__actual_axis, -1) - self.__mu_avg) * sigma_denom
            self.activations = np.moveaxis(X_hat * self.__gamma + self.__beta, -1, self.__actual_axis)
            return

        self.__M = prev_activations.shape[self.__actual_axis]

        if self.__gamma is None:
            self.__initialize_params((self.__M,))

        if self.__axis is None:
            self.__axis = tuple([ax for ax in np.arange(ndims) if ax != self.__actual_axis])

        mu = np.mean(prev_activations, axis = self.__axis)
        sigma = np.var(prev_activations, axis = self.__axis)
        self.__mu_avg = mu if self.__mu_avg is None else self.__mu_avg * self.momentum + (1 - self.momentum) * mu
        self.__sigma_avg =  sigma if self.__sigma_avg is None else self.__sigma_avg * self.momentum + (1 - self.momentum) * sigma

        self.__sigma_denom = 1 / np.sqrt(sigma + self.epsilon)
        self.__X_hat = (np.moveaxis(prev_activations, self.__actual_axis, -1) - mu) * self.__sigma_denom
        self.activations = np.moveaxis(self.__X_hat * self.__gamma + self.__beta, -1, self.__actual_axis)
        self.__X_hat = np.moveaxis(self.__X_hat, -1, self.__actual_axis)
            
    def backward(self, prev_activations, delta):
        if self.trainable:
            self.__dbeta = np.sum(delta, axis = self.__axis) if self.center else 0
            self.__dgamma = np.sum(self.__X_hat * delta, axis = self.__axis) if self.scale else 0

        n_samples = len(delta)

        # basically doing delta * gamma along the channel axis
        sum_str = ''.join([chr(c) for c in range(ord('i'), ord('i') + self.__actual_axis + 1)])
        axis_chr = chr(ord('i') + self.__actual_axis)
        sum_str = sum_str + f'...,{axis_chr}->{sum_str}...'
        dX_hat = np.einsum(sum_str, delta, self.__gamma)

        denom = (1 / self.__M) * self.__sigma_denom
        dX = self.__M * dX_hat - np.sum(dX_hat, axis = 0) - self.__X_hat * np.sum(dX_hat * self.__X_hat, axis = 0)
        dX = np.einsum(sum_str, dX, denom)

        return dX

    def output_shape(self, input_shape):
        return input_shape

    def __initialize_params(self, shape):
        functions = {
            'ones': np.ones,
            1: np.ones,
            'zeros': np.zeros,
            0: np.zeros
        }

        self.__gamma = functions[self.gamma_initializer](shape) if self.scale else 1
        self.__beta = functions[self.beta_initializer](shape) if self.center else 0

class Activation(Layer):
    """ Applies an activation function to an output.
    """
    def __init__(self, activation):
        self.activation = activation
        self._activation_func = ACTIVATION_FUNCTIONS[activation]
        self._activation_deriv = ACTIVATION_DERIVATIVES[activation]
        self.use_bias = False
        self.trainable = False

    def forward(self, prev_activations, mode = Mode.TRAIN):
        self.activations = self._activation_func(prev_activations)
        return self.activations

    def backward(self, prev_activations, delta):
        return prev_activations * delta

    def output_shape(self, input_shape):
        return input_shape

class Dense(Layer):
    """Just your regular densely-connected NN layer.
    """
    def __init__(self,
                 units,
                 activation = None,
                 use_bias = True,
                 input_shape = None,
                 kernel_reg_l2 = 0.0):
        super().__init__(activation = activation,
                         use_bias = use_bias,
                         ndims = 2,
                         input_shape = input_shape,
                         kernel_reg_l2 = kernel_reg_l2)
        self.units = units
        self.activations = np.array([])
        self.__weights = None
        if use_bias:
            self._bias = np.zeros((1, units))
            
    @property
    def weights(self): return self.__weights
    @weights.setter
    def weights(self, weights): self.__weights = weights
    @property
    def dW(self): return self.__dW    
        
    def forward(self, prev_activations, mode = Mode.TRAIN):
        if self.weights is None:
            weight_shape = (self.units, prev_activations.shape[1])
            self.__weights = np.random.normal(size = weight_shape)
            
        bias = self._bias if self.use_bias else 0
        self.__z = prev_activations.dot(self.__weights.T) + bias
        self.activations = self._activation_func(self.__z)
 
    def backward(self, prev_activations, delta):
        m = len(delta)
        
        activ_deriv = self._activation_deriv(self.__z)
        # summing is required in case of activations whose derivatives output a Jacobian matrix (s.a. softmax)
        sum_str = 'i...j,ij->i...' if len(activ_deriv.shape) > UNITS_NDIMS else 'i...j,ij->i...j'
        self.__dZ = np.einsum(sum_str, activ_deriv, delta)

        self.__dW = (1 / m) * self.__dZ.T.dot(prev_activations)
        if self.use_bias:
            self._dB = (1 / m) * np.sum(self.__dZ, axis = 0).reshape(1, -1)
        
        # delta is basically dE/dA of the layer one level back (derivative of activations)
        delta_new = self.__dZ.dot(self.__weights)
        return delta_new
        
    def output_shape(self, input_shape):
        if self._output_shape is None:
            if input_shape is None:
                return None
            self._output_shape = (input_shape[0], self.units)
            
        return  self._output_shape



