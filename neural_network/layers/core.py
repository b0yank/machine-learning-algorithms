import numpy as np
from abc import ABCMeta, abstractmethod

from .. import activations, initializers
from neural_network.utils import PADDING_SAME, PADDING_VALID, PADDING_TYPES, TIMESTEP_AXIS
from utils.graph import Node

class Layer(Node, metaclass = ABCMeta):
    """Abstract class acting as an interface for all layer types and providing some
       common initialization

       Warnings: 1) If inheriting classes do not call Layer.__init__(), they should at least call the Node.__init__() method.
                 2) All layers are assumed to be trainable. Layers that are not intended to be trained should explicitly set self.trainable = False
    """
    def __init__(self,
                 activation = None,
                 use_bias = False,
                 input_shape = None,
                 kernel_reg_l2 = 0.0):
        super().__init__()
        self.trainable = True
        self.input_shape = input_shape
        self._name = None

        self.activation = activation
        if activation == None:
            activation = activations.LINEAR
        self._activation = activations.get(activation)
        self.use_bias = use_bias
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

    def _compute_forward_message(self, inputs):
        self.forward(**inputs)
        return self.activations

    def _compute_backward_message(self, inputs):
        delta = self.backward(**inputs)
        return delta
    
    @abstractmethod
    def forward(self, prev_activations, train_mode = True, *args, **kwargs): pass
    @abstractmethod
    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs): pass
    @abstractmethod
    def output_shape(self, input_shape): pass

    def _add_weight(self, shape, initializer):
        initializer_func = initializers.get(initializer)
        return initializer_func(shape)

class Layer2D(Layer):
    def __init__(self,
                 kernel_size,
                 strides,
                 padding,
                 activation = None,
                 use_bias = False,
                 input_shape = None,
                 kernel_reg_l2 = 0.0):
        super().__init__(activation = activation,
                         use_bias = use_bias,
                         input_shape = input_shape,
                         kernel_reg_l2 = kernel_reg_l2)
        self.strides = strides
        self._strides = self._check_tuple(strides, 'strides')

        self.kernel_size = kernel_size
        self._kernel_size = self._check_tuple(kernel_size, 'kernel_size')
       
        if padding not in PADDING_TYPES:
            raise ValueError(f'Padding can be one of \'{PADDING_VALID}\' and \'{PADDING_SAME}\'')
        self.padding = padding

    def output_shape(self, input_shape):
        if not self._output_shape:
            if not input_shape:
                return None

            image_shape = input_shape[-2:]
            for img_shape, kernel_shape in zip(image_shape, self._kernel_size):
                if img_shape < kernel_shape:
                    raise InvalidShapeError(f'Negative dimension size caused by subtracting {kernel_shape} '
                                            f'from {img_shape} for layer {self._name}.')

            conv_shape = self._conv_steps(input_shape, self._kernel_size, self._strides)
            self._output_shape = list(input_shape)
            self._output_shape[-2:] = conv_shape
            self._output_shape = tuple(self._output_shape)
        return self._output_shape

    def _add_padding(self, activations, full = False):
        n_images = activations.shape[0]
        channel_shape = np.array(activations.shape[-2:])
        padding = self._kernel_size - 1
        padding_steps = np.array([1, 1])
        if full:
            # padding doubles on a full convolution
            padding = padding * 2

            ## in case of full convolution, activations need to be spaced out with padding = strides between elements
            channel_shape = (channel_shape - 1) * self._strides + 1
            padding_steps = self._strides

        padding_up, padding_down, padding_left, padding_right = self._calculate_padding(padding)

        padded = np.zeros(activations.shape[:-2] + (channel_shape[0] + padding[0], channel_shape[1] + padding[1]))
        padded_height, padded_width = padded.shape[-2:]

        padded[...,
               padding_up: padded_height - padding_down: padding_steps[1],
               padding_left: padded_width - padding_right: padding_steps[0]] = activations

        return padded

    def _calculate_padding(self, padding):
        pad_left = np.ceil(padding[0] / 2).astype('int16')
        pad_right = padding[0] - pad_left
        pad_up = np.ceil(padding[1] / 2).astype('int16')
        pad_down = padding[1] - pad_up

        return pad_up, pad_down, pad_left, pad_right

    def _clip_padding(self, inputs):
        padding = self._kernel_size - 1
        padding_up, padding_down, padding_left, padding_right = self._calculate_padding(padding)
        padded_height, padded_width = np.array(self.input_shape[-2:]) + padding
        return inputs[...,
                    padding_up: padded_height - padding_down,
                    padding_left: padded_width - padding_right]

    def _conv_steps(self, activations_shape, kernel_size, strides):
        x_steps = np.floor_divide(activations_shape[-1] - kernel_size[1], strides[1]) + 1
        y_steps = np.floor_divide(activations_shape[-2] - kernel_size[0], strides[0]) + 1
        
        return (x_steps, y_steps)
        
    def _check_tuple(self, tuple_, argument_name):
        NONPOSITIVE_ERROR = f'\'{argument_name}\' must be a positive integer, or a tuple of positive integers.'
        TYPE_ERROR = f'The `{argument_name}` argument must be a tuple of 2 integers. Received: {tuple_}'

        if isinstance(tuple_, int):
            if tuple_ <= 0:
                raise ValueError(NONPOSITIVE_ERROR)
            return np.array([tuple_, tuple_]).astype('int16')
        
        elif isinstance(tuple_, tuple):
            if not isinstance(tuple_[0], int) or not isinstance(tuple_[1], int):
                raise ValueError(TYPE_ERROR + f'including element {tuple_[0]} of type {type(tuple_[0])}')
            if tuple_[0] <= 0 or tuple_[0] <= 0:
                raise ValueError(NONPOSITIVE_ERROR)
            
            return np.array([tuple_[0], tuple_[1]]).astype('int16')
        else:
            raise TypeError(TYPE_ERROR)

class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.
    """
    def __init__(self):
        super(Layer, self).__init__()
        self.input_shape = None
        self._output_shape = None
        self.trainable = False
        
    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        self.input_shape = prev_activations.shape
        self.activations = prev_activations.reshape(len(prev_activations), -1)
    
    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
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
        super().__init__()
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
        self.__mu_avg_hat = None
        self.__sigma_avg_hat = None
        self.__t = 0

        self.trainable = True if self.scale or self.center else False

    @property
    def weights(self): return np.array([self.__gamma, self.__beta])
    @weights.setter
    def weights(self, weights): self.__gamma = weights[0]; self.__beta = weights[1]
    @property
    def dW(self): return np.array([self.__dgamma, self.__dbeta])

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        ndims = len(prev_activations.shape)
        # correct for negative axis input, e.g. axis=-1
        self.__actual_axis = self.axis if self.axis >= 0 else ndims + self.axis

        if not train_mode:
            #remove bias of exponentially weighted averages
            mu_avg_hat = self.__mu_avg / (1 - self.momentum ** self.__t)
            sigma_avg_hat = self.__sigma_avg / (1 - self.momentum ** self.__t)

            sigma_denom = 1 / np.sqrt(sigma_avg_hat + self.epsilon)
            X_hat = (np.moveaxis(prev_activations, self.__actual_axis, -1) - mu_avg_hat) * sigma_denom
            self.activations = np.moveaxis(X_hat * self.__gamma + self.__beta, -1, self.__actual_axis)
            return

        self.__M = prev_activations.shape[self.__actual_axis]

        if self.__gamma is None:
            self.__initialize_params((self.__M,))

        if self.__axis is None:
            self.__axis = tuple([ax for ax in np.arange(ndims) if ax != self.__actual_axis])

        batch_size = len(prev_activations)

        mu = np.mean(prev_activations, axis = self.__axis)
        sigma = np.var(prev_activations, axis = self.__axis) * (batch_size / (batch_size - 1))

        if self.__mu_avg is None:
            self.__mu_avg = np.zeros(mu.shape)
            self.__sigma_avg = np.zeros(sigma.shape)

        self.__mu_avg = self.__mu_avg * self.momentum + (1 - self.momentum) * mu
        self.__sigma_avg = self.__sigma_avg * self.momentum + (1 - self.momentum) * sigma
        self.__t += 1

        self.__sigma_denom = 1 / np.sqrt(sigma + self.epsilon)
        self.__X_hat = (np.moveaxis(prev_activations, self.__actual_axis, -1) - mu) * self.__sigma_denom
        self.activations = np.moveaxis(self.__X_hat * self.__gamma + self.__beta, -1, self.__actual_axis)
        self.__X_hat = np.moveaxis(self.__X_hat, -1, self.__actual_axis)
            
    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
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

class LayerNormalization(Layer):
    def __init__(self):
        super().__init__(use_bias=True)

        self.__gain = None
        self._bias = None
        self.__epsilon = 1e-6

    @property
    def weights(self): return self.__gain
    @weights.setter
    def weights(self, weights):
        self.__gain = weights
        
    @property
    def dW(self): return self.__dg

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        if self.__gain is None or self._bias is None:
            self.__gain = np.ones(shape=(1, prev_activations.shape[-1]))
            self._bias = np.zeros(shape=(1, prev_activations.shape[-1]))

        miu = np.sum(prev_activations, axis=0) / len(prev_activations)
        sigma = np.sum((prev_activations - miu) ** 2) / len(prev_activations)
        self.__denom = np.sqrt(sigma + self.__epsilon)
        norm = (prev_activations - miu) / self.__denom
       
        act = norm * self.__gain + self._bias
        self.activations = act

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        self._dB = np.mean(delta, axis=tuple([ax for ax in range(len(delta.shape) - 1)])).reshape(1, -1)
        self.__dg = np.mean(delta * self.activations, axis=tuple([ax for ax in range(len(delta.shape) - 1)])).reshape(1, -1)

        return delta * self.__gain / self.__denom

    def output_shape(self, input_shape): return input_shape

class Activation(Layer):
    """ Applies an activation function to an output.
    """
    def __init__(self, activation):
        super(Layer, self).__init__()

        self.activation = activation
        self._activation = activations.get(activation)
        self.use_bias = False
        self.trainable = False

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        self.activations = self._activation.get_activation(prev_activations)
        return self.activations

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        return self._activation.get_delta(prev_activations, delta)

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
        self.units = units
        self._weights = None
        if use_bias:
            self._bias = np.zeros((1, units))

        super().__init__(activation = activation,
                         use_bias = use_bias,
                         input_shape = input_shape,
                         kernel_reg_l2 = kernel_reg_l2)

        if input_shape is not None:
            self.weights = np.random.normal(size = (self.units, input_shape[1]))
            
    @property
    def weights(self): return self._weights
    @weights.setter
    def weights(self, weights): self._weights = weights
    @property
    def dW(self): return self.__dW 
        
    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        if self.weights is None:
            weight_shape = (self.units, prev_activations.shape[-1])
            self._weights = np.random.normal(size = weight_shape)
            
        bias = self._bias if self.use_bias else 0
        self.__z = prev_activations.dot(self.weights.T) + bias
        self.activations = self._activation.get_activation(self.__z)
 
    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        m = len(delta)
        
        dZ = self._activation.get_delta(self.__z, delta)

        if len(delta.shape) == 3:
            self.__dW = np.einsum('ijk,ijl->kl', dZ, prev_activations)
        else:
            self.__dW = dZ.T.dot(prev_activations)

        if self.use_bias:
            axes = (0, 1) if len(dZ.shape) == 3 else 0
            self._dB = (1 / m) * np.sum(dZ, axis = axes).reshape(1, -1)
        
        # delta is basically dE/dA of the layer one level back (derivative of activations)
        delta_new = dZ.dot(self.weights)
        return delta_new
        
    def output_shape(self, input_shape):
        return (self.units, input_shape[1])

class Dropout(Layer):
    def __init__(self, rate,
                 noise_shape = None,
                 seed = None):
        super(Layer, self).__init__()

        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.trainable = False
        self.__random_state = np.random.RandomState(seed = seed)

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        if self.seed is not None:
            np.random.RandomState()
        
        if self.rate < 0 or self.rate >= 1:
            raise ValueError('Dropout range is [0, 1).')

        if not train_mode:
            self.activations = prev_activations

        mask_shape = prev_activations.shape if self.noise_shape is None else noise_shape
        self.__mask = self.__random_state.uniform(size=mask_shape) >= self.rate

        self.activations = prev_activations * (self.__mask / (1 - self.rate))

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        if train_mode:
            delta *= (self.__mask / (1 - self.rate))
        return delta

    def output_shape(self, input_shape): return input_shape

class Masking(Layer):
    def __init__(self, mask_value = 0.):
        super(Layer, self).__init__()

        self.mask_value = mask_value
        self.trainable = False

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        activs_time_major = prev_activations.swapaxes(TIMESTEP_AXIS, 0)

        self.__unmasked_idxs = [np.any(activs_time_major[i] != self.mask_value) for i in range(activs_time_major.shape[0])]
        self.activations = activs_time_major[self.__unmasked_idxs].swapaxes(TIMESTEP_AXIS, 0)

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        deltas = np.ones(prev_activations.shape) * self.mask_value
        deltas.swapaxes(TIMESTEP_AXIS, 0)[self.__unmasked_idxs] = delta.swapaxes(TIMESTEP_AXIS, 0)

        return deltas

    def output_shape(self, input_shape):
        return input_shape[:TIMESTEP_AXIS] + (None,) + input_shape[TIMESTEP_AXIS + 1:]







