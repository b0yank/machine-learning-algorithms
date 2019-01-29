import numpy as np
from skimage.util.shape import view_as_windows

from .core import Layer
from neural_network.utils import InvalidShapeError, Mode

PADDING_SAME = 'same'
PADDING_VALID = 'valid'
PADDING_TYPES = {PADDING_SAME, PADDING_VALID}
CHANNEL_AXIS = 1
CONV2D_NDIMS = 4

class Conv2D(Layer):
    """2D convolution layer (e.g. spatial convolution over images).
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides = (1, 1),
                 padding = 'valid',
                 activation = None,
                 use_bias = True,
                 input_shape = None,
                 kernel_reg_l2 = 0.0):
        super().__init__(activation = activation,
                         use_bias = use_bias,
                         ndims = 4,
                         input_shape = input_shape,
                         kernel_reg_l2 = kernel_reg_l2)
        if use_bias:
            shape = tuple([1 if index != CHANNEL_AXIS else filters for index in range(CONV2D_NDIMS)])
            self._bias = np.zeros(shape)
    
        self.filters = filters
        self.__filters = None
        self.kernel_size = kernel_size
        self.__kernel_size = self.__check_tuple(kernel_size, 'kernel_size')
        self.strides = strides
        self.__strides = self.__check_tuple(strides, 'strides')
       
        if padding not in PADDING_TYPES:
            raise ValueError(f'Padding can be one of \'{PADDING_VALID}\' and \'{PADDING_SAME}\'')
        self.padding = padding
        
    def forward(self, prev_activations, mode):
        self.input_shape = prev_activations.shape
        if self.padding == PADDING_SAME:
            prev_activations = self.__add_padding(prev_activations)

        if self.__filters is None:
            self.__filters = np.random.normal(size = ((self.filters, prev_activations.shape[CHANNEL_AXIS])) + tuple(self.__kernel_size))
        
        self.__convolutions = self.__convolve(prev_activations)
        self.__X = prev_activations
        bias = self._bias if self.use_bias else 0
        self.activations = self._activation_func(self.__convolutions + bias)
    
    def backward(self, prev_activations, delta):
        m = len(delta)
        self.__dO = self._activation_deriv(self.__convolutions) * delta
        self.__dX = self.__convolve(self.__dO, self.__filters, strides = (1, 1), full = True)
        
        # filters need to have padding = strides added between elements of output derivative (dO)
        # they will act as the kernel in the derivative convolution done to find dF
        moduli = (np.array(self.__X.shape[-2:]) - self.__kernel_size) % self.__strides
        filter_pos = tuple((np.array(self.__dO.shape[-2:]) - 1) * self.__strides + 1 + moduli)
        dF_filters = np.zeros(self.__dO.shape[:-2] + filter_pos)
        dF_filters[..., 0:filter_pos[0]:self.__strides[0], 0:filter_pos[1]:self.__strides[1]] = self.__dO

        self.__dF = (1 / m) * np.sum(self.__convolve(self.__X, dF_filters, strides = (1, 1), einsum_str = 'ijklmnop,iqop->iqjkl'), axis = 0)
        
        if self.use_bias:
            out_shape = self.__dO.shape
            axis = tuple([ax for ax in range(len(out_shape)) if ax != CHANNEL_AXIS])
            denom = np.prod(out_shape) / out_shape[CHANNEL_AXIS]
            self._dB = (1 / denom) * np.sum(self.__dO, axis = axis).reshape(self._bias.shape)          
        
        # remove padding elements when returning the deltas to get proper shape
        padding = self.__kernel_size - 1
        padding_up, padding_down, padding_left, padding_right = self.__calculate_padding(padding)
        padded_height, padded_width = np.array(self.input_shape[-2:]) + padding
        return self.__dX[...,
                         padding_up: padded_height - padding_down,
                         padding_left: padded_width - padding_right]

    @property
    def weights(self): return self.__filters
    @weights.setter
    def weights(self, weights): self.__filters = weights
    @property
    def dW(self): return self.__dF
    @property
    def Z(self): return self.__X
    
    def output_shape(self, input_shape):
        if not self._output_shape:
            if not input_shape:
                return None

            image_shape = input_shape[-2:]
            for img_shape, kernel_shape in zip(image_shape, self.__kernel_size):
                if img_shape < kernel_shape:
                    raise InvalidShapeError(f'Negative dimension size caused by subtracting {kernel_shape} '
                                            f'from {img_shape} for layer {self._name}.')

            conv_shape = self.__conv_steps(input_shape, self.__kernel_size, self.__strides)
            self._output_shape = list(input_shape)
            self._output_shape[-2:] = conv_shape
            self._output_shape = tuple(self._output_shape)
        return self._output_shape
    
    def __convolve(self,
                   activations,
                   filters = None,
                   strides = None,
                   full = False,
                   einsum_str = 'ijklmnop,qjop->iqkl'):
        if full:
            activations = self.__add_padding(activations.copy(), full = True)
            filters = self.__filters.copy() if filters is None else filters.copy()
            filters = np.rot90(np.rot90(filters, axes = (2, 3)), axes = (2, 3)).swapaxes(0, 1)
        else:
            filters = self.__filters if filters is None else filters

        strides = self.__strides if strides is None else strides
        if not isinstance(strides, tuple):
            strides = tuple(strides)

        kernel_size = filters[0].shape[-2:]
        views = view_as_windows(activations, (1, 1) + kernel_size, (1, 1) + strides)

        convolutions = np.einsum(einsum_str, views, filters)
        return convolutions
    
    def __conv_steps(self, activations_shape, kernel_size, strides):
        x_steps = np.floor_divide(activations_shape[-1] - kernel_size[1], strides[1]) + 1
        y_steps = np.floor_divide(activations_shape[-2] - kernel_size[0], strides[0]) + 1
        
        return (x_steps, y_steps)
        
    def __add_padding(self, activations, full = False):
        n_images = activations.shape[0]
        channel_shape = np.array(activations.shape[-2:])
        padding = self.__kernel_size - 1
        padding_steps = np.array([1, 1])
        if full:
            # padding doubles on a full convolution
            padding = padding * 2

            # in case of full convolution, activations need to be spaced out with padding = strides between elements
            moduli = (channel_shape - self.__kernel_size) % self.__strides
            channel_shape = (channel_shape - 1) * self.__strides + 1 + moduli
            padding_steps = self.__strides

        padding_up, padding_down, padding_left, padding_right = self.__calculate_padding(padding)

        padded = np.zeros(activations.shape[:-2] + (channel_shape[0] + padding[0], channel_shape[1] + padding[1]))
        padded_height, padded_width = padded.shape[-2:]

        padded[...,
               padding_up: padded_height - padding_down: padding_steps[1],
               padding_left: padded_width - padding_right: padding_steps[0]] = activations

        return padded

    def __calculate_padding(self, padding):
        pad_left = np.ceil(padding[0] / 2).astype('int16')
        pad_right = padding[0] - pad_left
        pad_up = np.ceil(padding[1] / 2).astype('int16')
        pad_down = padding[1] - pad_up

        return pad_up, pad_down, pad_left, pad_right
        
    def __check_tuple(self, tuple_, argument_name):
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

