import numpy as np
from skimage.util.shape import view_as_windows

from .core import Layer, Layer2D
from neural_network.utils import InvalidShapeError, CHANNEL_AXIS, PADDING_SAME, PADDING_VALID, PADDING_TYPES

CONV2D_NDIMS = 4

class Conv2D(Layer2D):
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
        super().__init__(kernel_size = kernel_size,
                         activation = activation,
                         use_bias = use_bias,
                         input_shape = input_shape,
                         strides = strides,
                         padding = padding,
                         kernel_reg_l2 = kernel_reg_l2)
        if use_bias:
            shape = tuple([1 if index != CHANNEL_AXIS else filters for index in range(CONV2D_NDIMS)])
            self._bias = np.zeros(shape)
        else:
            self._bias = 0
    
        self.filters = filters
        self.__filters = None        

    @property
    def weights(self): return self.__filters
    @weights.setter
    def weights(self, weights): self.__filters = weights
    @property
    def dW(self): return self.__dF
    @property
    def Z(self): return self.__X
        
    def forward(self, prev_activations, train_mode = True):
        self.input_shape = prev_activations.shape
        if self.padding == PADDING_SAME:
            prev_activations = self._add_padding(prev_activations)

        if self.__filters is None:
            self.__filters = np.random.normal(size = ((self.filters, prev_activations.shape[CHANNEL_AXIS])) + tuple(self._kernel_size))
        
        self.__convolutions = self.__convolve(prev_activations)
        self.__X = prev_activations
        self.activations = self._activation.get_activation(self.__convolutions + self.bias)
    
    def backward(self, prev_activations, delta, train_mode = True):
        self.__dO = self._activation.get_delta(self.__convolutions + self.bias, delta)
        self.__dX = self.__convolve(self.__dO, self.__filters, strides = (1, 1), full = True)
        
        # filters need to have padding = strides added between elements of output derivative (dO)
        # they will act as the kernel in the derivative convolution done to find dF
        moduli = (np.array(self.__X.shape[-2:]) - self._kernel_size) % self._strides
        filter_pos = tuple((np.array(self.__dO.shape[-2:]) - 1) * self._strides + 1 + moduli)
        dF_filters = np.zeros(self.__dO.shape[:-2] + filter_pos)
        dF_filters[..., 0:filter_pos[0]:self._strides[0], 0:filter_pos[1]:self._strides[1]] = self.__dO

        self.__dF = np.mean(self.__convolve(self.__X, dF_filters, strides = (1, 1), einsum_str = 'ijklmnop,iqop->iqjkl'), axis = 0)
        
        if self.use_bias:
            axis = tuple([ax for ax in range(len(self.__dO.shape)) if ax != CHANNEL_AXIS])
            self._dB = np.mean(self.__dO, axis = axis).reshape(self._bias.shape)   

        if self.padding == PADDING_VALID:
            return self.__dX
            
        dX = np.zeros(self.__X.shape)
        dX[..., 0:self.__dX.shape[-2], 0:self.__dX.shape[-1]] = self.__dX
        
        # remove padding elements when returning the deltas to get proper shape
        dX = self._clip_padding(dX)
        return dX

    def __convolve(self,
                   activations,
                   filters = None,
                   strides = None,
                   full = False,
                   einsum_str = 'ijklmnop,qjop->iqkl'):
        if full:
            activations = self._add_padding(activations.copy(), full = True)
            filters = self.__filters.copy() if filters is None else filters.copy()
            filters = np.rot90(np.rot90(filters, axes = (2, 3)), axes = (2, 3)).swapaxes(0, 1)
        else:
            filters = self.__filters if filters is None else filters

        strides = self._strides if strides is None else strides
        if not isinstance(strides, tuple):
            strides = tuple(strides)

        kernel_size = filters[0].shape[-2:]
        views = view_as_windows(activations, (1, 1) + kernel_size, (1, 1) + strides)

        convolutions = np.einsum(einsum_str, views, filters)
        return convolutions


