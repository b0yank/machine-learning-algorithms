import numpy as np
from abc import ABCMeta, abstractmethod
from skimage.util.shape import view_as_windows

from .core import Layer, Layer2D
from neural_network.utils import CHANNEL_AXIS, PADDING_SAME, PADDING_VALID, PADDING_TYPES, im2col

HEIGHT_AXIS = -2
WIDTH_AXIS = -1

class Pooling2D(Layer2D, metaclass = ABCMeta):
    """Abstract class for different pooling 2D layers.
    """
    def __init__(self, pool_size, strides, padding):
        Layer2D.__init__(self, kernel_size = pool_size,
                         strides = strides,
                         padding = padding)
        self.trainable = False

    def forward(self, prev_activations, train_mode = True):
        self.input_shape = prev_activations.shape
        self.activations = self._pool(prev_activations, pool_size = self._kernel_size, strides = self._strides, padding = self.padding)    

    def _pool(self, inputs, pool_size, strides, padding):
        if padding == PADDING_SAME:
            inputs = self._add_padding(inputs)

        strides = self._strides if strides is None else strides
        if not isinstance(strides, tuple):
            strides = tuple(strides)

        kernel_size = self._kernel_size if pool_size is None else pool_size
        if not isinstance(kernel_size, tuple):
            kernel_size = tuple(kernel_size)

        views = view_as_windows(inputs, (1, 1) + kernel_size, (1, 1) + strides)
        pools = self._pooling_function(views)
        return pools.reshape(views.shape[:-4])

    @abstractmethod
    def _pooling_function(self, inputs):
       pass

    def _compute_deltas(self, mask, delta, activations_shape, activations_padded_shape):
        delta = mask * np.broadcast_to(delta[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis], mask.shape)
        delta = np.squeeze(delta).reshape((delta.shape[0], delta.shape[1], -1, delta.shape[-2], delta.shape[-1]))

        delta_view = view_as_windows(np.ones(activations_shape), (1, 1) + tuple(self._kernel_size), (1, 1) + tuple(self._strides))
        delta = np.einsum('ijklmnop,ijqop->ijklop', delta_view, delta)
        bs, n_f, n_y, n_x, w_y, w_x = delta.shape

        dX = np.zeros(activations_padded_shape)

        # sum all the deltas which are split into views
        for i in range(w_y):
            for j in range(w_x):
                dX[:,
                   :,
                   i: n_y * self._strides[1] + i: self._strides[1],
                   j: n_x * self._strides[0] + j: self._strides[0]] += delta[:, :, :, :, i, j]

        return dX

    def _check_delta_padding(self, delta):
        if self.padding == PADDING_VALID:
            return delta

        # remove padding elements when returning the deltas to get proper shape
        delta = self._clip_padding(delta)
        return delta

class MaxPooling2D(Pooling2D):
    """Max pooling operation for spatial data.
    """
    def __init__(self, pool_size, strides, padding):
        super().__init__(pool_size, strides, padding)

    def backward(self, prev_activations, delta, train_mode = True):
        activations_padded = self._add_padding(prev_activations) if self.padding == PADDING_SAME else prev_activations

        views = view_as_windows(activations_padded, (1, 1) + tuple(self._kernel_size), (1, 1) + tuple(self._strides))
        activations = np.broadcast_to(self.activations[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis], views.shape)

        mask = (views == activations).astype('int8')
        
        dX = self._compute_deltas(mask, delta, prev_activations.shape, activations_padded.shape)
        return self._check_delta_padding(dX)

    def _pooling_function(self, inputs):
        return np.max(inputs, axis = (HEIGHT_AXIS, WIDTH_AXIS))

class AveragePooling2D(Pooling2D):
    """Average pooling operation for spatial data.
    """
    def __init__(self, pool_size, strides, padding):
        super().__init__(pool_size, strides, padding)

    def backward(self, prev_activations, delta, train_mode = True):
        activations_padded = self._add_padding(prev_activations) if self.padding == PADDING_SAME else prev_activations

        views = view_as_windows(activations_padded, (1, 1) + tuple(self._kernel_size), (1, 1) + tuple(self._strides))
        activations = np.broadcast_to(self.activations[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis], views.shape)
        mask = (np.ones(views.shape) / (self._kernel_size[0] * self._kernel_size[1]))

        dX = self._compute_deltas(mask, delta, prev_activations.shape, activations_padded.shape)
        return self._check_delta_padding(dX)

    def _pooling_function(self, inputs):
        return np.mean(inputs, axis = (HEIGHT_AXIS, WIDTH_AXIS))

class GlobalPooling2D(Layer2D):
    """Abstract class for different global pooling 2D layers.
    """
    def __init__(self):
        super(Layer, self).__init__()
        self.trainable = False

    def forward(self, prev_activations, train_mode = True):
        self.activations = self._pooling_function(prev_activations)

    @abstractmethod
    def _pooling_function(self, inputs):
        pass

class GlobalAveragePooling2D(GlobalPooling2D):
    """Global average pooling operation for spatial data.
    """
    def __init__(self):
        super().__init__()

    def backward(self, prev_activations, delta, train_mode = True):
        return np.ones(prev_activations.shape) * delta[..., np.newaxis, np.newaxis] / prev_activations.shape[0]

    def _pooling_function(self, inputs):
        return np.mean(inputs, axis = (HEIGHT_AXIS, WIDTH_AXIS))

class GlobalMaxPooling2D(GlobalPooling2D):
    """Global max pooling operation for spatial data.
    """
    def __init__(self):
        super().__init__()

    def backward(self, prev_activations, delta, train_mode = True):
        batch_size, n_channels, img_height, img_width = prev_activations.shape
        activations_reshaped = prev_activations.reshape(batch_size * n_channels, img_height * img_width)
        argmax = np.argmax(activations_reshaped, axis = -1)

        delta_mask = np.zeros(activations_reshaped.shape)
        delta_mask[range(batch_size * n_channels), argmax] = 1
        delta_mask = delta_mask.reshape(prev_activations.shape)

        return delta_mask * delta[..., np.newaxis, np.newaxis]



    def _pooling_function(self, inputs):
        return np.max(inputs, axis = (HEIGHT_AXIS, WIDTH_AXIS))


# Aliases
AvgPool2D = AveragePooling2D
MaxPool2D = MaxPooling2D
GlobalMaxPool2D = GlobalMaxPooling2D
GlobalAvgPool2D = GlobalAveragePooling2D