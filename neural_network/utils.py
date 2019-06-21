import numpy as np
from skimage.util.shape import view_as_windows

class ModelNotTrainedError(Exception):
    pass

class InvalidInputError(Exception):
    pass

class ModelArchitectureError(Exception):
    pass

class NotBuiltError(Exception):
    pass

class InvalidShapeError(Exception):
    def __init__(self, expected_shape, shape):
        self.expected_shape = expected_shape
        self.shape = shape

CHANNEL_AXIS = 1
TIMESTEP_AXIS = 1

PADDING_SAME = 'same'
PADDING_VALID = 'valid'
PADDING_TYPES = {PADDING_SAME, PADDING_VALID}

def im2col(img, kernel_size, strides):
    """
    Converts a 4D image matrix into a 2D matrix

    img - image numpy matrix with dimensions (batch_size, n_channels, image_height, image_width)
    kernel_size - Tuple or numpy array. Must contain two values - (kernel_height, kernel_width)
    strides - Tuple or numpy array. Must contain two values - (stride_y, stride_x)
    """
    views = np.squeeze(view_as_windows(img, (1, 1) + tuple(kernel_size), (1, 1) + tuple(strides)))

    batch_size, n_channels, strides_y, strides_x, kernel_y, kernel_x = views.shape
    return views.reshape((batch_size * n_channels * strides_y * strides_x, kernel_y * kernel_x))


def col2im(col, desired_shape, kernel_size, strides):
    """
    Converts a 2D matrix into a 4D image matrix

    col - column 2D numpy matrix
    desired_shape - Tuple. Desired shape to reshape col to - dimensions (batch_size, n_channels, image_height, image_width)
    kernel_size - Tuple or numpy array. Must contain two values - (kernel_height, kernel_width)
    strides - Tuple or numpy array. Must contain two values - (stride_y, stride_x)
    """
    #if np.prod(col.shape) != np.prod(desired_shape):
    #    raise ValueError(f'Shape of \'col\' {col.shape} incompatible with \'desired_shape\' {desired_shape}')

    img = np.ones(desired_shape)
