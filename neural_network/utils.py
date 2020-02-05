import numpy as np
from scipy.sparse import lil_matrix
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

def onehot_encode(y, num_labels=None):
    """One-hot encodes a vector.

       Warning: Function expects y to be a 1D vector and, if it has more than one dimension, reshapes it into a vector.
                This is done in order to make use of the scipy.sparse package
                Returns a 2D sparse matrix.
                If labels are passed, they should be either a list, set or a dictionary {word: index}
    """
    y_reshaped = y.ravel()

    if num_labels is None:
        num_labels = len(set(y_reshaped))

    y_onehot = lil_matrix(y_reshaped.shape + (num_labels,), dtype='int16')
    y_onehot[np.arange(len(y_reshaped)), y_reshaped] = 1

    #y_onehot_old = np.zeros((len(labels), len(y)), dtype='int16')
    #for label in labels:
    #    label_row = y_onehot_old[labels.index(label)]
    #    label_row[np.nonzero(y == label)[0]] = 1
            
    return y_onehot.tocsr()
