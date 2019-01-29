from enum import Enum

class ModelNotTrainedError(Exception):
    pass

class InvalidInputError(Exception):
    pass

class InvalidShapeError(Exception):
    def __init__(self, expected_shape, shape):
        self.expected_shape = expected_shape
        self.shape = shape

class Mode(Enum):
    """ Enumeration indicating forward propagation mode.

    Main use is when making a test run and the model contains batch normalization layers.
    """
    TRAIN = 0
    TEST = 1
