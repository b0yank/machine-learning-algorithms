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


TIMESTEP_AXIS = 1