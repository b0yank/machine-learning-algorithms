from .core import Layer

RESIDUAL_INPUT = 'input'
RESIDUAL_GRADIENT = 'gradient'

class ResidualIn(Layer):
    def __init__(self, residual_node):
        self.residual_node = residual_node
        self.trainable = False

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        self.residual_node[RESIDUAL_INPUT] = prev_activations
        return prev_activations

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        return delta + self.residual_node[RESIDUAL_GRADIENT]

class ResidualOut(Layer):
    def __init__(self, residual_node):
        self.residual_node = residual_node
        self.trainable = False

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        return prev_activations + self.residual_node[RESIDUAL_INPUT]

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        self.residual_node[RESIDUAL_GRADIENT] = delta
        return delta

class ResidualConnection:
    def __init__(self):
        node = dict()
        self.__res_in = ResidualIn(node)
        self.__res_out = ResidualOut(node)

    def get_residual_in(self):  return self.__res_in
    def get_residual_out(self):  return self.__res_out
