import numpy as np

from .core import Layer
from neural_network.utils import TIMESTEP_AXIS, onehot_encode


class Embedding(Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer = 'uniform',
                 input_length = None,
                 weights = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        if weights is None:
            self._embeddings = self._add_weight((input_dim, output_dim), embeddings_initializer)
        elif weights.shape != (input_dim, output_dim):
            raise ValueError(f'Embedding matrix should have dimensions ({input_dim}, {output_dim}).')
        else:
            self._embeddings = weights
        self.input_length = input_length

    @property
    def weights(self): return self._embeddings
    @weights.setter
    def weights(self, weights): self._embeddings = weights
    @property
    def dW(self): return self.__dE

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        self.__onehot = onehot_encode(prev_activations, self.input_dim)

        # onehot_encode function returns a sparse, 2D matrix, so the result needs to be reshaped to multidimensional
        embeddings = (self.__onehot @ self.weights).reshape(prev_activations.shape + self.weights.shape[1:])

        sequence_length = embeddings.shape[TIMESTEP_AXIS]
        if self.input_length is not None and sequence_length != self.input_length:
            raise ValueError(f'"input_length" is {self.input_length}, but received input has shape {prev_activations.shape}')

        self.activations = embeddings

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        dE = self.__onehot.T @ delta.reshape((-1, delta.shape[-1]))
        self.__dE = dE

        delta_new = delta @ self.weights.T
        return np.sum(delta_new, axis=-1)

    def output_shape(self, input_shape):
        return input_shape[:TIMESTEP_AXIS] + self.input_length + input_shape[TIMESTEP_AXIS:]
