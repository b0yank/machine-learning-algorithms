import numpy as np

from .core import Layer
from neural_network.utils import TIMESTEP_AXIS


class Embedding(Layer):
    def __init__(self, input_dim, output_dim,
                 embeddings_initializer = 'uniform',
                 input_length = None,
                 weights = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        if weights is None:
            self.__embeddings = self._add_weight((input_dim, output_dim), embeddings_initializer)
        elif weights.shape != (input_dim, output_dim):
            raise ValueError(f'Embedding matrix should have dimensions ({input_dim}, {output_dim}).')
        else:
            self.__embeddings = weights
        self.input_length = input_length

    @property
    def weights(self): return self.__embeddings
    @weights.setter
    def weights(self, weights): self.__embeddings = weights
    @property
    def dW(self): return self.__dE

    def forward(self, prev_activations, train_mode = True):
        output = np.array([self.__embeddings[wordind] for wordind in prev_activations])\
                    .reshape(prev_activations.shape + (self.output_dim,))

        self.__used_words = list(set(prev_activations.ravel()))
        sequence_length = output.shape[TIMESTEP_AXIS]
        if self.input_length is not None and\
            sequence_length != self.input_length:
            if sequence_length < self.input_length:
                diff = self.input_length - sequence_length
                output = np.pad(output, [(0, 0) if i != TIMESTEP_AXIS else (0, diff) for i in range(len(output.shape))], 'constant')
            else:
                output = output.swapaxes(0, TIMESTEP_AXIS)[:sequence_length].swapaxes(0, TIMESTEP_AXIS)

        self.activations = output

    def backward(self, prev_activations, delta, train_mode = True):
        self.__dE = np.zeros(self.__embeddings.shape)
        for x, d in zip(prev_activations.ravel(), delta.reshape(-1, delta.shape[-1])):
            self.__dE[x] += d

        return prev_activations

    def output_shape(self, input_shape):
        return input_shape[:TIMESTEP_AXIS] + self.input_length + input_shape[TIMESTEP_AXIS:]