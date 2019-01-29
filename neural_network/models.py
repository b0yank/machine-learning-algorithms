import numpy as np

from neural_network.utils import InvalidShapeError, Mode
from .layers.core import Layer
from .losses import Loss, CrossEntropyLoss, MeanSquaredLoss
from .optimizers import Optimizer, Adam, SGD

# LOSS FUNCTIONS
MEAN_SQUARED = 'mse'
CROSS_ENTROPY = 'crossentropy'

# OPTIMIZERS
ADAM = 'adam'
SGD = 'sgd'

class Sequential:
    """Linear stack of layers.
    """
    def __init__(self, layers):
        if any([not isinstance(layer, Layer) for layer in layers]):
            raise TypeError('The added layer must be an instance of class Layer.')
        self.layers = layers
        self.__set_names()
        self.__compiled = False
        
    def compile(self, optimizer, loss):
        self.__loss = self.__get_loss(loss)
        self.__optimizer = self.__get_optimizer(optimizer)
        self.__compiled = True
        
    def fit(self, X, y, batch_size = 32, epochs = 1):
        if not self.__compiled:
            raise RuntimeError('You must compile a model before training/testing. '
                               'Use `model.compile(optimizer, loss)`.')
            
        n_samples = X.shape[0]
        self.__labels = list(set(y))
        y_onehot = self.__onehot_encode(y)
        X_batches = [X[i:i + batch_size] for i in range(0, n_samples, batch_size)]
        y_batches = [y_onehot[i:i + batch_size] for i in range(0, n_samples, batch_size)]
        
        for it in range(epochs):
            for batch_index in range(len(X_batches)):
                self.__forward(X_batches[batch_index], mode = Mode.TRAIN)
                self.__backward(X_batches[batch_index], y_batches[batch_index])

            self.__optimizer.decay_lr()

    def evaluate(self, X, y, batch_size = 32):
        n_samples = len(y)
        y_onehot = self.__onehot_encode(y)
        X_batches = [X[i:i + batch_size] for i in range(0, n_samples, batch_size)]
        y_batches = [y[i:i + batch_size] for i in range(0, n_samples, batch_size)]
        y_oh_batches = [y_onehot[i:i + batch_size] for i in range(0, n_samples, batch_size)]

        loss = 0
        accuracy = 0
        n_batches = len(X_batches)
        
        for batch_index in range(n_batches):
                X_batch = X_batches[batch_index]
                y_batch = y_batches[batch_index]
                onehot_batch = y_oh_batches[batch_index]

                self.__forward(X_batch, mode = Mode.TEST)
                activations = self.layers[-1].activations
                loss += self.__loss.get_loss(onehot_batch, activations)
                
                predictions = np.array([self.__labels[np.argmax(activation)] for activation in activations])
                diff = y_batch - predictions
                accuracy +=  1 - (np.count_nonzero(diff) / len(y_batch))

        loss /= n_batches
        accuracy /= n_batches
        return loss, accuracy
    
    def __forward(self, X_batch, mode):
        activations = X_batch
        for layer in self.layers:
            layer.forward(activations, mode)
            activations = layer.activations
    
    def __backward(self, X_batch, y_batch):
        output_layer = self.layers[-1]
        delta = self.__loss.output_deriv(y = output_layer.activations, t = y_batch)
    
        index = len(self.layers) - 1
        while index >= 0:
            layer = self.layers[index]
            prev_activations = self.layers[index - 1].activations if index > 0 else X_batch
            delta = layer.backward(prev_activations, delta)
            if layer.trainable:
                self.__optimizer.update_weights(layer)
        
            index -= 1
    
    def __get_optimizer(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == ADAM:
                return Adam()
            elif optimizer == SGD:
                return SGD()
            raise ValueError(f'Optimizer of type {optimizer} not recognized. '
                         f'Choose between Adam optimizer(\'{ADAM}\') '
                         f'and stochastic gradient descent(\'{SGD}\')')
        
        elif isinstance(optimizer, Optimizer):
            return optimizer
        
        else:
            raise ValueError('Invalid optimizer. Please pass an object which inherits '
                             'the Optimizer class, or name of optimizer as string. '
                             f'Supported optimizers: ({ADAM}, {SGD}).')
     
    def __get_loss(self, loss):
        if isinstance(loss, str):
            if loss == MEAN_SQUARED:
                return MeanSquaredLoss()
            elif loss == CROSS_ENTROPY:
                return CrossEntropyLoss()
            raise ValueError(f'Loss of type {loss} not recognized. '
                     f'Choose between mean squared loss(\'{MEAN_SQUARED}\') '
                     f'and cross-entropy loss(\'{CROSS_ENTROPY}\')')
            
        elif isinstance(loss, Loss):
            return loss
        
        else:
            raise ValueError('Invalid loss function. Please pass an object which inherits the Loss class, '
                             'or name of loss function as string. Supported loss functions: '
                             f'({MEAN_SQUARED}, {CROSS_ENTROPY}).')
            
    def __onehot_encode(self, y):
        y_onehot = np.zeros((len(self.__labels), len(y)), dtype='int16')
        for label in self.__labels:
            label_row = y_onehot[self.__labels.index(label)]
            label_row[np.nonzero(y == label)[0]] = 1
            
        return y_onehot.T
    
    def __set_names(self):
        nums = dict()
        for layer in self.layers:
            prefix = type(layer).__name__.lower()
            nums[prefix] = nums[prefix] + 1 if prefix in nums else 1
                
            layer._name = f'{prefix}_{nums[prefix]}'

