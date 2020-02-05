import numpy as np
from scipy import sparse
from abc import ABCMeta, abstractmethod
from nltk.translate.bleu_score import sentence_bleu

from utils.graph import Graph, INPUTS_NAME, GRADIENTS_NAME
from neural_network.utils import InvalidShapeError, ModelArchitectureError, onehot_encode
from .layers.core import Layer
from .losses import Loss, CrossEntropyLoss, MeanSquaredLoss
from .optimizers import Optimizer, Adam, SGD

# loss functions
MEAN_SQUARED = 'mse'
CROSS_ENTROPY = 'crossentropy'

# optimizers
ADAM = 'adam'
SGD = 'sgd'

# encoder-decoder input/output names
ENCODER_INPUTS_NAME = 'encoder_inputs'
ENCODER_GRADIENTS_NAME = 'encoder_gradients'
DECODER_INPUTS_NAME = 'decoder_inputs'
DECODER_GRADIENTS_NAME = 'decoder_gradients'

# used when models call self.evaluate within self.fit
# saves on splitting the data into batches again
INPUT_IN_BATCHES = 'batched'

class Model(metaclass = ABCMeta):
    """Abstract model class.
    """
    def __init__(self):
        self._compiled = False
        self._optimizer = None
        self._loss = None        

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self._optimizer = self._get_optimizer(optimizer)
        self.loss = loss
        self._loss = self._get_loss(loss)
        self._compiled = True

    @property
    def layers(self): pass

    @abstractmethod
    def _define_graph(self, residual_connections = []):
        pass

    def _get_optimizer(self, optimizer):
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
     
    def _get_loss(self, loss):
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

    def _set_names(self):
        nums = dict()
        for layer in self.layers:
            prefix = type(layer).__name__.lower()
            nums[prefix] = nums[prefix] + 1 if prefix in nums else 1
                
            layer._name = f'{prefix}_{nums[prefix]}'

class Sequential(Model):
    """Linear stack of layers.
    """
    def __init__(self, layers):
        super().__init__()
        if any([not isinstance(layer, Layer) for layer in layers]):
            raise TypeError('The added layer must be an instance of class Layer.')
        self.__layers = list(layers)
        
        self._set_names()
        self._define_graph()

    @property
    def layers(self): return self.__layers
        
    def fit(self, X, y, batch_size = 32, epochs = 1, *args, **kwargs):
        if not self._compiled:
            raise RuntimeError('You must compile a model before training/testing. '
                               'Use `model.compile(optimizer, loss)`.')
        
        n_samples = X.shape[0]
        self.__labels = list(set(y))
        y_onehot = onehot_encode(y) if len(self.__labels) > 2 else y.reshape(-1, 1)
        X_batches = [X[i:i + batch_size] for i in range(0, n_samples, batch_size)]
        y_batches = [y_onehot[i:i + batch_size] for i in range(0, n_samples, batch_size)]
        
        for it in range(epochs):
            for batch_index in range(len(X_batches)):
                y_batch = y_batches[batch_index]

                # y_batch would be a sparse matrix if number of labels > 2 and y was onehot encoded
                if isinstance(y_batch, sparse.csr_matrix):
                    y_batch = y_batch.toarray()

                self._forward(X_batches[batch_index], train_mode = True)
                self._backward(X_batches[batch_index], y_batch, train_mode = True)
            
            print(f'Epoch {it + 1}:')
            loss, accuracy = self.evaluate(X_batches, y_batches, batch_size, **{INPUT_IN_BATCHES: True})
            print(f'Training loss: {loss}, training accuracy: {accuracy}')

            self._optimizer.decay_lr()

    def evaluate(self, X, y, batch_size = 32, *args, **kwargs):
        if not self._compiled:
            raise RuntimeError('You must compile a model before training/testing. '
                               'Use `model.compile(optimizer, loss)`.')

        if INPUT_IN_BATCHES in kwargs and kwargs[INPUT_IN_BATCHES] == True:
            X_batches = X
            y_oh_batches = y

            # y_batches elements would be sparse matrices if number of labels > 2 and y was onehot encoded
            if isinstance(y_oh_batches[0], sparse.csr_matrix):
                y_batches = [oh_batch.toarray().argmax(axis=1).reshape(-1, 1) for oh_batch in y_oh_batches]
            else:
                y_batches = y
        else:
            n_samples = X.shape[0]
            X_batches = [X[i:i + batch_size] for i in range(0, n_samples, batch_size)]
            y_batches = [y[i:i + batch_size] for i in range(0, n_samples, batch_size)]
            if len(self.__labels) > 2:
                y_oh_batches = [onehot_encode(batch, num_labels=len(self.__labels)) for batch in y_batches]
            else:
                y_oh_batches = y_batches

        loss = 0
        accuracy = 0
        n_batches = len(X_batches)
        
        for batch_index in range(n_batches):
                X_batch = X_batches[batch_index]
                y_batch = y_batches[batch_index].reshape(-1, 1)

                onehot_batch = y_oh_batches[batch_index]

                # onehot_batch would be a sparse matrix if number of labels > 2 and y was onehot encoded
                if isinstance(onehot_batch, sparse.csr_matrix):
                    onehot_batch = onehot_batch.toarray()

                self._forward(X_batch, train_mode = False)
                activations = self.layers[-1].activations
                
                # if there is more than one activation per sample, then the labels were onehot encoded  
                if self.layers[-1].activations.shape[-1] == 1:
                    y_batch = y_batch
                    current_loss = self._loss.get_loss(y_batch, activations)
                    predictions = np.array([self.__labels[int(np.round(activation))] for activation in activations]).reshape((activations.shape[0], -1))
                else:
                    current_loss = self._loss.get_loss(onehot_batch, activations)
                    predictions = np.array([self.__labels[np.argmax(activation)] for activation in activations]).reshape((activations.shape[0], -1))
                loss += current_loss
                diff = y_batch - predictions
                accuracy +=  1 - (np.count_nonzero(diff) / len(y_batch))

        loss /= n_batches
        accuracy /= n_batches
        return loss, accuracy
    
    def _forward(self, X_batch, train_mode = True, *args, **kwargs):
        self._graph.forward(inputs_dict={INPUTS_NAME: X_batch},
                            additional_params={'train_mode': train_mode},
                            *args,
                            **kwargs)

        outputs = self._graph.outputs

        if len(outputs) == 1:
            return outputs[0]
        return outputs
    
    def _backward(self, X_batch, y_batch, train_mode = True, *args, **kwargs):
        delta = self._loss.output_deriv(y = self.layers[-1].activations, t = y_batch)

        self._graph.backward(gradients_dict={GRADIENTS_NAME: delta},
                             additional_params = {'train_mode': train_mode},
                             node_func=self._optimizer.update_weights,
                             *args,
                             **kwargs)

    def _define_graph(self, residual_connections = []):
        self.layers[0].input_edges[INPUTS_NAME] = None
        self.layers[-1].output_edges[GRADIENTS_NAME] = None
        node_connections = [(self.layers[idx], self.layers[idx + 1], (INPUTS_NAME, GRADIENTS_NAME)) for idx in range(0, len(self.layers) - 1)]
        node_connections += residual_connections

        self._graph = Graph(node_connections)

class EncoderDecoder(Model):
    def __init__(self,
                 encoder_layers,
                 decoder_layers,
                 link_layers,
                 start_of_sequence_token_id,
                 end_of_sequence_token_id,
                 padding_token_id = 0):
        """Abstract encoder-decoder architecture for sequence models.

            Parameters: 
            'encoder_layers'                - a list of layers that will comprise the encoder
            'decoder_layers'                - a list of layers that will comprise the decoder
            'link_layers'                   - a list of layers linking the encoder and the decoder
            'start_of_sequence_token_id'    - the id of the start of sequence token used
            'end_of_sequence_token_id'      - the id of the end of sequence token used
            'padding_token_id'              - id of the sequence padding token used

            Warning: All layers in 'link_layers' must be a part of the 'decoder_layers' list and must be able to accept two inputs (from the decoder and the encoder).
        """
        super().__init__()

        if any([ll not in decoder_layers for ll in link_layers]):
            raise ModelArchitectureError('\'link_layers\' must be a part of the \'decoder_layers\' list.')

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.link_layers = link_layers
        self.start_of_sequence_token_id = start_of_sequence_token_id
        self.end_of_sequence_token_id = end_of_sequence_token_id
        self.padding_token_id = padding_token_id

        self._set_names()
        self._define_graph()

    @property
    def layers(self): return self.encoder_layers + self.decoder_layers

    def fit(self, encoder_inputs, decoder_inputs, batch_size = 32, epochs = 1, *args, **kwargs):
        if not self._compiled:
            raise RuntimeError('You must compile a model before training/testing. '
                               'Use `model.compile(optimizer, loss)`.')

        ## merge batch_size and sequence length dimensions into one
        #encoder_inputs_flat = encoder_inputs.ravel()#encoder_inputs.reshape((sum(encoder_inputs.shape[:2]),) + encoder_inputs.shape[2:])
        #decoder_inputs_flat = decoder_inputs.ravel()#decoder_inputs.reshape((sum(decoder_inputs.shape[:2]),) + decoder_inputs.shape[2:])
        ##y_reshaped = y.reshape((-1,))
        n_samples = encoder_inputs.shape[0]
        
        self.__labels = list([0]*3459)#list(set(decoder_inputs))

        encoder_batches = [encoder_inputs[i:i + batch_size] for i in range(0, n_samples, batch_size)]
        decoder_batches = [decoder_inputs[i:i + batch_size] for i in range(0, n_samples, batch_size)]
        #y_batches = [y_reshaped[i:i + batch_size] for i in range(0, n_samples, batch_size)]
        
        for it in range(epochs):
            for batch_index in range(len(encoder_batches)):
                self._forward(encoder_inputs = encoder_batches[batch_index],
                              decoder_inputs = decoder_batches[batch_index],
                              train_mode=True)
                self._backward(encoder_inputs = encoder_batches[batch_index],
                               decoder_inputs = decoder_batches[batch_index],
                               y_batch = onehot_encode(decoder_batches[batch_index],
                                                       num_labels=len(self.__labels))\
                                                           .toarray()\
                                                           .reshape(decoder_batches[batch_index].shape + (len(self.__labels),)),
                               train_mode = True)
            
            print(f'Epoch {it + 1}:')
            loss, bleu = self.evaluate(encoder_inputs=encoder_batches, y=decoder_batches, **{INPUT_IN_BATCHES: True})
            print(f'Training loss: {loss}, training BLEU score: {bleu}')

            self._optimizer.decay_lr()

    def evaluate(self, encoder_inputs, y, batch_size = 32, epochs = 1, *args, **kwargs):
        if not self._compiled:
            raise RuntimeError('You must compile a model before training/testing. '
                               'Use `model.compile(optimizer, loss)`.')

        if INPUT_IN_BATCHES in kwargs and kwargs[INPUT_IN_BATCHES] == True:
            encoder_batches = encoder_inputs
            y_batches = y
        else:
            ## merge batch_size and sequence length dimensions into one
            #encoder_inputs = encoder_inputs.ravel()
            n_samples = encoder_inputs.shape[0]

            encoder_batches = [encoder_inputs[i:i + batch_size] for i in range(0, n_samples, batch_size)]
            y_reshaped = y.reshape((-1,))
            y_batches = [y_reshaped[i:i + batch_size] for i in range(0, n_samples, batch_size)]

        loss = 0
        bleu_score = 0
        n_batches = len(encoder_inputs)
        
        for batch_index in range(n_batches):
                encoder_batch = encoder_batches[batch_index]
                y_batch = y_batches[batch_index]
                onehot_batch = onehot_encode(y_batch, num_labels=len(self.__labels)).toarray()

                self._graph.clear_messages()
                self._forward(encoder_inputs=encoder_batch, decoder_inputs=y_batch, train_mode = False)
                activations = self.decoder_layers[-1].activations
                
                if self.layers[-1].activations.shape[-1] == 1:
                    current_loss = self._loss.get_loss(y_batch.reshape(-1, 1), activations)
                    hypotheses = np.array([self.__labels[int(np.round(activation))] for activation in activations])
                else:
                    current_loss = self._loss.get_loss(onehot_batch, activations)
                    hypotheses = np.array([self.__labels[np.argmax(sentence)] for activation in activations for sentence in activation]).reshape(activations.shape[:-1])
                loss += current_loss
                bleu_score += np.mean([sentence_bleu([y_batch[idx]], hypotheses[idx]) for idx in range(len(y_batch))])


        loss /= n_batches
        bleu_score /= n_batches
        return loss, bleu_score

    def _forward(self, encoder_inputs, decoder_inputs, train_mode = True, *args, **kwargs):
        if train_mode:
            self._graph.forward(inputs_dict={ENCODER_INPUTS_NAME: encoder_inputs, DECODER_INPUTS_NAME: decoder_inputs},
                                additional_params={'train_mode': train_mode})

            return self._graph.outputs[0]
        else:
            outputs_all = []
            for sequence_index in range(encoder_inputs.shape[0]):
                sequence_encoder_inputs = encoder_inputs[sequence_index][None]
                
                decoder_inputs = np.hstack([np.array([[self.start_of_sequence_token_id]]), np.ones((1, encoder_inputs.shape[1] - 1)) * self.padding_token_id])
                
                self._graph.forward(inputs_dict={ENCODER_INPUTS_NAME: sequence_encoder_inputs, DECODER_INPUTS_NAME: decoder_inputs},
                                    additional_params={'train_mode': train_mode})

                # "freeze" encoder layers so the graph skips them in the forward iteration as they have already done their computations
                for layer in self.encoder_layers:
                    layer.frozen = True

                output_index = 1
                outputs = self._graph.outputs[0]#.reshape(sequence_encoder_inputs.shape + (self._graph.outputs[0].shape[-1],))

                # recompute decoder outputs until it predicts an end-of-sequence token at element output_index
                # each time adding the output at index output_index to the decoder input
                while outputs[0][output_index].argmax() != self.end_of_sequence_token_id and output_index < encoder_inputs.shape[1] - 1:
                    decoder_inputs[0, output_index] = outputs[0][output_index].argmax()

                    # clear messages from decoder layers as they will have to be recomputed
                    for layer in self.decoder_layers:
                        layer.clear_child_edges()

                    self._graph.forward(inputs_dict={DECODER_INPUTS_NAME: decoder_inputs},
                                        additional_params={'train_mode': train_mode})
                    output_index += 1

                outputs_all.append(self._graph.outputs[0])

            # "unfreeze" encoder layers (see reasons for freezing above)
            for layer in self.encoder_layers:
                layer.frozen = False

            self._graph.clear_messages()

            return np.array([out[0] for out in outputs_all])

    def _backward(self, encoder_inputs, decoder_inputs, y_batch, train_mode = True, *args, **kwargs):
        activations = self.decoder_layers[-1].activations
        delta = self._loss.output_deriv(y = activations, t = y_batch)

        self._graph.backward(gradients_dict={GRADIENTS_NAME: delta},
                             additional_params = {ENCODER_INPUTS_NAME: encoder_inputs,
                                                  DECODER_INPUTS_NAME: decoder_inputs,
                                                  'train_mode': train_mode},
                                                  node_func=self._optimizer.update_weights,
                                                  *args,
                                                  **kwargs)

    def _define_graph(self, residual_connections = []):
        self.encoder_layers[0].input_edges[ENCODER_INPUTS_NAME] = None
        self.decoder_layers[0].input_edges[DECODER_INPUTS_NAME] = None
        self.decoder_layers[-1].output_edges[GRADIENTS_NAME] = None

        enco_conn = [(self.encoder_layers[idx], self.encoder_layers[idx + 1], (INPUTS_NAME, GRADIENTS_NAME)) for idx in range(0, len(self.encoder_layers) - 1)]
        deco_conn = [(self.decoder_layers[idx], self.decoder_layers[idx + 1], (INPUTS_NAME, GRADIENTS_NAME)) for idx in range(0, len(self.decoder_layers) - 1)]
        link_conn = [(self.encoder_layers[-1], link, (ENCODER_INPUTS_NAME, GRADIENTS_NAME)) for link in self.link_layers]
        node_connections = enco_conn + deco_conn + link_conn + residual_connections

        self._graph = Graph(node_connections)

