import numpy as np

from .. import activations, initializers
from ..layers.core import Activation, Dense, Dropout, Layer, LayerNormalization
from ..layers.residual import ResidualConnection
from ..layers.embeddings import Embedding
from ..models import EncoderDecoder, Sequential, ENCODER_INPUTS_NAME, ENCODER_GRADIENTS_NAME, DECODER_INPUTS_NAME, DECODER_GRADIENTS_NAME
from ..utils.graph import INPUTS_NAME, GRADIENTS_NAME

class Transformer(EncoderDecoder):
    """ Transformer architecture, as described by Vaswani A. et al. in 'Attention Is All You Need' (link: https://arxiv.org/abs/1608.05859)
        
        Warning: Constructed for the specific use of neural machine translation

        Parameters:
            'num_layers'                    - The number of encoders and decoders to be used.
            'd_model'                       - The size of various internal dimensions within the transformer, such as the embedding sizes
                                              and the sizes of the query, key, value weight matrices of the multihead attention layers.
            'num_heads'                     - Number of attention heads to be used.
            'd_ff'                          - Dimensionality of the inner layer of the positionwise feedforward network modules.
            'start_of_sequence_token_id'    - the id of the start of sequence token used
            'end_of_sequence_token_id'      - the id of the end of sequence token used
            'padding_token_id'              - id of the sequence padding token used
            'input_vocab_size'              - The size of the vocabulary used for the language which is being translated from.
            'target_vocab_size'             - The size of the vocabulary used for the language which is being translated to.
            'dropout_rate'                  - Dropout rate to be used in all the dropout layers.
            'ffn_activation'                - Activation function to be used inbetween the two dense layers of positionwise feedworward networks.
    """
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 start_of_sequence_token_id,
                 end_of_sequence_token_id,
                 padding_token_id,
                 input_vocab_size,
                 target_vocab_size = None,
                 dropout_rate = 0.1,
                 ffn_activation='relu'):
        embeddings_layer = SharedWeightsEmbedding(input_dim=input_vocab_size, output_dim=d_model, weights_scale=np.sqrt(d_model))

        # if target_vocab_size is not provided, the model uses three-way weight tying of the embeddings, as in the Transformer paper
        # so, encoder input, decoder input and output embedding matrices are actually just one, shared matrix
        # otherwise, encoder input embeddings use a separate matrix
        if target_vocab_size is None:
            decoder_embedding_weights = embeddings_layer.shared_weights
            target_vocab_size = input_vocab_size
        else:
            decoder_embedding_weights = None
         
        decoder_embeddings_layer = SharedWeightsEmbedding(input_dim=target_vocab_size,
                                                          output_dim=d_model,
                                                          weights=decoder_embedding_weights,
                                                          weights_scale=np.sqrt(d_model))

        self.__residual_connections = []

        encoder_dropout_1 = Dropout(dropout_rate)

        encoder_layers = [
            embeddings_layer,
            PositionalEncoding(d_model), 
            encoder_dropout_1 # -> res_first.get_residual_in(),
        ]

        decoder_dropout_1 = Dropout(dropout_rate)
        decoder_layers = [
            decoder_embeddings_layer,
            ShiftInput(steps=1, right=True),
            PositionalEncoding(d_model), # -> res_first.get_residual_in(),
            decoder_dropout_1
        ]
        link_layers = []
        for idx in range(num_layers):
            # encoder layers
            encoder_layernorm_1 = LayerNormalization()
            encoder_layernorm_2 = LayerNormalization()

            encoder_layernorm_1._name = 'encoder_layernorm_1'
            encoder_layernorm_2._name = 'encoder_layernorm_2'

            self.__residual_connections += [(encoder_dropout_1, encoder_layernorm_1, (INPUTS_NAME, GRADIENTS_NAME)),
                                            (encoder_layernorm_1, encoder_layernorm_2, (INPUTS_NAME, GRADIENTS_NAME))]

            encoder_layers += [
                MultiHeadAttention(d_model=d_model, num_heads=num_heads),
                Dropout(dropout_rate),
                encoder_layernorm_1, # <- res_first.get_residual_out(), -> res_second.get_residual_in(),

                PositionWiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff, ffn_activation=ffn_activation),
                Dropout(dropout_rate),
                encoder_layernorm_2 # <- res_second.get_residual_out(),
            ]

            # decoder layers
            decoder_layernorm_1 = LayerNormalization()
            decoder_layernorm_2 = LayerNormalization()
            decoder_layernorm_3 = LayerNormalization()

            decoder_layernorm_1._name = 'decoder_layernorm_1'
            decoder_layernorm_2._name = 'decoder_layernorm_2'
            decoder_layernorm_3._name = 'decoder_layernorm_3'

            encoder_link = MultiHeadAttention(d_model=d_model, num_heads=num_heads, encoder_input=True)

            encoder_link._name = 'encoder_link'

            self.__residual_connections += [(decoder_dropout_1, decoder_layernorm_1, (INPUTS_NAME, GRADIENTS_NAME)),
                                            (decoder_layernorm_1, decoder_layernorm_2, (INPUTS_NAME, GRADIENTS_NAME)),
                                            (decoder_layernorm_2, decoder_layernorm_3, (INPUTS_NAME, GRADIENTS_NAME))]
            link_layers.append(encoder_link)

            decoder_layers += [
                MultiHeadAttention(d_model=d_model, num_heads=num_heads, positional_masking=True),
                Dropout(dropout_rate),
                decoder_layernorm_1, # <- res_first.get_residual_out(), -> res_second.get_residual_in(),
                
                encoder_link,
                Dropout(dropout_rate),
                decoder_layernorm_2, # <- res_second.get_residual_out(), -> res_third.get_residual_in(),
                
                PositionWiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff, ffn_activation=ffn_activation),
                Dropout(dropout_rate),
                decoder_layernorm_3 # <- res_third.get_residual_out(),
            ]

        output_dense_layer = SharedWeightsDense(units=target_vocab_size, use_bias=False, weights=decoder_embeddings_layer.shared_weights)
        decoder_layers += [
            output_dense_layer,
            Activation(activations.SEQUENCE_SOFTMAX)
        ]

        super().__init__(encoder_layers,
                         decoder_layers,
                         link_layers,
                         start_of_sequence_token_id = start_of_sequence_token_id,
                         end_of_sequence_token_id = end_of_sequence_token_id,
                         padding_token_id = padding_token_id)

    def _define_graph(self, residual_connections = []):
        residual_connections = self.__residual_connections
        super()._define_graph(residual_connections)
 
        
class MultiHeadAttention(Layer):
    """ Multihead attention layer used in the Transformer architecture.
    """
    def __init__(self, d_model, num_heads, positional_masking=False, encoder_input=False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.positional_masking = positional_masking
        self.encoder_input = encoder_input
        self.trainable = True
        
        self.d_v = d_model // num_heads
        
        # for now, implementation uses the same d_v and d_k, as in original paper
        # might want to change that in the future
        # note that weights properties will have to be modified if d_k != d_v
        self.d_k  = self.d_v
            
        self.__wK = np.random.normal(size=(num_heads, d_model, self.d_k))
        self.__wQ = np.random.normal(size=(num_heads, d_model, self.d_k))
        self.__wV = np.random.normal(size=(num_heads, d_model, self.d_v))
        # self.__wO = np.random.normal(size=(num_heads * self.d_v, d_model))
        self.__wO = np.random.normal(size=(num_heads, d_model, self.d_v))
        self.__softmax = activations.get(activations.SOFTMAX)
        
    @property
    def weights(self): return np.vstack([self.__wQ, self.__wK, self.__wV, self.__wO])
    @weights.setter
    def weights(self, weights):
        self.__wQ = weights[:self.num_heads]
        self.__wK = weights[self.num_heads: 2*self.num_heads]
        self.__wV = weights[2*self.num_heads: 3*self.num_heads]
        self.__wO = weights[3*self.num_heads:]
        
    @property
    def dW(self): return np.vstack([self.__d_wQ, self.__d_wK, self.__d_wV, self.__d_wO])
    
    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        # if multi-head attention is used as part of a decoder in a Transormer architecture
        # without key, value input from an encoder, then key, value input is the same as for queries
        if self.encoder_input:
            if ENCODER_INPUTS_NAME not in kwargs:
                raise NameError(f'Encoder input missing. Expected to find parameter {ENCODER_INPUTS_NAME}')
            self.kv_input = kwargs[ENCODER_INPUTS_NAME]
        else:
            self.kv_input = prev_activations

        self.__Q = np.einsum('ijk,lkm->ijlm', prev_activations, self.__wQ)
        self.__K = np.einsum('ijk,lkm->ijlm', self.kv_input, self.__wK)
        self.__V = np.einsum('ijk,lkm->ijlm', self.kv_input, self.__wV)
        #self.__Q = np.einsum('ij,kjl->ikl', prev_activations, self.__wQ)
        #self.__K = np.einsum('ij,kjl->ikl', self.kv_input, self.__wK)
        #self.__V = np.einsum('ij,kjl->ikl', self.kv_input, self.__wV)
        
        qk = np.einsum('ijkl,ijml->ijkm', self.__Q, self.__K)/(self.d_k ** 4)#/np.sqrt(self.d_k)
        #qk = np.einsum('jkl,jml->jkm', self.__Q, self.__K)/(self.d_k ** 4)#/np.sqrt(self.d_k)

        if self.positional_masking:
            #qk[np.triu_indices_from(qk, k=1)] = np.float('-inf')
            for i in range(qk.shape[-2]):
                qk[..., i, range(i, qk.shape[-1])] = np.float('-inf')

        self.__qk_softmax = self.__softmax.get_activation(qk)

        self.__attention = np.einsum('ijkl,ijlm->ijkm', self.__qk_softmax, self.__V)
        #self.__attention = np.einsum('jkl,jlm->jkm', self.__qk_softmax, self.__V)\
        #                        .reshape((self.__V.shape[0],) + (-1,))

        activations = np.einsum('ijkl,kml->ijm', self.__attention, self.__wO)
        self.activations = activations

    def backward(self, prev_activations, delta, train_mode=True, *args, **kwargs):
        #self.__d_wO = np.einsum('ij,ikl->kjl', delta, self.__attention.reshape((-1, self.num_heads, self.d_v)))
        self.__d_wO = np.einsum('ijk,ijlm->lkm', delta, self.__attention)

        dattention = np.einsum('ijk,lkm->ijlm', delta, self.__wO)
        
        dV = np.einsum('ijkl,ijmk->ijml', dattention, self.__qk_softmax)
        self.__d_wV = np.einsum('ijk,ijlm->lkm', self.kv_input, dV)
        
        dsoftmax = np.einsum('ijkl,ijml->ijkm', dattention, self.__V)
        dsoftmax = self.__softmax.get_delta(self.__qk_softmax.reshape((-1, self.__qk_softmax.shape[-1])),
                                            dsoftmax.reshape((-1, dsoftmax.shape[-1])))\
                                                .reshape(self.__qk_softmax.shape)
    
        dK = np.einsum('ijkl,ijlm->ijkm', dsoftmax, self.__K/(self.d_k ** 4))#/np.sqrt(self.d_k))
        self.__d_wK = np.einsum('ijk,ijlm->lkm', self.kv_input, dK)
        
        dQ = np.einsum('ijkl,ijlm->ijkm', dsoftmax, self.__Q/(self.d_k ** 4))#/np.sqrt(self.d_k))
        self.__d_wQ = np.einsum('ijk,ijlm->lkm', prev_activations, dQ)

        delta_new = np.einsum('ijkl,kml->ijm', dQ, self.__wQ) +\
                    np.einsum('ijkl,kml->ijm', dK, self.__wK) +\
                    np.einsum('ijkl,kml->ijm', dV, self.__wV)
        
        return delta_new

    def output_shape(self, input_shape):
        return input_shape[:1] + (self.d_model,)

class ShiftInput(Layer):
    """ Shift inputs within each sequence. Used in the Transformer to shift decoder inputs one position to the right.
        However, it is designed to be able to shift input both left and right, as well as a set amount of positions ('steps')

        Parameters:
            'steps'            - How many steps/positions the input should be shifted.
            'right'            - Boolean variable. If True, the input is shifted to the right and if False, it is shifter to the left.
            'padding_element'  - Element with which to fill the void input positions after it is shifted. 
    """
    def __init__(self, steps, right, padding_element = 0):
        super(Layer,  self).__init__()
        self.steps = steps
        self.right = right
        self.padding_element = padding_element
        self.trainable = False

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        self.activations = self.__shift_input(prev_activations, forward = True)

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        return self.__shift_input(delta, forward = False)

    def output_shape(self, input_shape): return input_shape

    def __shift_input(self, input, forward):
        shifted = np.ones(input.shape) * self.padding_element

        if (self.right and forward) or (not self.right and not forward):
            #shifted[self.steps:] = input[:-self.steps]
            shifted[:, self.steps:] = input[:, :-self.steps]
        else:
            #shifted[:-self.steps] = input[self.steps:]
            shifted[:, :-self.steps] = input[:, self.steps:]

        return shifted

class SharedWeights:
    def __init__(self, weights = None, weights_shape = None, weights_initializer = None):
        if weights is not None:
            self.__weights = weights
        elif weights_shape is not None and weights_initializer is not None:
            init_func = initializers.get(weights_initializer)
            self.__weights = init_func(weights_shape)
        else:
            raise ValueError('Either "weights" should be passed, or a "weights_shape" together with a "weights_initializer"')

    @property
    def shape(self): return self.__weights.shape
    @property
    def T(self): return self.__weights.T

    @property
    def weights(self): return self.__weights
    @weights.setter
    def weights(self, value): self.__weights = value

class SharedWeightsDense(Dense):
    """ An extension to the regular Dense layer, allowing shared weights with other layers.

        Warning: the 'weights' parameter, if passed, should be an instance of the SharedWeights class (or one that's inheriting it), or a numpy array
    """
    def __init__(self,
                 units,
                 activation = None,
                 use_bias = True,
                 input_shape = None,
                 kernel_reg_l2 = 0.0,
                 weights = None,
                 weights_scale = 1.):
        super().__init__(units=units,
                         activation=activation,
                         use_bias=use_bias,
                         input_shape=input_shape,
                         kernel_reg_l2=kernel_reg_l2)
        if weights is not None:
            if isinstance(weights, (np.ndarray, np.generic)):
                self._weights = SharedWeights(weights=weights)
            elif isinstance(weights, SharedWeights):
                self._weights = weights
            else:
                raise TypeError('Parameter "weights" should either be a numpy array or an instance of SharedWeights.')

        self.weights_scale = weights_scale

    @property
    def weights(self): return self._weights.weights * self.weights_scale
    @weights.setter
    def weights(self, weights): self._weights.weights = weights / self.weights_scale
    @property
    def shared_weights(self): return self._weights
    @property
    def dW(self): return super().dW * self.weights_scale

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        if self.weights is None:
            weight_shape = (self.units, prev_activations.shape[-1])
            self._weights = SharedWeights(weights=np.random.normal(size = weight_shape))

        super().forward(prev_activations, train_mode, *args, **kwargs)

class SharedWeightsEmbedding(Embedding):
    """ An extension to the regular Embedding layer, allowing shared weights with other layers.

        Warning: the 'weights' parameter, if passed, should be an instance of the SharedWeights class (or one that's inheriting it), or a numpy array
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer = 'uniform',
                 input_length = None,
                 weights = None,
                 weights_scale = 1.):
        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         embeddings_initializer=embeddings_initializer,
                         input_length=input_length,
                         weights=weights)

        # parent class would have initialized the weights as a numpy array if "weights" was not passed
        if isinstance(self._embeddings, (np.ndarray, np.generic)):
            self._embeddings = SharedWeights(weights=self._embeddings)
        elif not isinstance(self._embeddings, SharedWeights):
            TypeError('Parameter "weights" should either be a numpy array or an instance of SharedWeights.')

        self.weights_scale = weights_scale

    @property
    def weights(self): return self._embeddings.weights * self.weights_scale
    @weights.setter
    def weights(self, weights): self._embeddings.weights = weights / self.weights_scale
    @property
    def dW(self): return super().dW * self.weights_scale
    @property
    def shared_weights(self): return self._embeddings

class PositionalEncoding(Layer):
    """ Layer which encodes the positions within each sequence of the transofmer input 
    """
    def __init__(self, d_model):
        super(Layer, self).__init__()

        self.d_model = d_model
        self.trainable = False

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        pos_dims = np.array(range(prev_activations.shape[0]))
        emb_dims = np.array(range(prev_activations.shape[1]))
        
        rads = pos_dims[:, np.newaxis] / np.power(10000.0,
                                                  2 * emb_dims[np.newaxis, :] / self.d_model)

        sines = np.sin(rads[:, 0::2])
        cosines = np.cos(rads[:, 1::2])

        pos_encoding = np.zeros(shape=(sines.shape[0], sines.shape[1] + cosines.shape[1]))
        pos_encoding[:, 0::2] = sines
        pos_encoding[:, 1::2] = cosines

        self.activations = prev_activations + pos_encoding[..., np.newaxis]

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        return delta

    def output_shape(self, input_shape): return input_shape


class PositionWiseFeedForwardNetwork(Layer):
    """ Feedforward network used in the Transformer architecture. Consists of two dense layers, with the first having
        an activation function set by parameter 'ffn_activation'. Usually (as in the original paper) ReLu activation.
    """
    def __init__(self, d_model, d_ff, ffn_activation='relu'):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.nn_input = Dense(d_ff, activation=ffn_activation)
        self.nn_output = Dense(d_model)

    @property
    def weights(self): return np.vstack([self.nn_input.weights, self.nn_output.weights.T])
    @weights.setter
    def weights(self, weights):
        self.nn_input.weights = weights[:self.d_model]
        self.nn_output.weights = weights[self.d_model:].T
    @property
    def dW(self): return np.vstack([self.nn_input.dW, self.nn_output.dW.T])

    def forward(self, prev_activations, train_mode = True, *args, **kwargs):
        self.nn_input.forward(prev_activations, train_mode)
        self.nn_output.forward(self.nn_input.activations, train_mode)
        self.activations = self.nn_output.activations

    def backward(self, prev_activations, delta, train_mode = True, *args, **kwargs):
        new_delta = self.nn_output.backward(self.nn_input.activations, delta, train_mode)
        return self.nn_input.backward(prev_activations, new_delta, train_mode)

    def output_shape(self, input_shape):
        return input_shape

