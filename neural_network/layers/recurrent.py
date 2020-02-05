import numpy as np
from abc import ABCMeta, abstractmethod

from .. import activations
from .core import Layer
from neural_network.utils import TIMESTEP_AXIS

class RNN(Layer):
    """Base class for recurrent layers.
    """
    def __init__(self,
                 units,
                 activation = 'tanh',
                 use_bias = True,
                 dropout = 0.,
                 recurrent_dropout = 0.,
                 kernel_initializer = 'glorot_uniform',
                 recurrent_initializer = 'orthogonal',
                 bias_initializer = 'zeros',
                 return_state = False,
                 return_sequences = False,
                 stateful = False):
        super().__init__(activation = activation,
                         use_bias = use_bias)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.stateful = stateful
        self._built = False

    @property
    def weights(self): return np.vstack([self._kernel, self._recurrent_kernel])
    @weights.setter
    def weights(self, weights):
        kernel_height = self._kernel.shape[0]
        self._kernel = weights[:kernel_height]
        self._recurrent_kernel = weights[kernel_height:]
    @property
    def dW(self): return np.vstack([self._dkernel, self._drecurrent_kernel])

    def forward(self, prev_activations, train_mode = True):
        state0 = np.zeros((prev_activations.shape[0], self.units)) 
        if self.stateful:
            state_len = min(len(prev_activations), len(self._states[-1]))
            state0[:state_len] = self._states[-1][:state_len]
        self._states = [state0]
        self._states_prefunc = []

    def output_shape(self, input_shape):
        return input_shape + (self.units,)

    def _build(self, input_shape, matrix_count = 1):
        self._kernel = self._add_weight(shape = (input_shape[-1], self.units * matrix_count), initializer = self.kernel_initializer)
        self._recurrent_kernel = self._add_weight(shape = (self.units, self.units * matrix_count), initializer = self.recurrent_initializer)
        if self.use_bias:
            self.bias = self._add_weight(shape = (1, self.units * matrix_count), initializer = self.bias_initializer)
        else:
            self.bias = self._add_weight(shape = (1, self.units * matrix_count), initializer = 'zeros')

        if self.dropout < 0 or\
           self.dropout >= 1 or\
           self.recurrent_dropout < 0 or\
           self.recurrent_dropout >= 1:
            raise ValueError('Dropout must be in the range [0, 1).')

        self._states = [np.zeros((input_shape[0], self.units))]
        self._built = True

    def _set_dropout(self, train_mode):
        if train_mode:
            self._kernel_drop_mask = (np.random.rand(self._kernel.shape[0], self._kernel.shape[1]) >= self.dropout) / (1 - self.dropout)
            kernel = self._kernel * self._kernel_drop_mask
            self._recurrent_kernel_drop_mask = (np.random.rand(self._recurrent_kernel.shape[0], self._recurrent_kernel.shape[1]) >= self.recurrent_dropout) / (1 - self.recurrent_dropout)
            recurrent_kernel = self._recurrent_kernel * self._recurrent_kernel_drop_mask

            return kernel, recurrent_kernel
        
        return self._kernel, self._recurrent_kernel

class SimpleRNN(RNN):
    """Fully-connected RNN where the output is to be fed back to input.
    """
    def __init__(self,
                 units,
                 activation = 'tanh',
                 use_bias = True,
                 dropout = 0.,
                 recurrent_dropout = 0.,
                 kernel_initializer = 'glorot_uniform',
                 recurrent_initializer = 'orthogonal',
                 bias_initializer = 'zeros',
                 return_state = False,
                 return_sequences = False,
                 stateful = False):
        super().__init__(units,
                         activation,
                         use_bias,
                         dropout,
                         recurrent_dropout,
                         kernel_initializer,
                         recurrent_initializer,
                         bias_initializer,
                         return_state, 
                         return_sequences,
                         stateful)
        
    def forward(self, prev_activations, train_mode = True):
        if not self._built:
            self._build(input_shape = prev_activations.shape, matrix_count = 1)
        super().forward(prev_activations, train_mode)

        kernel, recurrent_kernel = self._set_dropout(train_mode)

        for x_t in np.swapaxes(prev_activations, TIMESTEP_AXIS, 0):
            X_t = x_t.dot(kernel) + self.bias
            self._states_prefunc.append(self._states[-1].dot(recurrent_kernel) + X_t)
            self._states.append(self._activation.get_activation(self._states_prefunc[-1]))

        output_sequences = np.moveaxis(np.array(self._states[1:]), 0, TIMESTEP_AXIS) if self.return_sequences else self._states[-1]
        self.activations = [output_sequences, self._states[-1]] if self.return_state else output_sequences

    def backward(self, prev_activations, delta, train_mode = True):
        self._dkernel = np.zeros(self._kernel.shape)
        self._drecurrent_kernel = np.zeros(self._recurrent_kernel.shape)
        if self.use_bias:
            self._dB = np.zeros(self.bias.shape)
        
        delta_new = np.zeros(prev_activations.swapaxes(0, TIMESTEP_AXIS).shape)
        deltas = np.moveaxis(delta, TIMESTEP_AXIS, 0) if self.return_sequences else [delta]
        
        for t in np.arange(len(deltas))[::-1]:
            delta_t = deltas[t]

            X = np.moveaxis(prev_activations, TIMESTEP_AXIS, 0)
            if len(deltas) > 1:
                X = X[:t+1]

            for s_t in np.arange(len(X))[::-1]:
                x_t = X[s_t]
                last_state = self._states_prefunc[s_t]
                dactivation = self._activation.get_delta(last_state, delta_t)

                self._dkernel += x_t.T.dot(dactivation)
                self._drecurrent_kernel += self._states[s_t - 1].T.dot(dactivation)
                if self.use_bias:
                    self._dB += np.mean(dactivation, axis = 0).reshape(1, -1)

                delta_t = dactivation.dot(self._recurrent_kernel.T)
                delta_new[s_t] += dactivation.dot(self._kernel.T)

        if train_mode:
            self._dkernel *= self._kernel_drop_mask
            self._drecurrent_kernel *= self._recurrent_kernel_drop_mask
        
        return delta_new.swapaxes(0, TIMESTEP_AXIS)

class GRU(RNN):
    """Gated Recurrent Unit - Cho et al. 2014.
    """
    def __init__(self,
                 units,
                 activation = 'tanh',
                 recurrent_activation = 'hard_sigmoid',
                 use_bias = True,
                 dropout = 0.,
                 recurrent_dropout = 0.,
                 kernel_initializer = 'glorot_uniform',
                 recurrent_initializer = 'orthogonal',
                 bias_initializer = 'zeros',
                 return_state = False,
                 return_sequences = False,
                 stateful = False):
        super().__init__(units = units,
                         activation = activation,
                         use_bias = use_bias,
                         dropout = dropout,
                         recurrent_dropout = recurrent_dropout,
                         kernel_initializer = kernel_initializer,
                         recurrent_initializer = recurrent_initializer,
                         bias_initializer = bias_initializer,
                         return_state = return_state, 
                         return_sequences = return_sequences,
                         stateful = stateful)
        self.recurrent_activation = recurrent_activation
        self._recurrent_activation = activations.get(recurrent_activation)

    @property
    def W(self): return self._kernel[:, :self.units]
    @property
    def W_r(self): return self._kernel[:, self.units: self.units * 2] 
    @property
    def W_z(self): return self._kernel[:, self.units * 2:]
    @property
    def U(self): return self._recurrent_kernel[:, :self.units]
    @property
    def U_r(self): return self._recurrent_kernel[:, self.units: self.units * 2]
    @property
    def U_z(self): return self._recurrent_kernel[:, self.units * 2:]
    @property
    def b(self): return self.bias[:, :self.units] if self.use_bias else 0
    @property
    def b_r(self): return self.bias[:, self.units: self.units * 2] if self.use_bias else 0
    @property
    def b_z(self): return self.bias[:, self.units * 2:] if self.use_bias else 0


    def forward(self, prev_activations, train_mode = True):
        if not self._built:
            self._build(input_shape = prev_activations.shape, matrix_count = 3)
        super().forward(prev_activations, train_mode)

        self.__Z, self.__Z_prefunc, self.__R, self.__R_prefunc, self.__H_tilde = [[] for i in range(5)]

        W_all, U_all = self._set_dropout(train_mode)
        W, W_r, W_z = W_all[:, :self.units], W_all[:, self.units: self.units * 2], W_all[:, self.units * 2:]
        U, U_r, U_z = U_all[:, :self.units], U_all[:, self.units: self.units * 2], U_all[:, self.units * 2:]

        for x_t in np.swapaxes(prev_activations, TIMESTEP_AXIS, 0):
            z_t_prefunc = self._states[-1].dot(U_z) + x_t.dot(W_z) + self.b_z
            z_t = self._recurrent_activation.get_activation(z_t_prefunc)
            r_t_prefunc = self._states[-1].dot(U_r) + x_t.dot(W_r) + self.b_r
            r_t = self._recurrent_activation.get_activation(r_t_prefunc)
            self._states_prefunc.append((self._states[-1] * r_t).dot(U) + x_t.dot(W) + self.b)
            h_tilde = self._activation.get_activation(self._states_prefunc[-1])
            self._states.append(z_t * h_tilde + (1 - z_t) * self._states[-1])

            self.__Z.append(z_t)
            self.__Z_prefunc.append(z_t_prefunc)
            self.__R.append(r_t)
            self.__R_prefunc.append(r_t_prefunc)
            self.__H_tilde.append(h_tilde)

        output_sequences = np.moveaxis(np.array(self._states[1:]), 0, TIMESTEP_AXIS) if self.return_sequences else self._states[-1]
        self.activations = [output_sequences, self._states[-1]] if self.return_state else output_sequences

    def backward(self, prev_activations, delta, train_mode = True):
        self._dkernel = np.zeros(self._kernel.shape)
        self._drecurrent_kernel = np.zeros(self._recurrent_kernel.shape)
        if self.use_bias:
            self._dB = np.zeros(self.bias.shape)

        delta_new = np.zeros(prev_activations.swapaxes(0, TIMESTEP_AXIS).shape)
        deltas = np.moveaxis(delta, TIMESTEP_AXIS, 0) if self.return_sequences else [delta]

        for t in np.arange(len(deltas))[::-1]:
            delta_t = deltas[t]

            X = np.moveaxis(prev_activations, TIMESTEP_AXIS, 0)
            if len(deltas) > 1:
                X = X[:t+1]

            for s_t in np.arange(len(X))[::-1]:
                x_t = X[s_t]
                z_t = self.__Z[s_t]
                z_t_prefunc = self.__Z_prefunc[s_t]
                r_t = self.__R[s_t]
                r_t_prefunc = self.__R_prefunc[s_t]
                h_tm1 = self._states[s_t - 1]
                h_tilde = self.__H_tilde[s_t]
               
                activ_deriv = self._activation.get_derivative(self._states_prefunc[s_t])
                activ_deriv_r = self._recurrent_activation.get_derivative(r_t_prefunc)
                activ_deriv_z = self._recurrent_activation.get_derivative(z_t_prefunc)

                d = delta_t * z_t * activ_deriv
                dW = x_t.T.dot(d)
                dU = h_tm1.T.dot(d)

                d_r = (d * h_tm1 * activ_deriv_r).dot(self.U.T)
                dW_r = x_t.T.dot(d_r)
                dU_r = h_tm1.T.dot(d_r)

                d_z =  delta_t * (h_tilde - h_tm1) * activ_deriv_z
                dW_z = x_t.T.dot(d_z)
                dU_z = h_tm1.T.dot(d_z)

                self._dkernel += np.hstack([dW, dW_r, dW_z])
                self._drecurrent_kernel += np.hstack([dU, dU_r, dU_z])
                if self.use_bias:
                    db = np.mean(d, axis = 0).reshape(1, -1)
                    db_r = np.mean(d_r, axis = 0).reshape(1, -1)
                    db_z = np.mean(d_z, axis = 0).reshape(1, -1)
                    self._dB += np.hstack([db, db_r, db_z])

                delta_t = delta_t * (1 - z_t) +\
                          d_z.dot(self.U_z.T) +\
                          d_r.dot(self.U_r.T) +\
                          (delta_t * z_t * activ_deriv * r_t).dot(self.U.T)

                delta_new[s_t] += d.dot(self.W.T) + d_r.dot(self.W_r.T) + d_z.dot(self.W_z.T)

        if train_mode:
            self._dkernel *= self._kernel_drop_mask
            self._drecurrent_kernel *= self._recurrent_kernel_drop_mask
        
        return delta_new.swapaxes(0, TIMESTEP_AXIS)

class LSTM(RNN):
    """Long Short-Term Memory layer - Hochreiter 1997.
    """
    def __init__(self,
                 units,
                 activation = 'tanh',
                 recurrent_activation = 'hard_sigmoid',
                 use_bias = True,
                 dropout = 0.,
                 recurrent_dropout = 0.,
                 kernel_initializer = 'glorot_uniform',
                 recurrent_initializer = 'orthogonal',
                 bias_initializer = 'zeros',
                 unit_forget_bias = True,
                 return_state = False,
                 return_sequences = False,
                 stateful = False):
        super().__init__(units = units,
                         activation = activation,
                         use_bias = use_bias,
                         dropout = dropout,
                         recurrent_dropout = recurrent_dropout,
                         kernel_initializer = kernel_initializer,
                         recurrent_initializer = recurrent_initializer,
                         bias_initializer = bias_initializer,
                         return_state = return_state, 
                         return_sequences = return_sequences,
                         stateful = stateful)
        self.recurrent_activation = recurrent_activation
        self._recurrent_activation = activations.get(recurrent_activation)
        self.unit_forget_bias = unit_forget_bias

    @property
    def W_f(self): return self._kernel[:, :self.units]
    @property
    def U_f(self): return self._recurrent_kernel[:, :self.units]
    @property
    def b_f(self): return self.bias[:, :self.units] if self.use_bias else 0
    @property
    def W_i(self): return self._kernel[:, self.units:self.units*2]
    @property
    def U_i(self): return self._recurrent_kernel[:, self.units:self.units*2]
    @property
    def b_i(self): return self.bias[:, self.units:self.units*2] if self.use_bias else 0
    @property
    def W_c(self): return self._kernel[:, self.units*2: self.units*3]
    @property
    def U_c(self): return self._recurrent_kernel[:, self.units*2: self.units*3]
    @property
    def b_c(self): return self.bias[:, self.units*2: self.units*3] if self.use_bias else 0
    @property
    def W_o(self): return self._kernel[:, self.units*3:]
    @property
    def U_o(self): return self._recurrent_kernel[:, self.units*3:]
    @property
    def b_o(self): return self.bias[:, self.units*3:] if self.use_bias else 0

    def forward(self, prev_activations, train_mode = True):
        if not self._built:
            self._build(input_shape = prev_activations.shape)

        state0 = np.zeros((prev_activations.shape[0], self.units))
        cell_state0 = np.zeros((prev_activations.shape[0], self.units))
        if self.stateful:
            state_len = min(len(prev_activations), len(self._states[-1]))
            state0[:state_len] = self._states[-1][:state_len]
            cell_state0[:state_len] = self.__cell_states[-1][:state_len]
        self._states = [state0]
        self.__cell_states = [cell_state0]

        self.__F, self.__F_prefunc, self.__I, self.__I_prefunc,\
            self.__O, self.__O_prefunc, self.__C_tilde, self.__C_tilde_prefunc = [[] for i in range(8)]

        W_all, U_all = self._set_dropout(train_mode)
        W_f, W_i, W_c, W_o = W_all[:, :self.units], W_all[:, self.units:self.units*2],  W_all[:, self.units*2: self.units*3], W_all[:, self.units*3:]
        U_f, U_i, U_c, U_o = U_all[:, :self.units], U_all[:, self.units:self.units*2],  U_all[:, self.units*2: self.units*3], U_all[:, self.units*3:]

        for x_t in np.swapaxes(prev_activations, TIMESTEP_AXIS, 0):
            # forget gate
            self.__F_prefunc.append(self._states[-1].dot(U_f) + x_t.dot(W_f) + self.b_f)
            self.__F.append(self._recurrent_activation.get_activation(self.__F_prefunc[-1]))

            # input gate
            self.__I_prefunc.append(self._states[-1].dot(U_i) + x_t.dot(W_i) + self.b_i)
            self.__I.append(self._recurrent_activation.get_activation(self.__I_prefunc[-1]))

            # cell state candidate values
            self.__C_tilde_prefunc.append(self._states[-1].dot(U_c) + x_t.dot(W_c) + self.b_c)
            self.__C_tilde.append(self._activation.get_activation(self.__C_tilde_prefunc[-1]))

            # new cell state
            self.__cell_states.append(self.__F[-1] * self.__cell_states[-1] + self.__I[-1] * self.__C_tilde[-1])

            # output gate
            self.__O_prefunc.append(self._states[-1].dot(U_o) + x_t.dot(W_o) + self.b_o)
            self.__O.append(self._recurrent_activation.get_activation(self.__O_prefunc[-1]))

            # new hidden state
            self._states.append(self.__O[-1] * self._activation.get_activation(self.__cell_states[-1]))

        output_sequences = np.moveaxis(np.array(self._states[1:]), 0, TIMESTEP_AXIS) if self.return_sequences else self._states[-1]
        self.activations = [output_sequences, self._states[-1], self.__cell_states[-1]] if self.return_state else output_sequences

    def backward(self, prev_activations, delta, train_mode = True):
        self._dkernel = np.zeros(self._kernel.shape)
        self._drecurrent_kernel = np.zeros(self._recurrent_kernel.shape)
        if self.use_bias:
            self._dB = np.zeros(self.bias.shape)

        delta_new = np.zeros(prev_activations.swapaxes(0, TIMESTEP_AXIS).shape)
        deltas = np.moveaxis(delta, TIMESTEP_AXIS, 0) if self.return_sequences else [delta]

        for t in np.arange(len(deltas))[::-1]:
            delta_t = deltas[t]

            X = np.moveaxis(prev_activations, TIMESTEP_AXIS, 0)
            if len(deltas) > 1:
                X = X[:t+1]

            for s_t in np.arange(len(X))[::-1]:
                x_t = X[s_t]
                f_t, f_t_prefunc = self.__F[s_t], self.__F_prefunc[s_t]
                i_t, i_t_prefunc = self.__I[s_t], self.__I_prefunc[s_t]
                o_t, o_t_prefunc  = self.__O[s_t], self.__O_prefunc[s_t]
                C_tilde, C_tilde_prefunc = self.__C_tilde[-1], self.__C_tilde_prefunc[-1]
                C_t = self.__cell_states[s_t]
                C_tm1, h_tm1 = self.__cell_states[s_t - 1], self._states[-1]
                dC_t = o_t * self._activation.get_delta(C_t, delta_t)

                d_o = self._activation.get_delta(C_t, delta_t) * self._recurrent_activation.get_derivative(o_t_prefunc)
                dW_o = x_t.T.dot(d_o)
                dU_o = h_tm1.T.dot(d_o)

                d_f = dC_t * C_tm1 * self._recurrent_activation.get_derivative(f_t_prefunc)
                dW_f = x_t.T.dot(d_f)
                dU_f = h_tm1.T.dot(d_f)

                d_i = dC_t * C_tilde * self._recurrent_activation.get_derivative(i_t_prefunc)
                dW_i = x_t.T.dot(d_i)
                dU_i = h_tm1.T.dot(d_i)

                d_c = dC_t * i_t * self._activation.get_derivative(C_tilde_prefunc)
                dW_c = x_t.T.dot(d_c)
                dU_c = h_tm1.T.dot(d_c)

                self._dkernel += np.hstack([dW_f, dW_i, dW_c, dW_o])
                self._drecurrent_kernel += np.hstack([dU_f, dU_i, dU_c, dU_o])
                if self.use_bias:
                    db_f = np.mean(d_f, axis = 0).reshape(1, -1)
                    db_i = np.mean(d_i, axis = 0).reshape(1, -1)
                    db_c = np.mean(d_c, axis = 0).reshape(1, -1)
                    db_o = np.mean(d_o, axis = 0).reshape(1, -1)
                    self._dB += np.hstack([db_f, db_i, db_c, db_o])

                delta_t = d_o.dot(self.U_o.T) + d_f.dot(self.U_f.T) + d_i.dot(self.U_i.T) + d_c.dot(self.U_c.T)
                delta_new[s_t] += d_o.dot(self.W_o.T) + d_f.dot(self.W_f.T) + d_i.dot(self.W_i.T) + d_c.dot(self.W_c.T)

        if train_mode:
            self._dkernel *= self._kernel_drop_mask
            self._drecurrent_kernel *= self._recurrent_kernel_drop_mask
        
        return delta_new.swapaxes(0, TIMESTEP_AXIS)

    def _build(self, input_shape):
        self._kernel = self._add_weight(shape = (input_shape[-1], self.units * 4), initializer = self.kernel_initializer)
        self._recurrent_kernel = self._add_weight(shape = (self.units, self.units * 4), initializer = self.recurrent_initializer)
        if self.use_bias:
            if self.unit_forget_bias:
                self.bias = np.hstack([self._add_weight(shape = (1, self.units), initializer = 'ones'),
                                       self._add_weight(shape = (1, self.units), initializer = self.bias_initializer),
                                       self._add_weight(shape = (1, self.units * 2), initializer = self.bias_initializer)])
            else:
                self.bias = self._add_weight(shape = (1, self.units * 4), initializer = self.bias_initializer)

        if self.dropout < 0 or\
           self.dropout >= 1 or\
           self.recurrent_dropout < 0 or\
           self.recurrent_dropout >= 1:
            raise ValueError('Dropout must be in the range [0, 1).')

        self._states = [np.zeros((input_shape[0], self.units))]
        self.__cell_states = [np.zeros((input_shape[0], self.units))]
        self._built = True
