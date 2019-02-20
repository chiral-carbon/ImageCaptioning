import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

def time_distributed_dense(x, w, b=None, dropout=None, input_dim=None, output_dim=None, timesteps=None):
       	# Apply y.w + b for every temporal slice y of x.
        print(x.shape)
        print(w.shape)
        if not input_dim:
            input_dim = K.shape(x)[2]
        if not timesteps:
            timesteps = K.shape(x)[1]
        if not output_dim:
            output_dim = K.shape(w)[1]

        print(output_dim)
        print(timesteps)
        print(input_dim)

        if dropout:
            ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
            dropout_matrix = K.dropout(ones, dropout)
            expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
            x *= expanded_dropout_matrix

        x = K.reshape(x, (-1, input_dim))

        x = K.dot(x, w)
        if b:
            x = x + b
        x = K.reshape(x, (-1, timesteps, output_dim))
        return x

def init_feature_matrix(self, feature_name):
        return self.add_weight(shape=(self.units,), name=feature_name, initializer=self.kernel_initializer, 
        			regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)

class AttentionDecoder(Recurrent):

    def __init__(self, units, output_dim, activation='tanh', return_probabilities=False, name='AttentionDecoder', kernel_initializer='glorot_uniform', 
                 recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        """
        units: dimension of the hidden state and the attention matrices
        output_dim: the number of labels in the output space

        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

    def build(self, input_shape):

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s
       
        #  Matrices for creating the context vector
        self.V_a = init_feature_matrix(self, 'V_a')
        self.U_a = init_feature_matrix(self, 'U_a')
        self.W_a = init_feature_matrix(self, 'W_a')
        self.b_a = init_feature_matrix(self, 'b_a')
        
        # Matrices for the r (reset) gate
        self.C_r = init_feature_matrix(self, 'C_r')
        self.U_r = init_feature_matrix(self, 'U_r')
        self.W_r = init_feature_matrix(self, 'W_r')
        self.b_r = init_feature_matrix(self, 'b_r')
        
        # Matrices for the z (update) gate
        self.C_z = init_feature_matrix(self, 'C_z')
        self.U_z = init_feature_matrix(self, 'U_z')
        self.W_z = init_feature_matrix(self, 'W_z')
        self.b_z = init_feature_matrix(self, 'b_z')
        
        # Matrices for the proposal
        self.C_p = init_feature_matrix(self, 'C_p')
        self.U_p = init_feature_matrix(self, 'U_p')
        self.W_p = init_feature_matrix(self, 'W_p')
        self.b_p = init_feature_matrix(self, 'b_p')
       
        # Matrices for making the final prediction vector
        self.C_o = init_feature_matrix(self, 'C_o')
        self.U_o = init_feature_matrix(self, 'U_o')
        self.W_o = init_feature_matrix(self, 'W_o')
        self.b_o = init_feature_matrix(self, 'b_o')

        # For creating the initial state
        self.W_s = init_feature_matrix(self, 'W_s')
       
        self.input_spec = [InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        self.x_seq = x
        self._uxpb = time_distributed_dense(self.x_seq, self.U_a, b=self.b_a, input_dim=self.input_dim, timesteps=self.timesteps, output_dim=self.units)

        return super(AttentionDecoder, self).call(x)

    def get_initial_state(self, inputs):
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]   # states

    def step(self, x, states):

        ytm, stm = states
        _stm = K.repeat(stm, self.timesteps)
        _Wxstm = K.dot(_stm, self.W_a)
        et = K.dot(activations.tanh(_Wxstm + self._uxpb), K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at = at/at_sum_repeated  # vector of size (batchsize, timesteps, 1)
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        rt = activations.sigmoid(K.dot(ytm, self.W_r) + K.dot(stm, self.U_r)+ K.dot(context, self.C_r) + self.b_r)
        zt = activations.sigmoid(K.dot(ytm, self.W_z) + K.dot(stm, self.U_z) + K.dot(context, self.C_z) + self.b_z)
        s_tp = activations.tanh(K.dot(ytm, self.W_p)+ K.dot((rt * stm), self.U_p) + K.dot(context, self.C_p) + self.b_p)
        st = (1-zt)*stm + zt * s_tp
        yt = activations.softmax(K.dot(ytm, self.W_o) + K.dot(stm, self.U_o) + K.dot(context, self.C_o) + self.b_o)

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))