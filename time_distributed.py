from keras import backend as K

def time_distributed_dense(x, w, b=None, dropout=None, input_dim=None, output_dim=None, timesteps=None):
        '''Apply y.w + b for every temporal slice y of x.
        '''
        if not input_dim:
            input_dim = K.shape(x)[2]
        if not timesteps:
            timesteps = K.shape(x)[1]
        if not output_dim:
            output_dim = K.shape(w)[1]

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