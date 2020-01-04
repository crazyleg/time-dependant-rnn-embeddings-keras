import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class TimeEmbedding(Layer):
    def __init__(self, hidden_embedding_size, output_dim, **kwargs):
        super(TimeEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_embedding_size = hidden_embedding_size


    def build(self, input_shape):
        self.emb_weights = self.add_weight(name='weights', shape=(self.hidden_embedding_size,), initializer='uniform',
                                           trainable=True)
        self.emb_biases = self.add_weight(name='biases', shape=(self.hidden_embedding_size,), initializer='uniform',
                                          trainable=True)
        self.emb_final = self.add_weight(name='embedding_matrix', shape=(self.hidden_embedding_size, self.output_dim),
                                         initializer='uniform', trainable=True)


    def call(self, x):
        x = tf.keras.backend.expand_dims(x)
        x = tf.keras.activations.softmax(x * self.emb_weights + self.emb_biases)
        x = tf.einsum('bsv,vi->bsi', x, self.emb_final)
        return x


    def get_config(self):
        config = super(TimeEmbedding, self).get_config()
        config.update({'time_dims': self.output_dim, 'hidden_embedding_size': self.hidden_embedding_size})
        return config

