import tensorflow as tf
from time_embedding_layer import TimeEmbedding


class ModelTest(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ModelTest, self).__init__(**kwargs)
        self.time_emb = TimeEmbedding(20,64)

    def call(self, input1):
        emb = self.time_emb(input1)
        return emb


def model_generator2():
    input1 = tf.keras.layers.Input(shape=(1))

    result = ModelTest()(input1)
    model = tf.keras.models.Model(input1, result)
    return model

model = model_generator2()
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(1e-4))
model.summary()

results = model(tf.random.uniform([16,20]))
assert results.shape == [16,20,64]