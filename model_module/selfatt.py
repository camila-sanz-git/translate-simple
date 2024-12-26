import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, dk, dv):
    super().__init__()
    self.d_model = d_model #same as initial embedding
    self.dk = dk #free new parameter
    self.dv = dv #free new parameter

    #query:
    self.Wq = self.add_weight(shape=(self.d_model, self.dk), initializer= "glorot_uniform", trainable=True, name='query')
    self.bq = self.add_weight(shape=(self.dk,), initializer= "glorot_uniform", trainable=True, name='query_bias')

    #key:
    self.Wk = self.add_weight(shape=(self.d_model, self.dk), initializer= "glorot_uniform", trainable=True, name='key')
    self.bk = self.add_weight(shape=(self.dk,), initializer= "glorot_uniform", trainable=True, name='key_bias')

    #value
    self.Wv = self.add_weight(shape=(self.d_model, self.dv), initializer= "glorot_uniform", trainable=True, name='value')
    self.bv = self.add_weight(shape=(self.dv,), initializer= "glorot_uniform", trainable=True, name='value_bias')

  def call(self, inputs, mask=None):
    query_matrix = tf.matmul(inputs, self.Wq) + self.bq #matmul makes matrix multiplication
    key_matrix = tf.matmul(inputs, self.Wk) + self.bk
    value_matrix = tf.matmul(inputs, self.Wv) + self.bv

    scores = tf.matmul(query_matrix, key_matrix, transpose_b=True)/tf.math.sqrt(tf.cast(dk, tf.float32))
    mask = tf.cast(mask, dtype=scores.dtype)
    mask_expanded = tf.expand_dims(mask, axis=1)
    mask_bidimensional = mask_expanded * tf.transpose(mask_expanded, perm=[0, 2, 1])  # (batch_size, sequence_size, sequence_size)

    scores = tf.where(mask_bidimensional == 0, -1e9, scores)

    attention_scores = tf.nn.softmax(scores) #softmax por fila

    mask_bidimensional = tf.cast(mask_bidimensional, dtype=attention_scores.dtype)
    attention_scores = tf.where(mask_bidimensional == 0, tf.cast(0, tf.float32), attention_scores)

    attention_output = tf.matmul(attention_scores, value_matrix)

    return attention_scores, attention_output