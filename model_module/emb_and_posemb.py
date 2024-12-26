import numpy as np
import tensorflow as tf

def positional_encoding_func(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)

class EmbeddingAndPosition(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_seq_length):
      super(EmbeddingAndPosition, self).__init__()
      self.vocab_size = vocab_size
      self.d_model = d_model #emb_output_dim
      self.max_seq_length = max_seq_length
      self.embedding = Embedding(self.vocab_size + 1, self.d_model, embeddings_initializer="uniform", mask_zero=True)
      self.positional_encoder = positional_encoding_func(self.max_seq_length, self.d_model)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask  # Retornar máscara existente
        else:
            return tf.cast(tf.not_equal(inputs, 0), tf.int32)

    def call(self, input, mask=None):

      x = self.embedding(input)
      x = x + self.positional_encoder[tf.newaxis, :self.max_seq_length, :]
      mask = self.compute_mask(input)
      x *= tf.cast(mask[:, :, tf.newaxis], dtype=tf.float32)# Not sure is this is needed

      return x, mask