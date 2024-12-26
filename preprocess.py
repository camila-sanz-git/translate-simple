import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
import numpy as np

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class PreProcess():
  def __init__(self, data, english_column, spanish_column, fix_seq_size):
    self.data = data
    self.english_column = english_column
    self.spanish_column = spanish_column
    self.fix_seq_size = fix_seq_size

  def remove_punctuation(self, text):
    return text.translate(str.maketrans('', '', string.punctuation + "¿¡"))

  def clean_data(self):
    self.data[self.english_column] = self.data[self.english_column].apply(self.remove_punctuation).str.lower()
    self.data[self.spanish_column] = self.data[self.spanish_column].apply(self.remove_punctuation).str.lower()
    self.data[self.spanish_column] = self.data[self.spanish_column].apply(lambda x: f'<start> {x} <end>')
    return self.data

  def split_data(self):
    df = self.clean_data()
    X_train, X_test, y_train, y_test = train_test_split(df[self.english_column], df[self.spanish_column], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

  def data_tokenization(self, df_train, df_test):
    #X_train, X_test, Y_train, Y_test = split_data() --> X_train = eng_words, Y_train = spanish_words (translate)
    #Apply for df_train and df_test separatly

    text_vectorizer = TextVectorization(max_tokens=None, standardize=None, split='whitespace', pad_to_max_tokens=False, vocabulary=None)
    text_vectorizer.adapt(df_train.values)
    vectorize_train = text_vectorizer(df_train.values)
    vectorize_test = text_vectorizer(df_test.values) # Las que no conoce las pone con UNK

    #adjust size:
    size_diff = self.fix_seq_size - vectorize_train.shape[1]
    zeros = tf.zeros((tf.shape(vectorize_train)[0], size_diff), dtype=tf.int64)
    vectorize_train = tf.concat([vectorize_train, zeros], axis=1)

    size_diff = self.fix_seq_size - vectorize_test.shape[1]
    zeros = tf.zeros((tf.shape(vectorize_test)[0], size_diff), dtype=tf.int64)
    vectorize_test = tf.concat([vectorize_test, zeros], axis=1)

    #extract vocab
    vocab_list = text_vectorizer.get_vocabulary()
    vocab = {i:vocab_list[i] for i in np.arange(1, len(vocab_list))}
    inverse_vocab = {vocab_list[i]:i for i in np.arange(1, len(vocab_list))}

    return vectorize_train, vectorize_test, vocab, inverse_vocab