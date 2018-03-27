import os
import tensorflow as tf
import numpy as np

class GRU4Rec:

    def __init__(self, sess, is_training=True, layers=1, rnn_size=100, n_epochs=3,
                 batch_size=200, learning_rate=0.001, n_items=10000):
        self.sess = sess
        self.is_training = is_training
        self.layers = layers
        self.rnn_size = rnn_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_items = n_items
        self.item_key = 'ItemId'
        self.session_key = 'Session_Id'
        self.time = 'Time'

    #####################ACTIVATION FUNCTIONS####################
    def linear(self, X):
        return X

    def tanh(self, X):
        return tf.nn.tanh(X)

    def softmax(self, X):
        return tf.nn.softmax(X)

    def relu(self, X):
        return tf.nn.relu(X)

    ########################LOSS FUNCTION########################
    def cross_entronpy(self, yhat):
        print(tf.diag_part(yhat))
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat)))

    def build_model(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in
                      range(self.layers)]

        initializer = tf.random_normal_initializer(mean=0, stddev=0.95)
        embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
        softmax_W = tf.get_variable('softmax_W', [self.n_items, self.rnn_size], initializer=initializer)
        softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

        cell = tf.contrib.rnn.GRUCell(self.rnn_size, activation=self.relu)
        stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * self.layers)

        inputs = tf.nn.embedding_lookup(embedding, self.X)
        output, state = stacked_cell(inputs, tuple(self.state))
        self.final_state = state

        sample_W = tf.nn.embedding_lookup(softmax_W, self.Y)
        print(sample_W)
        sample_b = tf.nn.embedding_lookup(softmax_b, self.Y)

        logits = tf.matmul(output, sample_W, transpose_b=True) + sample_b
        self.yhat = self.softmax(logits)
        self.cost = self.cross_entronpy(self.yhat)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def fit(self, data):
        itemids = data[self.item_key].unique()
        item_map = np.arange(len(itemids))



GRU4Rec(tf.Session()).build_model()
