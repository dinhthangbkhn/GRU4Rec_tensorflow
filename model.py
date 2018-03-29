import os
import tensorflow as tf
import numpy as np
import pandas as pd


class GRU4Rec:

    def __init__(self, sess, is_training=True, layers=2, rnn_size=100, n_epochs=3,
                 batch_size=200, learning_rate=0.001, n_items = -1):
        self.sess = sess
        self.is_training = is_training
        self.layers = layers
        self.rnn_size = rnn_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.item_key = 'ItemId'
        self.session_key = 'SessionId'
        self.time = 'Time'
        self.epoch_finished = False
        self.n_items = n_items
        self.build_model()
        self.sess.run(tf.global_variables_initializer())

    #####################ACTIVATION FUNCTIONS####################
    def linear(self, X):
        return X

    def tanh(self, X):
        return tf.nn.tanh(X)

    def softmax(self, X):
        return tf.nn.softmax(X)

    def relu(self, X):
        return tf.nn.relu(X)

    ########################LOSS FUNCTIONS########################
    def cross_entronpy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat)))

    ########################BUILD MODEL###########################
    def build_model(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in
                      range(self.layers)]

        initializer = tf.random_uniform_initializer(minval=-0.95, maxval=0.95)
        embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
        softmax_W = tf.get_variable('softmax_W', [self.n_items, self.rnn_size], initializer=initializer)
        softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(1.0))

        cell = tf.contrib.rnn.GRUCell(self.rnn_size, activation=self.relu)
        stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * self.layers)

        inputs = tf.nn.embedding_lookup(embedding, self.X)
        output, state = stacked_cell(inputs, tuple(self.state))
        # print(state)
        self.final_state = state # layers * batch_size * rnn_size

        sample_W = tf.nn.embedding_lookup(softmax_W, self.Y)
        sample_b = tf.nn.embedding_lookup(softmax_b, self.Y)

        logits = tf.matmul(output, sample_W, transpose_b=True) + sample_b
        self.yhat = self.softmax(logits)
        self.cost = self.cross_entronpy(self.yhat)

        # if not self.is_training:
        #     print('not training')
        #     return

        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)


    def create_offset_session(self, data):
        offset_session = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        offset_session[1:] = data.groupby(self.session_key).size().cumsum()
        return offset_session

    def fit(self, data):
        print('fitting data')
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': itemidmap[itemids].values}),
                        on=self.item_key, how='inner')
        data = data.sort_values([self.session_key, self.item_key], ascending=True)
        offset_session = self.create_offset_session(
            data)  # offset_session is an array of number that is the order of beginning row of session in data

        print('begin training')
        for epoch in range(self.n_epochs):
            epoch_cost = []
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
            iters = np.arange(self.batch_size) #array that contain what you want to become input
            maxiter = iters.max()
            session_idx_array = np.arange(len(offset_session) - 1) #array of session idx
            start = offset_session[session_idx_array[iters]]  # input row in frame data of a minibatch
            end = offset_session[session_idx_array[iters + 1]]  # output row in frame data of a minibatch
            self.epoch_finished = False

            ######Training an epoch#########
            while not self.epoch_finished:
                out_idx = data['ItemIdx'].values[start] #itemidx of output item in batch
                minlen= (end-start).min() #the minimum length of sessions

                for i in range(minlen-1):
                    in_idx = out_idx
                    out_idx = data['ItemIdx'].values[start+i+1]
                    # print(in_idx)
                    # print(out_idx)
                    fetches = [self.cost, self.final_state, self.train_op, self.yhat]
                    feed_dict = {self.X: in_idx, self.Y: out_idx}
                    for j in range(self.layers):
                        feed_dict[self.state[j]] = state[j]

                    cost, state, _, yhat = self.sess.run(fetches, feed_dict)
                    # print(yhat)
                    epoch_cost.append(cost)

                start = start+minlen-1
                mask = np.arange(len(iters))[(end-start)<=1] #array of session that end up
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_session)-1:
                        self.epoch_finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_session[session_idx_array[maxiter]]
                    end[idx] = offset_session[session_idx_array[maxiter]+1]
                if len(mask):
                    for i in range(self.layers):
                        state[i][mask] = 0
            #######Epoch end#########
            print('Epoch {} - Cost: {}'.format(epoch,np.mean(epoch_cost)))
    def predict(self):





data = pd.read_csv('./rsc15_test.txt',sep='\t', dtype={'ItemId': np.int64})
n_items = len(data['ItemId'].unique())

model = GRU4Rec(tf.Session(), n_epochs=10, learning_rate=0.1, batch_size=10, n_items = n_items)
model.fit(data)
# data = pd.read_csv('./rsc15_test.txt', sep='\t', dtype={'ItemId': np.int64}))