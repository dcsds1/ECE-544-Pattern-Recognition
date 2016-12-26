# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:05:50 2016

@author: changsongdong
"""

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 32
display_step = 10

n_input = 1
n_steps = 49
n_hidden = 100
n_classes = 10

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

def RNN(x, weights, biases):

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)

    rnn_model = rnn_cell.BasicRNNCell(n_hidden)

    outputs, states = rnn.rnn(rnn_model, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights) + biases

pred = RNN(x, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

tf.scalar_summary('accuracy', accuracy)
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.train.SummaryWriter('./RNN_pixel', sess.graph)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, 28, 28))[:, ::4, ::4].reshape((batch_size, n_steps, n_input))

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cross_entropy, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + \
                  ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
            summary_str = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(summary_str, step)
        step += 1

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, 28, 28))[:, ::4, ::4].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label})
