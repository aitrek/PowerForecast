"""
@Project   : PowerForecast
@Module    : mlp_model.py
@Author    : HjwGivenLyy [1752929469@qq.com]
@Created   : 1/25/19 10:58 AM
@Desc      : MLP algorithm
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.tsatools import lagmat

from base import FILE_PATH
from base import evaluation_criteria_model

FILE_PATH = FILE_PATH + "daily_data.csv"

np.random.seed(5)


class NetworkConstantVariable:
    """常量值类"""
    # Data split (values represent percentage)
    train_ratio = 0.7
    val_ratio = 0.14
    test_ratio = 0.16

    # Parameters
    learning_rate = 10 ** -3
    min_step_size_train = 10 ** -5
    training_epochs = 1000
    display_step = 100

    # Network Parameters
    n_input = 10
    n_classes = 1
    # n_hidden = (n_input + n_classes) / 2
    n_hidden = 50


def load_data():
    """get date from file library"""
    data = pd.read_csv(FILE_PATH)
    rtn = data.values.astype('float64')
    return rtn


def get_mlp_sets(ncv, train_feature, train_label):
    """Splits data into three subsets"""

    train_index = int(len(train_feature) * ncv.train_ratio)
    val_index = int(len(train_feature) * ncv.val_ratio) + train_index

    train_x = train_feature[:train_index, :]
    train_y = train_label[:train_index]

    val_x = train_feature[train_index:val_index, :]
    val_y = train_label[train_index:val_index]

    test_x = train_feature[val_index:, :]
    test_y = train_label[val_index:]

    return train_x, train_y, val_x, val_y, test_x, test_y


def get_multilayer_perceptron(ncv, x):
    """Store layers weight & bias"""

    # Create model
    weights = {
        'h1': tf.Variable(tf.random_normal([ncv.n_input, ncv.n_hidden],
                                           dtype=tf.float64)),
        'out': tf.Variable(tf.random_normal([ncv.n_hidden, ncv.n_classes],
                                            dtype=tf.float64))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([ncv.n_hidden], dtype=tf.float64)),
        'out': tf.Variable(tf.random_normal([ncv.n_classes], dtype=tf.float64))
    }

    # Hidden layer with relu activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with tanh activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

    return out_layer


def run_mlp(ncv, inp, outp):

    train_x, train_y, val_x, val_y, test_x, test_y = get_mlp_sets(ncv, inp, outp)

    # tf Graph input
    x = tf.placeholder("float64", [None, ncv.n_input])
    y = tf.placeholder("float64", [None, ncv.n_classes])

    # Construct model
    predict_value = get_multilayer_perceptron(ncv, x)

    # Define loss and optimizer
    cost = tf.nn.l2_loss(tf.subtract(predict_value, y))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=ncv.learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        last_cost = ncv.min_step_size_train + 1
        for epoch in range(ncv.training_epochs):

            # Training data
            for i in range(len(train_x)):
                batch_x = np.reshape(train_x[i, :], (1, ncv.n_input))
                batch_y = np.reshape(train_y[i], (1, ncv.n_classes))

                # Run optimization
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            # Calculating data error
            c = 0.0
            for i in range(len(val_x)):
                batch_x = np.reshape(val_x[i, :], (1, ncv.n_input))
                batch_y = np.reshape(val_y[i], (1, ncv.n_classes))

                # Run Cost function
                c += sess.run(cost, feed_dict={x: batch_x, y: batch_y})

            c /= len(val_x)
            # Display logs per epoch step
            if epoch % ncv.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=",
                      "{:.30f}".format(c))

            if abs(c - last_cost) < ncv.min_step_size_train:
                break

            last_cost = c

        nn_predictions = np.array([])
        for i in range(len(test_x)):
            batch_x = np.reshape(test_x[i, :], (1, ncv.n_input))
            nn_predictions = np.append(nn_predictions, sess.run(
                predict_value, feed_dict={x: batch_x})[0])

        print("Optimization Finished!")

    nn_predictions.flatten()

    return [test_y, nn_predictions]


if __name__ == "__main__":

    ncv1 = NetworkConstantVariable()

    inp = lagmat(load_data(), ncv1.n_input, trim='both')
    inp = inp[1:]
    print(inp)
    outp = inp[1:, 0]
    print(outp.shape)
    inp = inp[:-1]
    print(inp.shape)

    real_value, predicted_value = run_mlp(ncv1, inp, outp)
    print(real_value, predicted_value)
    mape = evaluation_criteria_model(real_value, predicted_value)
    print("mape = ", mape)
