"""
Poor man's LSTM cell applied to MNIST. Use `train` function to train the model.
"""

import tvm
import numpy as np
import nnvm
from nnvm import sym
from nnvm.compiler.graph_util import gradients
from nnvm.compiler.optimizer import SGD

import math

from keras.datasets import mnist

def lstm_gate(op, U, b, x):
    """
    op - nonlinearity operation
    x - input tensor of shape (1,a)
    U - weight matrix of shape (a,b)
    b - bias (1,b)

    return tensor of shape (1,b)
    """
    return op(sym.matmul(x, U) + b)


def glorot_uniform(shape):
    assert len(shape) == 2
    fan_in = shape[0]
    fan_out = shape[1]
    limit = math.sqrt(fan_in + fan_out)
    return np.random.uniform(-limit, limit, shape)


def lstm_layer(num_timesteps: int, num_inputs: int, num_units: int,
               init=glorot_uniform, bias_init=sym.zeros):
    """
    Create a single cell and replicate it `num_timesteps` times for training.
    Return X,[(batch_size,num_classes) x num_timesteps]
    """
    X = sym.Variable("X", shape=(-1, num_timesteps, num_inputs), dtype='float32')

    U_shape = (num_inputs + num_units, num_units)
    b_shape = (1, num_units)
    Ug = sym.Variable("Ug", init=init(U_shape))
    bg = sym.Variable("bg", init=bias_init(shape=b_shape))

    Ui = sym.Variable("Ui", init=init(U_shape))
    bi = sym.Variable("bi", init=bias_init(shape=b_shape))

    Uf = sym.Variable("Uf", init=init(U_shape))
    bf = sym.Variable("bf", init=bias_init(shape=b_shape) + sym.ones(shape=b_shape))

    Uo = sym.Variable("Uo", init=init(U_shape))
    bo = sym.Variable("bo", init=bias_init(shape=b_shape))

    def cell(x_t, s_t, h_t):
        xh_t = sym.concatenate(x_t, h_t, axis=1)
        g = lstm_gate(sym.tanh, Ug, bg, xh_t)
        i = lstm_gate(sym.sigmoid, Ui, bi, xh_t)
        f = lstm_gate(sym.sigmoid, Uf, bf, xh_t)
        o = lstm_gate(sym.sigmoid, Uo, bo, xh_t)

        s_t1 = s_t * f + g * i
        h_t1 = sym.tanh(s_t1) * o
        return (s_t1, h_t1)

    xs = sym.split(X, indices_or_sections=num_timesteps, axis=1)

    # in TF:
    # batch_size = sym.shape(X)[0]
    # s_shape = sym.stack([batch_size, num_units], name="s_shape")
    #
    # s = sym.zeros(s_shape, dtype=np.float32)
    if num_units > num_inputs:
        s_like = sym.pad(xs[0], pad_width=((0, 0), (0, num_units-num_inputs)))
    else:
        s_like = xs[0][:, 0:num_units] # TODO untested
    s = sym.zeros_like(s_like)

    h = s
    outputs = []
    for x in xs:
        x = sym.squeeze(x, axis=1)
        s, h = cell(x, s, h)
        outputs.append(h)

    return X, outputs


def lstm_and_dense_layer(num_timesteps: int, num_inputs: int, num_classes: int, num_hidden: int,
                         dense_init=np.random.normal):
    """
    Use `model` with `num_hidden` LSTM units, translate them `num_classes` classes using a dense layer.
    """
    W = sym.Variable("W", init=dense_init(size=[num_hidden, num_classes]))
    b = sym.Variable("b", init=dense_init(size=[1, num_classes]))

    X, outputs = lstm_layer(num_timesteps, num_inputs, num_units=num_hidden)
    cls = sym.matmul(outputs[-1], W) + b
    return X, cls


def mnist_load():
    """ Load MNIST data and convert its ys to one-hot encoding """
    (Xl, yl), (Xt, yt) = mnist.load_data()

    def oh(y):
        yoh = np.zeros((y.shape[0], 10), dtype=np.float32)
        yoh[np.arange(y.shape[0]), y] = 1
        return yoh

    def as_float(X):
        return X.astype(np.float32) / 255.0

    return (as_float(Xl), oh(yl)), (as_float(Xt), oh(yt))


def _flatten_outer_dims(tensor):
    """Flattens tensor's outer dimensions and keeps its last dimension."""
    return tensor # we only pass 2-dim tensors here, I think # TODO check


def softmax_cross_entropy_with_logits(logits, labels):
    """ Ported from tf.nn.softmax_cross_entropy_with_logits_v2"""
    # # labels and logits must be of the same type
    # labels = math_ops.cast(labels, logits.dtype)
    # input_rank = array_ops.rank(logits)
    # # For shape inference.
    # shape = logits.get_shape()
    #
    # input_shape = array_ops.shape(logits)

    # Make precise_logits and labels into matrices.
    logits = _flatten_outer_dims(logits)
    labels = _flatten_outer_dims(labels)

    # # Do the actual op computation.
    # # The second output tensor contains the gradients.  We use it in
    # # _CrossEntropyGrad() in nn_grad but not here.
    # cost, unused_backprop = gen_nn_ops.softmax_cross_entropy_with_logits(
    #     logits, labels)
    cost = -sym.sum(sym.log_softmax(logits)*labels, axis=-1) # up to constant

    # # The output cost shape should be the input minus dim.
    # output_shape = array_ops.slice(input_shape, [0],
    #                                [math_ops.subtract(input_rank, 1)])
    # cost = sym.reshape(cost, output_shape)

    # Make shape inference work since reshape and transpose may erase its static
    # shape.
    # if not context.executing_eagerly(
    # ) and shape is not None and shape.dims is not None:
    #     shape = shape.as_list()
    #     del shape[dim]
    #     cost.set_shape(shape)

    return cost


def train(Xl, yl, Xt, yt):
    """ Main train
    :param Xl training examples tensor, shape [num_examples, num_inputs, num_timesteps]
    :param yl one-hot encoded training labels tensor, shape [num_examples, num_classes]
    :param Xt test examples tensor, shape [num_examples, num_inputs, num_timesteps]
    :param yt one-hot encoded test labels tensor, shape [num_examples, num_classes]
    """
    batch_size = 64
    num_timesteps = Xl.shape[2] # number of rows (each row in the image is considered as a timestep)
    num_inputs = Xl.shape[1] # length of each row
    num_hidden = 128
    num_classes = yl.shape[1]
    training_steps = 500 # TODO temporary to make faster, 10000 to actually train
    learning_rate = 0.001
    num_examples = Xl.shape[0]

    X, lstm_out = lstm_and_dense_layer(num_timesteps, num_inputs, num_classes=num_classes, num_hidden=num_hidden)
    y = sym.Variable("y", shape=(-1, num_classes), dtype='float32')

    loss_op = sym.mean(softmax_cross_entropy_with_logits(logits=lstm_out, labels=y))
    optimizer = SGD(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    prediction = sym.softmax(lstm_out, name='prediction')
    correct_pred = sym.equal(sym.argmax(prediction, 1), sym.argmax(y, 1))
    accuracy_op = sym.reduce_mean(sym.cast(correct_pred, np.float32))

    with tf.Session(graph=tf.Graph()) as sess:

        sess.run(variables.global_variables_initializer())

        epoch = -1
        batch_start = 0
        batch_end = batch_size

        def next_batch():
            nonlocal epoch, batch_start, batch_end, Xl, yl
            if batch_end > num_examples or epoch == -1:
                epoch += 1
                batch_start = 0
                batch_end = batch_size
                perm0 = np.arange(num_examples)
                np.random.shuffle(perm0)
                Xl = Xl[perm0]
                yl = yl[perm0]
            Xi_ = Xl[batch_start:batch_end, :, :]
            yi_ = yl[batch_start:batch_end, :]
            batch_start = batch_end
            batch_end = batch_start + batch_size
            return {X: Xi_, y: yi_}

        for step in range(training_steps + 1):
            batch = next_batch()
            sess.run(train_op, feed_dict=batch)

            if step % 100 == 0:
                loss_, acc_ = sess.run((loss_op, accuracy_op), feed_dict=batch)
                print("epoch", epoch, "step", step, "loss", "{:.4f}".format(loss_), "acc",
                      "{:.2f}".format(acc_))

        print("Optimization Finished!")
        print("Testing Accuracy:",
            sess.run(accuracy_op, feed_dict={X: Xt, y: yt}))

        save_model = False
        if save_model:
            print("Saving the model")
            simple_save(sess, export_dir='./lstm', inputs={"images":X}, outputs={"out":prediction})


if __name__ == '__main__':
    (Xl, yl), (Xt, yt) = mnist_load()
    train(Xl, yl, Xt, yt)
