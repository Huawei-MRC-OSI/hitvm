"""
Poor man's LSTM cell applied to MNIST.
"""

import math

import nnvm
import numpy as np
import tvm
from keras.datasets import mnist
from nnvm import sym
from nnvm.compiler.graph_util import gradients
from nnvm.compiler.optimizer import SGD
from tvm.contrib import graph_runtime

from .lstm_classifier_tf import read_vars

_load_model = True
_var_values = read_vars() if _load_model else {}
_global_vars = {}

def Variable(name, **kwargs):
    if name in _var_values:
        kwargs["init"] = _var_values[name]
    var = sym.Variable(name, **kwargs)
    _global_vars[name] = var
    return var

# TODO currently working with fixed example number, 0 should mean "any number" but is not working yet
# (division by 0)
_batch_size = 10000

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
    X = Variable("X", shape=(_batch_size, num_timesteps, num_inputs), dtype='float32')

    U_shape = (num_inputs + num_units, num_units)
    b_shape = (1, num_units)
    Ug = Variable("Ug", init=init(U_shape))
    bg = Variable("bg", init=bias_init(shape=b_shape))

    Ui = Variable("Ui", init=init(U_shape))
    bi = Variable("bi", init=bias_init(shape=b_shape))

    Uf = Variable("Uf", init=init(U_shape))
    bf = Variable("bf", init=bias_init(shape=b_shape) + sym.ones(shape=b_shape))

    Uo = Variable("Uo", init=init(U_shape))
    bo = Variable("bo", init=bias_init(shape=b_shape))

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
    xs = [sym.squeeze(x, axis=1) for x in xs]

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
        # x = sym.squeeze(x, axis=1)
        s, h = cell(x, s, h)
        outputs.append(h)

    return X, outputs


def lstm_and_dense_layer(num_timesteps: int, num_inputs: int, num_classes: int, num_hidden: int,
                         dense_init=np.random.normal):
    """
    Use `model` with `num_hidden` LSTM units, translate them `num_classes` classes using a dense layer.
    """
    W = Variable("W", init=dense_init(size=[num_hidden, num_classes]))
    b = Variable("b", init=dense_init(size=[1, num_classes]))

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


# split into separate train and infer functions?
def run(Xl, yl, Xt, yt):
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

    X, lstm_out = lstm_and_dense_layer(num_timesteps, num_inputs, num_classes=num_classes, num_hidden=num_hidden)
    y = Variable("y", shape=(_batch_size, num_classes), dtype='float32')

    prediction_op = sym.softmax(lstm_out, name='prediction')
    correct_pred = sym.broadcast_equal(sym.argmax(prediction_op, axis=1), sym.argmax(y, axis=1))
    accuracy_op = sym.mean(sym.cast(correct_pred, dtype='float32'))

    if _load_model:
        # inference only

        # TODO for some reason immediate InferShape doesn't work here.
        # Applying CorrectLayout or not gives different errors.
        graph = nnvm.graph.create(accuracy_op) # .apply('CorrectLayout').apply('InferShape').apply('InferType')
        shape_dict = {name: _var_values[name].shape for name in _var_values}
        shape_dict.update(X=(_batch_size, num_timesteps, num_inputs), y=(_batch_size, num_classes))
        dtype_dict = {name: "float32" for name in shape_dict}
        graph = nnvm.compiler.graph_attr.set_shape_inputs(graph, shape_dict)
        graph = nnvm.compiler.graph_attr.set_dtype_inputs(graph, dtype_dict)
        graph = graph.apply('InferShape').apply('InferType')
        print("Initial NNVM graph", graph.ir(join_node_attrs=['shape','dtype']))

        lowered_graph, libmod, params = nnvm.compiler.build(graph=graph, params=_var_values, target='llvm')
        # print("Lowered NNVM graph", lowered_graph.ir()) # join_node_attrs doesn't work here
        graph_module = graph_runtime.create(lowered_graph, libmod, tvm.cpu(0))
        graph_module.set_input(**params)
        graph_module.run(X=Xt, y=yt)
        accuracy_value = graph_module.get_output(0)
        print("Testing Accuracy:", accuracy_value)


    else:
        learning_rate = 0.001
        training_steps = 500  # TODO temporary to make faster, 10000 to actually train
        num_examples = Xl.shape[0]

        optimizer = SGD(learning_rate=learning_rate)
        loss_op = sym.mean(softmax_cross_entropy_with_logits(logits=lstm_out, labels=y))
        # Doesn't work currently, mean and concatenate don't have registered gradients in NNVM
        train_op = optimizer.minimize(loss_op)

        # no point unless we fix train_op

        # with tf.Session(graph=tf.Graph()) as sess:
        #
        #     sess.run(variables.global_variables_initializer())
        #
        #     epoch = -1
        #     batch_start = 0
        #     batch_end = batch_size
        #
        #     def next_batch():
        #         nonlocal epoch, batch_start, batch_end, Xl, yl
        #         if batch_end > num_examples or epoch == -1:
        #             epoch += 1
        #             batch_start = 0
        #             batch_end = batch_size
        #             perm0 = np.arange(num_examples)
        #             np.random.shuffle(perm0)
        #             Xl = Xl[perm0]
        #             yl = yl[perm0]
        #         Xi_ = Xl[batch_start:batch_end, :, :]
        #         yi_ = yl[batch_start:batch_end, :]
        #         batch_start = batch_end
        #         batch_end = batch_start + batch_size
        #         return {X: Xi_, y: yi_}
        #
        #     for step in range(training_steps + 1):
        #         batch = next_batch()
        #         sess.run(train_op, feed_dict=batch)
        #
        #         if step % 100 == 0:
        #             loss_, acc_ = sess.run((loss_op, accuracy_op), feed_dict=batch)
        #             print("epoch", epoch, "step", step, "loss", "{:.4f}".format(loss_), "acc",
        #                   "{:.2f}".format(acc_))
        #
        #     print("Optimization Finished!")
        #     print("Testing Accuracy:",
        #         sess.run(accuracy_op, feed_dict={X: Xt, y: yt}))
        #
        #     save_model = False
        #     if save_model:
        #         print("Saving the model")
        #         simple_save(sess, export_dir='./lstm', inputs={"images":X}, outputs={"out":prediction_op})


if __name__ == '__main__':
    (Xl, yl), (Xt, yt) = mnist_load()
    run(Xl, yl, Xt, yt)
