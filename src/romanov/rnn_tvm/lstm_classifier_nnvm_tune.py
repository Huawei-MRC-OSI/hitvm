"""
Based on tune_nnvm_x86.py, applied to lstm_classifier_nnvm.py
"""

from .lstm_classifier_nnvm import *
from .lstm_classifier_nnvm import _batch_size, _var_values

import os
import numpy as np

import nnvm.testing
import nnvm.compiler
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import graph_runtime


#################################################################
# Define network
# --------------
# First we need to define the network in nnvm symbol API.
# Here use the one from lstm_classifier_nnvm.py

def get_network():
    # TODO change LSTM to make batch_size an argument?
    image_side = 28
    num_classes = 10
    input_shape = (_batch_size, image_side, image_side)
    output_shape = (_batch_size, num_classes)

    # TODO deduplication
    X, lstm_out = lstm_and_dense_layer(image_side, image_side, num_classes=num_classes, num_hidden=128)
    y = Variable("y", shape=(_batch_size, num_classes), dtype='float32')

    prediction_op = sym.softmax(lstm_out, name='prediction')

    graph = nnvm.graph.create(prediction_op)  # .apply('CorrectLayout').apply('InferShape').apply('InferType')
    shape_dict = {name: _var_values[name].shape for name in _var_values}
    shape_dict.update(X=(_batch_size, image_side, image_side), y=(_batch_size, num_classes))
    dtype_dict = {name: "float32" for name in shape_dict}
    graph = nnvm.compiler.graph_attr.set_shape_inputs(graph, shape_dict)
    graph = nnvm.compiler.graph_attr.set_dtype_inputs(graph, dtype_dict)
    graph = graph.apply('InferShape').apply('InferType')
    return graph, _var_values, shape_dict, dtype_dict


# Replace "llvm" with the correct target of your cpu.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".

# TODO figure out what it should be,
# see https://stackoverflow.com/questions/15036909/clang-how-to-list-supported-target-architectures to start
target = "llvm"
dtype = "float32"  # same as in LSTM classifier

log_file = "lstm_nnvm.log"

# Set number of threads used for tuning based on the number of
# physical cpu cores on your machine.
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

#################################################################
# Configure tensor tuning settings and create tasks
#
# We will use local mode for tuning configuration. RPC tracker
# mode can be setup similarly to the approach in
# :ref:`tune_nnvm_arm` tutorial.

# You can skip the implementation of this function for this tutorial.
def tune_kernels(tasks,
                 builder=autotvm.LocalBuilder(),
                 runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
                 tuner='ga',
                 early_stopping=None,
                 log_filename=log_file):
    measure_option = autotvm.measure_option(builder, runner)

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        if tsk.workload:
            op_name = tsk.workload[0]
            if op_name == 'conv2d':
                func_create = 'topi_x86_conv2d_NCHWc'
            elif op_name == 'depthwise_conv2d_nchw':
                func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
            else:
                raise ValueError("Tuning {} is not supported on x86".format(op_name))

            task = autotvm.task.create(func_create, args=tsk.args,
                                       target=target, template_key='direct')
            task.workload = tsk.workload
        else:
            task = tsk

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=1000)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        print("n_trial", n_trial)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(**tuning_opt):
    # extract workloads from nnvm graph
    print("Extract tasks...")
    graph, params, shape_dict, dtype_dict = get_network()

    # FIXME wtf, how is inference time here different from lstm_classifier_nnvm???
    # After fixing, apply to commented out code below
    lowered_graph, libmod, params = nnvm.compiler.build(graph=graph, params=_var_values, target='llvm')
    print("Lowered NNVM graph", lowered_graph.ir())  # join_node_attrs doesn't work here
    graph_module = graph_runtime.create(lowered_graph, libmod, tvm.cpu(0))
    Xt = tvm.nd.array((np.random.uniform(size=shape_dict['X'])).astype(dtype))
    graph_module.set_input(X=Xt, **params)

    eval_time = True

    if eval_time:
        ftimer = graph_module.module.time_evaluator("run", tvm.cpu(0), number=1, repeat=1)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(f"Mean inference time (std dev): {np.mean(prof_res):.2f} ms ({np.std(prof_res):.2f} ms)")

    # print("Initial NNVM graph", graph.ir(join_node_attrs=['shape', 'dtype']))
    # # TODO determine important tasks to extract
    # tasks = autotvm.task.extract_from_graph(graph, target=target,
    #                                         shape=shape_dict, dtype=dtype_dict,
    #                                         symbols=(nnvm.sym.dense,))
    # print("Extracted tasks", tasks)
    #
    # # run tuning tasks
    # print("Tuning...")
    # tune_kernels(tasks, **tuning_opt)
    #
    # # compile kernels with history best records
    # with autotvm.apply_history_best(log_file):
    #     print("Compile...")
    #     with nnvm.compiler.build_config(): #opt_level=3):
    #         lowered_graph, lib, params = nnvm.compiler.build(
    #             graph, target=target, shape=shape_dict, params=params, dtype=dtype_dict)
    #         print("Lowered NNVM graph", lowered_graph.ir())
    #
    #     # upload parameters to device
    #     ctx = tvm.cpu(0)
    #     input_name = 'X'
    #     data_tvm = tvm.nd.array((np.random.uniform(size=shape_dict[input_name])).astype(dtype))
    #     module = graph_runtime.create(lowered_graph, lib, ctx)
    #     module.set_input(input_name, data_tvm)
    #     module.set_input(**params)
    #
    #     # evaluate
    #     print("Evaluate inference time cost...")
    #     # TODO number=1, repeat=1 is temporary because tuning badly fails and slows down NNVM at the moment
    #     ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=1)  # number=2, repeat=3)
    #     prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    #     print(f"Mean inference time (std dev): {np.mean(prof_res):.2f} ms ({np.std(prof_res):.2f} ms)")


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

if __name__ == '__main__':
    tune_and_evaluate()

######################################################################
# Sample Output
# -------------
# The tuning needs to compile many programs and extract feature from them.
# So a high performance CPU is recommended.
# One sample output is listed below.
#
# .. code-block:: bash
#
#    Extract tasks...
#    Tuning...
#    [Task  1/12]  Current/Best:  598.05/2497.63 GFLOPS | Progress: (252/252) | 1357.95 s Done.
#    [Task  2/12]  Current/Best:  522.63/2279.24 GFLOPS | Progress: (784/784) | 3989.60 s Done.
#    [Task  3/12]  Current/Best:  447.33/1927.69 GFLOPS | Progress: (784/784) | 3869.14 s Done.
#    [Task  4/12]  Current/Best:  481.11/1912.34 GFLOPS | Progress: (672/672) | 3274.25 s Done.
#    [Task  5/12]  Current/Best:  414.09/1598.45 GFLOPS | Progress: (672/672) | 2720.78 s Done.
#    [Task  6/12]  Current/Best:  508.96/2273.20 GFLOPS | Progress: (768/768) | 3718.75 s Done.
#    [Task  7/12]  Current/Best:  469.14/1955.79 GFLOPS | Progress: (576/576) | 2665.67 s Done.
#    [Task  8/12]  Current/Best:  230.91/1658.97 GFLOPS | Progress: (576/576) | 2435.01 s Done.
#    [Task  9/12]  Current/Best:  487.75/2295.19 GFLOPS | Progress: (648/648) | 3009.95 s Done.
#    [Task 10/12]  Current/Best:  182.33/1734.45 GFLOPS | Progress: (360/360) | 1755.06 s Done.
#    [Task 11/12]  Current/Best:  372.18/1745.15 GFLOPS | Progress: (360/360) | 1684.50 s Done.
#    [Task 12/12]  Current/Best:  215.34/2271.11 GFLOPS | Progress: (400/400) | 2128.74 s Done.
#    Compile...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 3.16 ms (0.03 ms)
