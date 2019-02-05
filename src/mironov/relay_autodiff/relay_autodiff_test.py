import tvm
import topi
import numpy as np

from exercise.runners import run_tvm

all_tests = {}

def add_testcase(name,f):
  global all_tests
  all_tests[name] = f
  return ()

def run_tests():
  for name,f in all_tests.items():
    try:
      f(name)
    except tvm.TVMError as e:
      print(name, "FAILED", e)

def _test_diff_1arg(f, ishape=(4,5), *args, **kwargs):
  def _run(name):
    a = tvm.placeholder(ishape,name="A",dtype=tvm.float32)
    try:
      t = f(a, *args, **kwargs)
    except Exception as e:
      print(name, "SKIPPED")
      return
    dt = tvm.differentiate(t,[a])
    res = run_tvm(0,1, { a:1*np.ones(ishape).astype(np.float32) }, dt[0])
    print(name, "SUCCEEDED")
    return res
  return _run

def _test_diff_2arg(f, ishapeA=(4,5), ishapeB=(4,5), *args, **kwargs):
  def _run(name):
    a = tvm.placeholder(ishapeA,name="a",dtype=tvm.float32)
    b = tvm.placeholder(ishapeB,name="b",dtype=tvm.float32)
    t=None; dt=None
    try:
      t = f(a,b, *args, **kwargs)
    except Exception as e:
      print(name, "SKIPPED")
      return
    dt = tvm.differentiate(t,[a,b])
    res = run_tvm(0,1, {
        a:1.4*np.ones(ishapeA).astype(np.float32)
      , b:1.8*np.ones(ishapeB).astype(np.float32)
      }, dt[0])
    print(name, "SUCCEEDED")
    return res
  return _run


# # Level 1
# * [ ]  tvm.relay.log
add_testcase("log", _test_diff_1arg(topi.log))

# * [X]  tvm.relay.sqrt
add_testcase("sqrt", _test_diff_1arg(topi.sqrt))

# * [ ]  tvm.relay.exp
add_testcase("exp", _test_diff_1arg(topi.exp))

# * [ ]  tvm.relay.sigmoid
add_testcase("sigmoid", _test_diff_1arg(topi.sigmoid))

# * [ ]  tvm.relay.add
add_testcase("add", _test_diff_2arg(lambda a,b: a+b))

# * [ ]  tvm.relay.subtract
add_testcase("substract", _test_diff_2arg(lambda a,b: a-b))
# * [ ]  tvm.relay.multiply
add_testcase("multiply", _test_diff_2arg(lambda a,b: a*b))
# * [ ]  tvm.relay.divide
add_testcase("divide", _test_diff_2arg(lambda a,b: a/b))
# * [ ]  tvm.relay.mod
add_testcase("mod", _test_diff_2arg(lambda a,b: a%b))
# * [ ]  tvm.relay.tanh
add_testcase("tanh", _test_diff_1arg(topi.tanh))
# * [?]  tvm.relay.concatenate
# * [ ]  tvm.relay.expand_dims
add_testcase("expand_dims", _test_diff_1arg(topi.expand_dims, axis=1))
# * [ ]  tvm.relay.nn.softmax
add_testcase("softmax", _test_diff_1arg(topi.nn.softmax))
# * [ ]  tvm.relay.nn.log_softmax
add_testcase("log_softmax", _test_diff_1arg(topi.nn.log_softmax))
# * [ ]  tvm.relay.nn.relu
add_testcase("relu", _test_diff_1arg(topi.nn.relu))
# * [?]  tvm.relay.nn.dropout
# * [?]  tvm.relay.nn.batch_norm
# * [?]  tvm.relay.nn.bias_add

# # Level 2
# * [?]  tvm.relay.nn.conv2d
# * [?]  tvm.relay.nn.conv2d_transpose
# * [?]  tvm.relay.nn.dense
# * [ ]  tvm.relay.nn.max_pool2d
## TODO: add_testcase("max_pool", _test_diff_1arg(topi.nn.pool))
##       Pool2DCompute
# * [ ]  tvm.relay.nn.avg_pool2d
## TODO: Pool2DCompute
# * [ ]  tvm.relay.nn.global_max_pool2d
## TODO: GlobalPool2DCompute
# * [ ]  tvm.relay.nn.global_avg_pool2d
## TODO: GlobalPool2DCompute
# * [ ]  tvm.relay.nn.upsampling
## TODO: topi::nn::upsampling
# * [ ]  tvm.relay.nn.batch_flatten
## TODO: topi::nn::flatten
# * [ ]  tvm.relay.nn.pad
## TODO: topi::pad
# * [ ]  tvm.relay.nn.lrn
## TODO: topi::lrn
# * [?]  tvm.relay.nn.l2_normalize
# * [?]  tvm.relay.nn.contrib_conv2d_winograd_without_weight_transform
# * [?]  tvm.relay.nn.contrib_conv2d_winograd_weight_transform

# # Level 3
# * [ ]  tvm.relay.nn.leaky_relu
# * [X]  tvm.relay.nn.prelu
add_testcase("prelu", _test_diff_2arg(topi.nn.prelu, ishapeA=(3,4,5,6), ishapeB=(1)) )
# [17:14:43]
# /home/mironov/proj/hitvm/src/mironov/tvm/include/tvm/packed_func_ext.h:121:
# Check failed: type_code_ == kNodeHandle (0 vs. 8)  expected NodeHandle but get
# int

# * [ ]  tvm.relay.reshape
# * [ ]   tvm.relay.reshape_like
# * [ ]  tvm.relay.copy
# * [ ]  tvm.relay.transpose
# * [ ]  tvm.relay.squeeze
# * [ ]  tvm.relay.floor
# * [ ]  tvm.relay.ceil
# * [ ]  tvm.relay.trunc
# * [ ]  tvm.relay.clip
# * [ ]  tvm.relay.round
# * [ ]  tvm.relay.abs
# * [ ]  tvm.relay.negative
# * [ ]  tvm.relay.take
# * [ ]  tvm.relay.zeros
# * [ ]  tvm.relay.zeros_like
# * [ ]  tvm.relay.ones
# * [ ]  tvm.relay.ones_like
# * [ ]  tvm.relay.full
# * [ ]  tvm.relay.full_like
# * [ ]  tvm.relay.cast
# * [ ]  tvm.relay.split

# # Level 4
# * [ ]  tvm.relay.right_shift
# * [ ]  tvm.relay.left_shift
# * [ ]  tvm.relay.equal
# * [ ]  tvm.relay.not_equal
# * [ ]  tvm.relay.greater
# * [ ]  tvm.relay.greater_equal
# * [ ]  tvm.relay.less
# * [ ]  tvm.relay.less_equal
# * [ ]  tvm.relay.maximum
# * [ ]  tvm.relay.minimum
# * [ ]  tvm.relay.power
# * [ ]  tvm.relay.where
# * [ ]  tvm.relay.argmax
# * [ ]  tvm.relay.argmin
# * [ ]  tvm.relay.sum
# * [ ]  tvm.relay.max
# * [ ]  tvm.relay.min
# * [ ]  tvm.relay.mean
# * [ ]  tvm.relay.prod
# * [ ]  tvm.relay.strided_slice
# * [ ]  tvm.relay.broadcast_to

# # Level 5
# * [ ]  tvm.relay.image.resize
# * [ ]  tvm.relay.vision.multibox_prior
# * [ ]  tvm.relay.vision.multibox_transform_loc
# * [ ]  tvm.relay.vision.nms

# # Level 10
# * [ ]  tvm.relay.broadcast_to_like
# * [ ]  tvm.relay.collapse_sum_like
# * [ ]  tvm.relay.slice_like
# * [ ]  tvm.relay.layout_transform
# * [ ]  tvm.relay.device_copy
# * [ ]  tvm.relay.annotation.on_device
