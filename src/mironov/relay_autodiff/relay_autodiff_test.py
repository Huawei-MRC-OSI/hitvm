import tvm
import topi
import numpy as np

from exercise.runners import run_tvm

all_tests = {}

class TestArgs:
  def __init__(self):
    self.nwarmup = 0
    self.nloops = 1

def add_testcase(name_func):
  global all_tests
  (name,f) = name_func
  all_tests[name] = f
  return ()

def log_reset():
  with open("relay_autodiff_test.log","w") as f:
    f.write("")

def log_verbose(*args):
  with open("relay_autodiff_test.log","a") as f:
    f.write(" ".join([str(a) for a in args])+"\n")

def log(*args):
  print(*args)
  log_verbose(*args)

def run_tests():
  log_reset()
  for name,f in all_tests.items():
    try:
      f(TestArgs())
    except tvm.TVMError as e:
      log(name, "FAILED")
      log_verbose(e)

def _test_diff_1arg(name,f, ishape=(1,3,4,5), *args, **kwargs):
  def _run(targs):
    a = tvm.placeholder(ishape,name="A",dtype=tvm.float32)
    try:
      t = f(a, *args, **kwargs)
    except Exception as e:
      log(name, "SKIPPED")
      log_verbose(e)
      return
    dt = tvm.differentiate(t,[a])
    res = run_tvm(0,1, { a:1*np.ones(ishape).astype(np.float32) }, dt[0])
    print(name, "SUCCEEDED")
    return res
  return name,_run

def _test_diff_2arg(name, f, ishapeA=(1,3,4,5), ishapeB=(4,5), *args, **kwargs):
  def _run(targs):
    a = tvm.placeholder(ishapeA,name="a",dtype=tvm.float32)
    b = tvm.placeholder(ishapeB,name="b",dtype=tvm.float32)
    t=None; dt=None
    # try:
    t = f(a,b, *args, **kwargs)
    # except Exception as e:
    #   log(name, "SKIPPED")
    #   log_verbose(e)
    #   return
    dt = tvm.differentiate(t,[a,b])
    res = run_tvm(0,1, {
        a:1.4*np.ones(ishapeA).astype(np.float32)
      , b:1.8*np.ones(ishapeB).astype(np.float32)
      }, dt[0])
    print(name, "SUCCEEDED")
    return res
  return name,_run


# # Level 1
# * [ ]  tvm.relay.log
add_testcase(_test_diff_1arg("log", topi.log))

# * [X]  tvm.relay.sqrt
add_testcase(_test_diff_1arg("sqrt", topi.sqrt))

# * [ ]  tvm.relay.exp
add_testcase(_test_diff_1arg("exp", topi.exp))

# * [ ]  tvm.relay.sigmoid
add_testcase(_test_diff_1arg("sigmoid", topi.sigmoid))

# * [ ]  tvm.relay.add
add_testcase(_test_diff_2arg("add", lambda a,b: a+b))

# * [ ]  tvm.relay.subtract
add_testcase(_test_diff_2arg("substract", lambda a,b: a-b))
# * [ ]  tvm.relay.multiply
add_testcase(_test_diff_2arg("multiply", lambda a,b: a*b))
# * [ ]  tvm.relay.divide
add_testcase(_test_diff_2arg("divide", lambda a,b: a/b))
# * [X]  tvm.relay.mod
## TODO: Probably, a usage error
add_testcase(_test_diff_2arg("mod", lambda a,b: a%b))
# * [ ]  tvm.relay.tanh
add_testcase(_test_diff_1arg("tanh", topi.tanh))
# * [?]  tvm.relay.concatenate
# * [ ]  tvm.relay.expand_dims
add_testcase(_test_diff_1arg("expand_dims", topi.expand_dims, axis=1))
# * [ ]  tvm.relay.nn.softmax
add_testcase(_test_diff_1arg("softmax", topi.nn.softmax))
# * [ ]  tvm.relay.nn.log_softmax
add_testcase(_test_diff_1arg("log_softmax", topi.nn.log_softmax, ishape=(4,5)))
# * [ ]  tvm.relay.nn.relu
add_testcase(_test_diff_1arg("relu", topi.nn.relu))
# * [?]  tvm.relay.nn.dropout
# * [?]  tvm.relay.nn.batch_norm
# * [?]  tvm.relay.nn.bias_add

# # Level 2
# * [?]  tvm.relay.nn.conv2d
# * [?]  tvm.relay.nn.conv2d_transpose
# * [?]  tvm.relay.nn.dense
# * [ ]  tvm.relay.nn.max_pool2d
def test_pool(name,pooling_method):
  def _run(targs=TestArgs()):
    shapeX = (1,1,4,5)
    x = tvm.placeholder(shapeX,name="a",dtype=tvm.float32)
    pool = topi.nn.pool(x,(2,2),(1,1),(0,0,0,0),pool_type=pooling_method)
    dpool = tvm.differentiate(pool,[x])
    res = run_tvm(0,1, {
        x:2.0*np.ones(shapeX).astype(np.float32)
      }, dpool[0])
    print(name, "SUCCEEDED")
    return res
  return name,_run
add_testcase(test_pool("max_pool", pooling_method="max"))

# * [ ]  tvm.relay.nn.avg_pool2d
add_testcase(test_pool("avg_pool", pooling_method="avg"))
# * [ ]  tvm.relay.nn.global_max_pool2d
add_testcase(_test_diff_1arg("global_max_pool", topi.nn.global_pool, pool_type="max"))
# * [ ]  tvm.relay.nn.global_avg_pool2d
add_testcase(_test_diff_1arg("global_avg_pool", topi.nn.global_pool, pool_type="avg"))
# * [ ]  tvm.relay.nn.upsampling
add_testcase(_test_diff_1arg("upsampling", topi.nn.upsampling, scale=2))
# * [ ]  tvm.relay.nn.batch_flatten
add_testcase(_test_diff_1arg("flatten", topi.nn.flatten))
# * [ ]  tvm.relay.nn.pad
add_testcase(_test_diff_1arg("pad", topi.nn.pad, pad_before=(1,1,1,1), pad_after=(1,1,1,1)))
# * [X]  tvm.relay.nn.lrn
add_testcase(_test_diff_1arg("lrn", topi.nn.lrn, size=3))
# * [?]  tvm.relay.nn.l2_normalize
# * [?]  tvm.relay.nn.contrib_conv2d_winograd_without_weight_transform
# * [?]  tvm.relay.nn.contrib_conv2d_winograd_weight_transform

# # Level 3
# * [ ]  tvm.relay.nn.leaky_relu
add_testcase(_test_diff_1arg("leaky_relu", topi.nn.leaky_relu, alpha=0.3))
# * [ ]  tvm.relay.nn.prelu
add_testcase(_test_diff_2arg("prelu", topi.nn.prelu, ishapeA=(1,2,5,6), ishapeB=(2,)) )
# * [ ]  tvm.relay.reshape
add_testcase(_test_diff_1arg("reshape", topi.reshape, newshape=(3,4,1,5)) )
# * [?]   tvm.relay.reshape_like
##NOTE: Implementation of relay.reshape_like is the same of relay.reshape, both use
# topi.reshape. Is it a mistake?
add_testcase(_test_diff_1arg("reshape_like", topi.reshape, newshape=(3,4,1,5)) )
# * [ ]  tvm.relay.copy
##NOTE: Implemented as identity op on the TVM level
add_testcase(_test_diff_1arg("copy_identity", topi.identity) )
# * [ ]  tvm.relay.transpose
add_testcase(_test_diff_1arg("transpose", topi.transpose) )
# * [ ]  tvm.relay.squeeze
add_testcase(_test_diff_1arg("squeeze", topi.squeeze) )
# * [X]  tvm.relay.floor
add_testcase(_test_diff_1arg("floor", topi.floor) )
# * [X]  tvm.relay.ceil
add_testcase(_test_diff_1arg("ceil", topi.ceil) )
# * [X]  tvm.relay.trunc
add_testcase(_test_diff_1arg("trunc", topi.trunc) )
# * [?]  tvm.relay.clip
# add_testcase(_test_diff_1arg("clip", topi.clip) )
# * [X]  tvm.relay.round
add_testcase(_test_diff_1arg("round", topi.round) )
# * [ ]  tvm.relay.abs
add_testcase(_test_diff_1arg("abs", topi.abs) )
# * [ ]  tvm.relay.negative
add_testcase(_test_diff_1arg("negative", topi.negative) )
# * [ ]  tvm.relay.take
def _test_take(name):
  def _run(targs=TestArgs()):
    shapeX=(1,1,4,5); shapeI=(4,)
    x = tvm.placeholder(shapeX,name="x",dtype=tvm.float32)
    i = tvm.placeholder(shapeI,name="i",dtype=tvm.int32)
    t = topi.take(x,i)
    dt = tvm.differentiate(t,[x])
    res = run_tvm(0,1, {
        x:2.0*np.ones(shapeX).astype(np.float32)
      , i:np.array([1,2,3,1]).astype(np.int32)
      }, dt[0])
    print(name, "SUCCEEDED")
    return res
  return name,_run
add_testcase(_test_take("take"))
# * [?]  tvm.relay.zeros
# * [?]  tvm.relay.zeros_like
# * [?]  tvm.relay.ones
# * [?]  tvm.relay.ones_like
# * [ ]  tvm.relay.full
## TODO: add_testcase(_test_diff_1arg("full", topi.full) )
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
