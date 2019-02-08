import tvm
import topi
import numpy as np
from tvm.testing import check_numerical_grads, estimate_performance, PerformanceEstimate
import time
import inspect
import sys
from exercise.runners import run_tvm

# Whether to dump the generated code
verbose = False

def get_shape(tensor, param_values=None):
    if param_values is None:
        param_values = {}
    return [tvm.ir_pass.Simplify(tvm.ir_pass.Substitute(s, param_values)).value
            for s in tensor.shape]

def check_equivalence(outputs1, outputs2, inputs, in_range=(-10, 10), iters=10):
    outputs1 = list(outputs1)
    outputs2 = list(outputs2)
    sched1 = tvm.create_schedule([o.op for o in outputs1])
    mout1 = tvm.build(sched1, outputs1 + inputs)

    sched2 = tvm.create_schedule([o.op for o in outputs2])
    mout2 = tvm.build(sched2, outputs2 + inputs)

    arguments1 = [tvm.nd.empty(get_shape(t), t.dtype) for t in outputs1 + inputs]
    arguments2 = [tvm.nd.empty(get_shape(t), t.dtype) for t in outputs1 + inputs]

    for i in range(iters):
        arguments1 = []
        arguments2 = []
        for a in outputs1 + inputs:
            val = np.random.uniform(in_range[0], in_range[1], size=get_shape(a)).astype(a.dtype)
            arguments1.append(tvm.nd.array(val))
            arguments2.append(tvm.nd.array(val))
        mout1(*arguments1)
        mout2(*arguments2)

        for j, _ in enumerate(outputs1):
            tvm.testing.assert_allclose(arguments1[j].asnumpy(), arguments2[j].asnumpy())

def check_grad(out, inputs, args=[], in_range=(-10,10), perf=None, param_values=None):
    line = inspect.getframeinfo(inspect.stack()[1][0]).lineno

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if param_values is None:
        param_values = {}

    if verbose:
        print("\n" + 80*"=" + "\n")
        print("Testing gradients, line {}\n".format(line))
        print("Original tensors:\n")
        print(tvm.PrintTensorRecursively(out))
        print()

    sout = tvm.create_schedule(out.op)
    mout = tvm.build(sout, [out] + inputs + args)

    ones = topi.full_like(out, 1.0)

    grads = list(tvm.differentiate(out, inputs, ones))

    if verbose:
        print("Gradients:\n")
        print(tvm.PrintTensorsRecursively(grads))
        print()

    grads_sched = tvm.create_schedule([g.op for g in grads])
    mgrad = tvm.build(grads_sched, grads + inputs + args)

    lowered = tvm.lower(grads_sched, grads + inputs + args, simple_mode=True)

    if verbose:
        print("Lowered gradients:\n")
        print(lowered)
        print()

    if perf != False:
        est = estimate_performance(grads, param_values=param_values)
        est_lowered = estimate_performance(lowered, param_values=param_values)

        if verbose:
            print("Note: performance tuples are (iterations, multiplications, memory)")
            print("Expected performance of grads: {}".format(perf))
            print("Estimated performance of grads: {}".format(est.as_tuple()))
            print("Estimated performance of lowered grads: {}".format(est_lowered.as_tuple()))
            print()

        if est_lowered.memory > est.memory:
            print("WARNING: Line {}: The estimated memory consumption increased after lowering, "
                  "this may indicate that tensor bounds have been expanded too much".format(line))
            print("before: {}  after: {}".format(est, est_lowered))

        (iters, mults, mem) = est.as_tuple()
        if perf is None or isinstance(perf, str):
            pass
            # print("WARNING: Line {}: No performance information, you may set it to {}"
            #       .format(line, est.as_tuple()))
            # if isinstance(perf, str):
            #     print("0,/{!r}/{{s/{!r}/{}/}}".format(perf, perf, (iters, mults, mem)))
        elif perf != (iters, mults, mem):
            (ref_iters, ref_mults, ref_mem) = perf
            ref_est = PerformanceEstimate(*perf)

            if est <= ref_est:
                print("WARNING: Line {}: Estimated performance {} is better than {}. "
                      "Use this with sed:"
                      .format(line, est.as_tuple(), ref_est.as_tuple()))
                print("0,/{}/{{s/{}/{}/}}".format(perf, perf, (iters, mults, mem)))
            elif est >= ref_est:
                print("WARNING: Line {}: Estimated performance {} IS WORSE THAN {}"
                      .format(line, est.as_tuple(), ref_est.as_tuple()))
            else:
                print("WARNING: Line {}: Estimated performance {} does not match {}"
                      .format(line, est.as_tuple(), ref_est.as_tuple()))

            EST_RTOL = 1.5
            if iters > ref_iters*EST_RTOL or mults > ref_mults*EST_RTOL or mem > ref_mem*EST_RTOL:
                raise AssertionError("Line {}: Some of the estimated performance metrics are much "
                                     "worse than the reference ones (by {}): "
                                     "estimated {}, expected {}"
                                     .format(line, EST_RTOL, est.as_tuple(), ref_est.as_tuple()))

    input_vals = [tvm.nd.array(np.random.uniform(in_range[0], in_range[1],
                                                 size=get_shape(a, param_values)).astype(a.dtype))
                  for a in inputs]
    arg_vals = [tvm.nd.array(np.random.uniform(in_range[0], in_range[1],
                                               size=get_shape(a, param_values)).astype(a.dtype))
                for a in args]

    def fun(*arguments):
        arrays = [tvm.nd.empty(get_shape(out, param_values), out.dtype)] + \
            [tvm.nd.array(a) for a in list(arguments) + arg_vals]
        mout(*arrays)
        return arrays[0].asnumpy().sum()

    g_arg_vals = \
        [tvm.nd.empty(get_shape(i, param_values), g.dtype) for i, g in zip(inputs, grads)] + \
        input_vals + arg_vals
    mgrad(*g_arg_vals)
    g_res = [g_arg_vals[g].asnumpy() for g, _ in enumerate(grads)]

    check_numerical_grads(fun, [a.asnumpy() for a in input_vals], g_res)









all_tests = []

sev_easy = 1
sev_need_to_think = 2
sev_need_to_debug = 3

class TestSkippedException(Exception):
  def __init__(self, err):
    self.err = err

class TestReportAddSection(Exception):
  def __init__(self, comm):
    self.comment = comm

expected_missing_compute = 0
expected_missing_python = 1
expected_other = 2
class TestFailedExpectedException(Exception):
  def __init__(self, reason, comm=''):
    self.reason = reason
    self.comment = comm

class TestArgs:
  def __init__(self):
    self.nwarmup = 0
    self.nloops = 1

class TestCase:
  def __init__(self, name, func, fail_fix_severity=None, fail_comment=''):
    self.name=name
    self.func=func
    self.severity=fail_fix_severity
    self.comment=fail_comment

def add_testcase(name_func,*args, **kwargs):
  global all_tests
  (name,f) = name_func
  all_tests.append(TestCase(name,f,*args,**kwargs))
  return f

def add_testcase_fail(name,reason,comment=''):
  global all_tests
  def _oops():
    raise TestFailedExpectedException(reason,comment)
  all_tests.append(TestCase(name,lambda x: _oops()))

def add_testcase_missing_compute(name):
  global all_tests
  def _oops():
    raise TestFailedExpectedException(expected_missing_compute)
  all_tests.append(TestCase(name,lambda x: _oops()))

def add_section(comment):
  global all_tests
  def _oops():
    raise TestReportAddSection(comment)
  all_tests.append(TestCase("_",lambda x: _oops()))

def log_reset():
  with open("relay_autodiff_test.log","w") as f:
    f.write("")

def log_verbose(*args):
  with open("relay_autodiff_test.log","a") as f:
    f.write(" ".join([str(a) for a in args])+"\n")

def log(*args):
  print(*args)
  log_verbose(*args)

def color(name,code):
  return "!["+name+"](https://placehold.it/15/"+code+"/000000?text=+)"

color_green = color("green","00CC00")
color_yellow = color("yellow","ffff56")
color_orange = color("orange","ff8000")
color_red = color("red","ff3333")
color_grey = color("grey","C0C0C0")

def run_tests():
  with open("report.md","w") as rep:
    log_reset()
    rep.write("Legend:\n")
    rep.write(" * "+ color_green + " Supported\n")
    rep.write(" * "+ color_yellow + " Easy to add\n")
    rep.write(" * "+ color_orange + " Need to think\n")
    rep.write(" * "+ color_red + " Need to debug\n")
    rep.write(" * "+ color_grey + " Unable to check\n")
    rep.write("\n")

    rep.write("|Status | Name | Comment\n")
    rep.write("|:-----:|:-----|:-------\n")
    for tc in all_tests:
      name = tc.name
      f = tc.func

      def _report_fail():
        emo=""
        if tc.severity == sev_easy:
          emo=color_yellow
        elif tc.severity == sev_need_to_think:
          emo=color_orange
        elif tc.severity == sev_need_to_debug:
          emo=color_red
        else:
          assert False, name + ": Invalid severity code " + str(tc.severity)
        rep.write("|"+emo+"|tvm.relay."+name+"|"+tc.comment+"\n")

      try:
        f(TestArgs())
        log(name, "SUCCEEDED")
        rep.write("|"+color_green+"|tvm.relay."+name+"||\n")
      except TestReportAddSection as e:
        log()
        log(e.comment)
        log("================")
        log()
        rep.write("|**"+e.comment+"**||\n")
      except TestSkippedException as e:
        log(name, "SKIPPED")
        log_verbose(e.err)
        assert False, "Shouldn't be skipped tests at this point"
      except TestFailedExpectedException as e:
        log(name, "FAILED_EXPECTED", e.comment)
        if e.reason == expected_missing_python:
          rep.write("|"+color_grey+"|tvm.relay."+name+"|:snake: Missing Python API\n")
        elif e.reason == expected_missing_compute:
          rep.write("|"+color("grey","C0C0C0")+"|tvm.relay."+name+"|:computer: Missing FTVMCompute attribute\n")
        elif e.reason == expected_other:
          rep.write("|"+color("grey","C0C0C0")+"|tvm.relay."+name+"|"+e.comment+"\n")
        else:
          assert False, name + ":Invalid expected fail reason: "+str(e.reason)
      except tvm.TVMError as e:
        log(name, "FAILED")
        log_verbose(e)
        _report_fail()
      except ValueError as e:
        log(name, "FAILED")
        log_verbose(e)
        _report_fail()


def _test_diff_0arg(name, f, ishape=(1,3,4,5), *args, **kwargs):
  def _run(targs):
    a = tvm.placeholder(ishape,name="A",dtype=tvm.float32)
    try:
      t = f(*args, **kwargs)
    except Exception as e:
      raise TestSkippedException(e)
    # dt = tvm.differentiate(t,[a])
    # res = run_tvm(0,1, { a:1*np.ones(ishape).astype(np.float32) }, dt[0])
    check_grad(t, a)
    return ()
  return name,_run

def _test_diff_1arg(name,f, ishape=(1,3,4,5), *args, **kwargs):
  def _run(targs=TestArgs()):
    a = tvm.placeholder(ishape,name="A",dtype=tvm.float32)
    t = None
    try:
      t = f(a, *args, **kwargs)
    except Exception as e:
      raise TestSkippedException(e)
    check_grad(t, a, in_range=(-0.1, +0.1))
    # dt = tvm.differentiate(t,[a])
    # res = run_tvm(0,1, { a:1*np.ones(ishape).astype(np.float32) }, dt[0])
    return ()
  return name,_run

def _test_diff_2arg(name, f, ishapeA=(1,3,4,5), ishapeB=(4,5), *args, **kwargs):
  def _run(targs=TestArgs()):
    a = tvm.placeholder(ishapeA,name="a",dtype=tvm.float32)
    b = tvm.placeholder(ishapeB,name="b",dtype=tvm.float32)
    t=None; dt=None
    try:
      t = f(a,b, *args, **kwargs)
    except Exception as e:
      raise TestSkippedException(e)
    # dt = tvm.differentiate(t,[a,b])
    # res = run_tvm(0,1, {
    #     a:1.4*np.ones(ishapeA).astype(np.float32)
    #   , b:1.8*np.ones(ishapeB).astype(np.float32)
    #   }, dt[0])
    check_grad(t, [a,b], in_range=(-0.1,0.1))
    return ()
  return name,_run

integer_gradients = ":1234: Integer gradients"

# # Level 1
add_section("Level 1")
# * [X]  tvm.relay.log
## TODO: add a note about negaive values
add_testcase(_test_diff_1arg("log", topi.log),
    sev_need_to_think, "Currently we do not assert on negative values which may be incorrect")

# * [X]  tvm.relay.sqrt
add_testcase(_test_diff_1arg("sqrt", topi.sqrt), sev_easy,
  "Missing by accident, easy to fix")

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
def _test_mod():
  name="mod"
  def _run(targs=TestArgs()):
    shapeX = (1,)
    x = tvm.placeholder(shapeX,name="x",dtype=tvm.float32)
    b = tvm.placeholder((1,),name="b",dtype=tvm.float32)
    t = topi.mod(x,b)
    check_grad(t, [x,b])
    return ()
  return name,_run
tmod = add_testcase(_test_mod(), sev_need_to_think, integer_gradients)
# * [ ]  tvm.relay.tanh
add_testcase(_test_diff_1arg("tanh", topi.tanh))
# * [ ]  tvm.relay.concatenate
def _test_concatenate():
  name="concatenate"
  def _run(targs=TestArgs()):
    shapeX = (1,3,4,4)
    a = tvm.placeholder(shapeX,name="a",dtype=tvm.float32)
    b = tvm.placeholder(shapeX,name="b",dtype=tvm.float32)
    t = topi.concatenate([a,b],axis=0)
    check_grad(t, [a,b])
    return ()
  return name,_run
tconcat = add_testcase(_test_concatenate())
# * [ ]  tvm.relay.expand_dims
texpand_dims = add_testcase(_test_diff_1arg("expand_dims", topi.expand_dims, axis=1))
# * [ ]  tvm.relay.nn.softmax
add_testcase(_test_diff_1arg("softmax", topi.nn.softmax))
# * [ ]  tvm.relay.nn.log_softmax
add_testcase(_test_diff_1arg("log_softmax", topi.nn.log_softmax, ishape=(4,5)))
# * [ ]  tvm.relay.nn.relu
add_testcase(_test_diff_1arg("relu", topi.nn.relu))
# * [?]  tvm.relay.nn.dropout
add_testcase_missing_compute("dropout")
# * [?]  tvm.relay.nn.batch_norm
add_testcase_missing_compute("batch_norm")
# * [ ]  tvm.relay.nn.bias_add
tbiasadd = add_testcase(("bias_add",texpand_dims))

# # Level 2
add_section("Level 2")
# * [ ]  tvm.relay.nn.conv2d
def _test_conv2d():
  name="conv2d"
  def _run(ta=TestArgs()):
    x = tvm.placeholder((10, 2, 10, 10), name='x')
    w = tvm.placeholder((5, 2, 3, 3), name='w')
    t = topi.nn.conv2d(x, w, 1, 1, 1, layout='NCHW')
    check_grad(t, [x,w])
    w2 = tvm.placeholder((5, 2, 3, 3), name='w') # OIHW
    t2 = topi.nn.depthwise_conv2d_nhwc(x, w2, 1, 1, 1)
    check_grad(t2, [x,w2])
    return ()
  return name,_run
add_testcase(_test_conv2d())

# * [ ]  tvm.relay.nn.conv2d_transpose
def _test_conv2d_transpose():
  name="conv2d_transpose"
  def _run(ta=TestArgs()):
    x = tvm.placeholder((10, 2, 10, 10), name='x')
    w = tvm.placeholder((5, 2, 3, 3), name='w')
    t = topi.nn.conv2d_transpose_nchw(x, w, (1,1), (1,1), out_dtype=tvm.float32)
    check_grad(t, [x,w])
    return ()
  return name,_run
tconv2dt = add_testcase(_test_conv2d_transpose())
# * [ ]  tvm.relay.nn.dense
tdense = add_testcase(_test_diff_2arg("dense", topi.nn.dense, ishapeA=(5,6), ishapeB=(5,6)) )
# * [ ]  tvm.relay.nn.max_pool2d
def test_pool(name,pooling_method):
  def _run(targs=TestArgs()):
    shapeX = (1,1,4,5)
    x = tvm.placeholder(shapeX,name="a",dtype=tvm.float32)
    pool = topi.nn.pool(x,(2,2),(1,1),(0,0,0,0),pool_type=pooling_method)
    check_grad(pool, x)
    return ()
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
add_testcase(_test_diff_1arg("lrn", topi.nn.lrn, size=3),
    sev_easy, "Blocked by missing pow intrinsic")
# * [X]  tvm.relay.nn.l2_normalize
l2n = add_testcase(_test_diff_1arg("l2_normalize", topi.nn.l2_normalize,
  eps=1e-3), sev_easy, "Blocked by missing sqrt intrinsic")
# * [X]  tvm.relay.nn.contrib_conv2d_winograd_without_weight_transform
# def _test_conv2d_winograd_wwt():
#   name="conv2d_winograd_without_weight_transform"
#   def _run(ta=TestArgs()):
#     x = tvm.placeholder((10, 2, 10, 10), name='x')
#     w = tvm.placeholder((5, 2, 3, 3), name='w')
#     t = topi.nn.conv2d_winograd_without_weight_transform(x,w,1,1,1,layout='NCHW',out_dtype=tvm.float32,tile_size=1)
#     check_grad(t, [x,w])
#     return ()
#   return name,_run
# tconv2dwinwwt = add_testcase(_test_conv2d_winograd_wwt(), sev_need_to_debug, "Missing TOPI implementation")
add_testcase_fail("conv2d_winograd_without_weight_transform",expected_other, "Missing TOPI implementation")

# * [ ]  tvm.relay.nn.contrib_conv2d_winograd_weight_transform
def _test_conv2d_winograd_wt():
  name="conv2d_winograd_weight_transform"
  def _run(ta=TestArgs()):
    x = tvm.placeholder((10, 2, 3, 3), name='x')
    t = topi.nn.conv2d_winograd_weight_transform(x,2)
    check_grad(t, [x])
    return ()
  return name,_run
tconv2dwinwt = add_testcase(_test_conv2d_winograd_wt(), sev_need_to_debug)

# # Level 3
add_section("Level 3")
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
add_testcase(_test_diff_1arg("floor", topi.floor),
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.ceil
add_testcase(_test_diff_1arg("ceil", topi.ceil) ,
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.trunc
add_testcase(_test_diff_1arg("trunc", topi.trunc) ,
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.clip
tclip = add_testcase(_test_diff_1arg("clip", topi.clip, a_min = 0.1, a_max = 0.9),
    sev_need_to_debug, "Missing Not operation")
# * [X]  tvm.relay.round
add_testcase(_test_diff_1arg("round", topi.round) ,
    sev_need_to_think, integer_gradients)
# * [ ]  tvm.relay.abs
add_testcase(_test_diff_1arg("abs", topi.abs) )
# * [ ]  tvm.relay.negative
add_testcase(_test_diff_1arg("negative", topi.negative) )
# * [ ]  tvm.relay.take
# TODO: note about a) potentially bad code
def _test_take(name):
  def _run(targs=TestArgs()):
    shapeX=(1,1,4,5); shapeI=(4,)
    x = tvm.placeholder(shapeX,name="x",dtype=tvm.float32)
    i = tvm.placeholder(shapeI,name="i",dtype=tvm.int32)
    t = topi.take(x,i)
    # dt = tvm.differentiate(t,[x])
    # res = run_tvm(0,1, {
    #     x:2.0*np.ones(shapeX).astype(np.float32)
    #   , i:np.array([1,-2,3,1]).astype(np.int32)
    #   }, dt[0])
    check_grad(t, x, [i], in_range=(0,1))
    return ()
  return name,_run
add_testcase(_test_take("take"))
# * [ ]  tvm.relay.zeros
tfull = add_testcase(_test_diff_0arg("zeros", topi.full, fill_value=0, shape=(1,3,4,5), dtype=tvm.float32) )
# * [ ]  tvm.relay.zeros_like
tfull_like = add_testcase(_test_diff_1arg("zeros_like", topi.full_like, fill_value=0) )
# * [ ]  tvm.relay.ones
add_testcase(("ones",tfull))
# * [ ]  tvm.relay.ones_like
add_testcase(("ones_like",tfull_like))
# * [ ]  tvm.relay.full
add_testcase(("full",tfull))
# * [ ]  tvm.relay.full_like
add_testcase(("full_like", tfull_like))
# * [?]  tvm.relay.cast
# ## FIXME: Write a comment about type conversion. The semantic of the gradient
# of round operation is not clear
#add_testcase(_test_diff_1arg("cast", topi.cast, dtype=tvm.int32) )
add_testcase_fail("cast", expected_other, "Currently, differentiate returns zeros for non-float32 inputs")
# * [ ]  tvm.relay.split
def _test_split():
  name="split"
  def _run(targs=TestArgs()):
    shapeX=(1,1,4,5);
    x = tvm.placeholder(shapeX,name="x",dtype=tvm.float32)
    t = topi.split(x, indices_or_sections=(1,2), axis=2)
    # dt0 = tvm.differentiate(t[0],[x])[0]
    # dt1 = tvm.differentiate(t[1],[x])[0]
    # res = run_tvm(0,1, {
    #     x:2.0*np.ones(shapeX).astype(np.float32)
    #   }, dt0+dt1)
    check_grad(t[0],x)
    check_grad(t[1],x)
    return ()
  return name,_run
add_testcase(_test_split())

# # Level 4
add_section("Level 4")
# * [X]  tvm.relay.right_shift
add_testcase(_test_diff_2arg("right_shift", topi.right_shift),
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.left_shift
lsh = add_testcase(_test_diff_2arg("left_shift", topi.left_shift),
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.equal
add_testcase(_test_diff_2arg("equal", topi.equal),
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.not_equal
add_testcase(_test_diff_2arg("not_equal", topi.not_equal),
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.greater
add_testcase(_test_diff_2arg("greater", topi.greater),
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.greater_equal
add_testcase(_test_diff_2arg("greater_equal", topi.greater_equal),
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.less
add_testcase(_test_diff_2arg("less", topi.less),
    sev_need_to_think, integer_gradients)
# * [X]  tvm.relay.less_equal
add_testcase(_test_diff_2arg("less_equal", topi.less_equal),
    sev_need_to_think, integer_gradients)
# * [ ]  tvm.relay.maximum
add_testcase(_test_diff_2arg("maximum", topi.maximum))
# * [ ]  tvm.relay.minimum
add_testcase(_test_diff_2arg("minimum", topi.minimum))
# * [X]  tvm.relay.power
add_testcase(_test_diff_2arg("power", topi.power),
    sev_easy, "Missing by accident, should be easy to fix.")
# * [X]  tvm.relay.where
# def _test_where():
#   name="where"
#   def _run(targs=TestArgs()):
#     shape=(1,1,4,5);
#     x = tvm.placeholder(shape,name="x",dtype=tvm.float32)
#     y = tvm.placeholder(shape,name="y",dtype=tvm.float32)
#     t0 = topi.where(0, x, y)
#     t1 = topi.where(1, x, y)
#     dt0 = tvm.differentiate(t0,[x,y])[0]
#     dt1 = tvm.differentiate(t1,[x,y])[0]
#     res = run_tvm(0,1, {
#         x:2.0*np.ones(shapeX).astype(np.float32)
#       }, dt0+dt1)
#     print(name, "SUCCEEDED")
#     return res
#   return name,_run
add_testcase_fail("where", expected_missing_python)
# * [X]  tvm.relay.argmax
amax = add_testcase(_test_diff_1arg("argmax", topi.argmax, axis=2),
    sev_need_to_debug)
# * [X]  tvm.relay.argmin
add_testcase(_test_diff_1arg("argmin", topi.argmin),
    sev_need_to_debug)
# * [ ]  tvm.relay.sum
add_testcase(_test_diff_1arg("sum", topi.sum, axis=[2,3]))
# * [ ]  tvm.relay.max
add_testcase(_test_diff_1arg("max", topi.sum, axis=[2,3]))
# * [ ]  tvm.relay.min
add_testcase(_test_diff_1arg("min", topi.min, axis=[2,3]))
# * [ ]  tvm.relay.mean
add_testcase(_test_diff_1arg("mean", lambda x: topi.sum(x) / 42))
# * [ ]  tvm.relay.prod
add_testcase(_test_diff_1arg("prod", topi.prod))
# * [ ]  tvm.relay.strided_slice
ssl = add_testcase(_test_diff_1arg("strided_slice", topi.strided_slice,
  begin=[0,0,0,0], end=[1,1,1,1], strides=[1,1,1,1]))
# * [ ]  tvm.relay.broadcast_to
b_to = add_testcase(_test_diff_1arg("broadcast_to", topi.broadcast_to, shape=(2,3,4,5)))

# # Level 5
add_section("Level 5")
# * [X]  tvm.relay.image.resize
add_testcase(_test_diff_1arg("resize", topi.image.resize, size=(8,10)),
    sev_need_to_think, "Blocked by missing floor intrinsic")
# * [X]  tvm.relay.vision.multibox_prior
mbox = add_testcase(_test_diff_1arg("multibox_prior",
  topi.vision.ssd.multibox_prior), sev_need_to_debug)
# * [X]  tvm.relay.vision.multibox_transform_loc
def _multibox_transform_loc():
  name="multibox_transform_loc"
  def _run(targs=TestArgs()):
    cls = tvm.placeholder((5,5,3),name="cls",dtype=tvm.float32)
    pred = tvm.placeholder((5,5,3),name="pred",dtype=tvm.float32)
    anchor = tvm.placeholder((5,5,3),name="anchor",dtype=tvm.float32)
    t = topi.vision.ssd.multibox_transform_loc(cls,pred,anchor)
    dt = tvm.differentiate(t[0],[cls,pred,anchor])[0]
    res = run_tvm(0,1, {
        cls:0.5*np.ones((5,5,3)).astype(np.float32)
      , pred:0*np.ones((5,5,3)).astype(np.float32)
      , anchor:0*np.ones((5,5,3)).astype(np.float32)
      }, dt)
    return res
  return name,_run
add_testcase(_multibox_transform_loc(),
    sev_need_to_debug)
# * [X]  tvm.relay.vision.nms
def _nms():
  name="nms"
  def _run(targs=TestArgs()):
    data = tvm.placeholder((1,3,6),name="data",dtype=tvm.float32)
    vc = tvm.placeholder((4,),name="vc",dtype=tvm.int32)
    t = topi.vision.nms(data,vc)
    dt = tvm.differentiate(t[0],[data])[0]
    res = run_tvm(0,1, {
        data:0.5*np.ones((1,3,6)).astype(np.float32)
      , vc:0*np.ones((4,)).astype(np.int32)
      }, dt)
    return res
  return name,_run
add_testcase(_nms(), sev_need_to_debug)

# # Level 10
add_section("Level 10")
# * [ ]  tvm.relay.broadcast_to_like
add_testcase(("broadcast_to_like",b_to))
# * [?]  tvm.relay.collapse_sum_like
#add_testcase(_test_diff_1arg("collapse_sum_like", topi.collapse_sum, shape=(3,4,5)))
add_testcase_fail("collapse_sum_like", expected_missing_python)
# * [ ]  tvm.relay.slice_like
add_testcase(("slice_like",ssl))
# * [?]  tvm.relay.layout_transform
#add_testcase(_test_diff_1arg("layout_transform", topi.layout_transform, src="NCHW", dst="HWNC"))
add_testcase_fail("layout_transform", expected_missing_python)
# * [ ]  tvm.relay.device_copy
add_testcase_missing_compute("device_copy")
# * [ ]  tvm.relay.annotation.on_device
add_testcase_missing_compute("on_device")
