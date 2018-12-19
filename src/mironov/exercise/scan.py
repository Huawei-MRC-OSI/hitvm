import tvm
import topi
import numpy as np
import time

from exercise.runners import run_tvm,with_tvm

def get_shape(t):
  return [tvm.ir_pass.Simplify(s).value for s in t.shape]

def sample1():
  """ The following code is equivalent to numpy.cumsum """
  m = 5
  n = 5
  X = tvm.placeholder((m, n), name="X")
  s_state = tvm.placeholder((m, n), name='state')
  s_init = tvm.compute((1, n), lambda _, i: X[0, i], name='init')
  s_update = tvm.compute((m, n), lambda t, i: s_state[t-1,i] + X[t,i], name='update')
  R = tvm.scan(s_init, s_update, s_state, X, name='scan')
  return run_tvm(0,1,
    { X:1*np.ones((5,5)).astype(np.float32) },
    R,
    debug=True)

