import logging
import sys

import numpy as np
import tvm

from topi import tag
from tvm import autotvm

@autotvm.template
def dense(batch, in_dim, out_dim, use_bias, dtype):
    data = tvm.placeholder((batch, in_dim), name="data", dtype=dtype)
    weight = tvm.placeholder((out_dim, in_dim), name="weight", dtype=dtype)
    if use_bias:
        bias = tvm.placeholder((out_dim,), name="bias", dtype=dtype)

    k = tvm.reduce_axis((0, in_dim), name='k')
    matmul = tvm.compute((batch, out_dim),
                         lambda i, j: tvm.sum(data[i, k] * weight[j, k], axis=k),
                         tag='dense')
    if use_bias:
        result = tvm.compute((batch, out_dim),
                             lambda i, j: matmul[i, j] + bias[j],
                             tag=tag.BROADCAST)
    else:
        result = matmul

    s = tvm.create_schedule(result.op)

    # TOPI schedule (from topi/x86/nn.py):
    # # Write cache for blocks
    # CC = s.cache_write(matmul, 'local')
    #
    # bnx = 1
    # bny = 4
    # x, y = matmul.op.axis
    # xo, yo, xi, yi = s[matmul].tile(x, y, bnx, bny)
    #
    # xc, yc = s[CC].op.axis
    # k, = s[CC].op.reduce_axis
    # ko, ki = s[CC].split(k, factor=4)
    # s[CC].reorder(ko, xc, ki, yc)
    #
    # s[CC].unroll(ki)
    # s[CC].vectorize(yc)
    #
    # s[matmul].unroll(xi)
    # s[matmul].vectorize(yi)
    #
    # fused = s[matmul].fuse(xo, yo)
    # s[matmul].parallel(fused)
    # s[CC].compute_at(s[matmul], fused)

    outs = [data, weight, bias, result] if use_bias else [data, weight, result]

    print(tvm.lower(s, outs, simple_mode=True))

    return s, outs


if __name__ == '__main__':
    batch = 10000
    in_dim = 128
    out_dim = 10
    use_bias = True
    dtype = 'float32'
    target = 'llvm'
    tune = False

    if tune:
        task = autotvm.task.create(dense, args=(batch, in_dim, out_dim, use_bias, dtype), target=target)

        logger = logging.getLogger('autotvm')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=5))

        # begin tuning, log records to file `matmul.log`
        tuner = autotvm.tuner.XGBTuner(task)
        log_file = 'dense.log'
        tuner.tune(n_trial=11,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(log_file)])

        # apply history best from log file
        with autotvm.apply_history_best(log_file):
            with tvm.target.create(target):
                s, arg_bufs = dense(batch, in_dim, out_dim, use_bias, dtype)
                func = tvm.build(s, arg_bufs)
    else:
        s, arg_bufs = dense(batch, in_dim, out_dim, use_bias, dtype)
        func = tvm.build(s, arg_bufs)

    data_np = np.random.uniform(size=(batch, in_dim)).astype(dtype)
    weight_np = np.random.uniform(size=(out_dim, in_dim)).astype(dtype)
    bias_np = np.random.uniform(size=(out_dim,)).astype(dtype)
    dense_np = np.matmul(data_np, np.transpose(weight_np)) + bias_np
    data_tvm = tvm.nd.array(data_np)
    weight_tvm = tvm.nd.array(weight_np)
    bias_tvm = tvm.nd.array(bias_np)

    dense_tvm = tvm.nd.empty(dense_np.shape)
    # check correctness
    # func(data_tvm, weight_tvm, bias_tvm, dense_tvm)
    # tvm.testing.assert_allclose(dense_np, dense_tvm.asnumpy(), rtol=1e-2)

    time_func = func.time_evaluator(func.entry_name, tvm.cpu(0), number=10, repeat=10)
    times = np.array(time_func(data_tvm, weight_tvm, bias_tvm, dense_tvm).results) * 1000
    # Example output with default schedule: Baseline (no scheduling): 11.47 ms (stddev 0.76 ms)
    # Example output with TOPI schedule: Baseline (no scheduling): 2.05 ms (stddev 1.35 ms)
    print(f"Baseline (no scheduling): {times.mean():.2f} ms (stddev {times.std():.2f} ms)")