import tvm
import time
import numpy as np

device = "cuda"
suffix = "ptx"

n = tvm.var ("n")
A = tvm.placeholder ((n), name='A', dtype="float32")
B = tvm.placeholder ((n), name='B', dtype="float32")
C = tvm.compute (A.shape, lambda *i: A(*i) + B(*i), name='C')
s = tvm.create_schedule (C.op)

bx, tx = s[C].split (C.op.axis[0], factor=64)
s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

module = tvm.build(s, [A, B, C], device, target_host="llvm")

temp = tvm.contrib.util.tempdir()
module.save (temp.relpath("myadd.o"))

# Save device code
module.imported_modules[0].save(temp.relpath("myadd.%s" %suffix))
# Create shared library
tvm.contrib.cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])

myadd = tvm.module.load (temp.relpath("myadd.so"))
# Import "deviced" code
myadd_device = tvm.module.load(temp.relpath("myadd.%s" %suffix))
# Import module
myadd.import_module(myadd_device)

ctx = tvm.context (device, 0)
n = 1024
a = tvm.nd.array(np.random.uniform(size=(n)).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=(n)).astype(A.dtype), ctx)
c = tvm.nd.array(np.random.uniform(size=(n)).astype(A.dtype), ctx)
t0 = time.time()
myadd(a, b, c)
t1 = time.time()
print (device)
print ("GPU time: %s" %(t1 - t0))

