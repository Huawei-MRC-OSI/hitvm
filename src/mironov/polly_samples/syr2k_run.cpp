
#include <iostream>
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <sys/time.h>
#include <time.h>

#include "syr2k.hpp"

using namespace std;
using namespace tvm;

static double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main(void) {

  tvm::runtime::Module mod =
    tvm::runtime::Module::LoadFromFile(TVM_SO);

  tvm::runtime::PackedFunc f = mod.GetFunction("syr2k");
  CHECK(f != nullptr);

  DLTensor* C;
  int ndim = 2;
  int dtype_code = kDLFloat;
  int dtype_bits = 64;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[2] = {N,N};

  /* Prepare the input data */
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
    device_type, device_id, &C);

  double sta, fin;

  /* Call the function */
  sta = rtclock();
  f(C);
  fin = rtclock();

  /* Print the result */
  /*
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      printf("%1.2f ", static_cast<double*>(C->data)[i*shape[1]+j]);
    }
    printf("\n");
  }
  */
  cout << endl;

  printf ("%0.6f\n", fin - sta);

  return 0;
}
