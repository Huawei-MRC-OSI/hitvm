#include <random>
#include <iomanip>
#include <array>
#include <exception>

#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <tvm/build_module.h>
#include <topi/broadcast.h>

using namespace std;

int main(int argc, char **argv)
{
  /* Shape variable */
  auto n = tvm::var("n");
  tvm::Array<tvm::Expr> shape = {n};
  tvm::Tensor A = tvm::placeholder(shape, tvm::Float(32), "A");
  tvm::Tensor B = tvm::placeholder(shape, tvm::Float(32), "B");

  /* Build a graph for computing A + B */
  tvm::Tensor C = tvm::compute(shape, tvm::FCompute([=](auto i){ return A(i) + B(i); } )) ;

  /* Prepare a function `vecadd` with no optimizations */
  tvm::Schedule s = tvm::create_schedule({C->op});
  tvm::BuildConfig config = tvm::build_config();
  std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
  auto lowered = tvm::lower(s, {A,B,C}, "vecadd", binds, config);

  /* Output IR dump to stderr */
  cerr << lowered[0]->body << endl;

  tvm::IterVar block_idx = tvm::thread_axis(tvm::Range(), "blockIdx.x");
  tvm::IterVar thread_idx = tvm::thread_axis(tvm::Range(), "threadIdx.x");

  tvm::IterVar i,j;
  s[C].split(C->op->root_iter_vars()[0],64,&i,&j);
  s[C].bind(i, block_idx);
  s[C].bind(j, thread_idx);

  /* Output IR dump to stderr */
  tvm::Target target = tvm::target::cuda();
  tvm::Target target_host = tvm::target::llvm();
  tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);
  mod->SaveToFile(std::string(argv[0]) + ".cuda", "cuda");
  return 0;
}


