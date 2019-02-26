#include <random>
#include <iomanip>
#include <array>
#include <exception>

#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <tvm/build_module.h>
#include <topi/broadcast.h>

#include "syr2k.hpp"

using namespace std;
using topi::operator+;
using topi::operator-;
using topi::operator*;
using topi::operator/;
using topi::operator%;

int main(int argc, char **argv)
{
  tvm::Array<tvm::Expr> shape_A = {N,M};
  tvm::Array<tvm::Expr> shape_B = {N,M};
  tvm::Array<tvm::Expr> shape_C = {N,N};

  tvm::Type typ = tvm::Float(64);

  // Initialize
  tvm::Tensor A = tvm::compute(shape_A, tvm::FCompute([=](auto axis){
      auto i=axis[0], j=axis[1];
      return tvm::cast(typ, (i*j))%N / N; } ), "A");

  tvm::Tensor B = tvm::compute(shape_B, tvm::FCompute([=](auto axis){
      auto i=axis[0], j=axis[1];
      return tvm::cast(typ, (i*j))%M / M; } ), "B");

  tvm::Tensor C = tvm::compute(shape_C, tvm::FCompute([=](auto axis){
      auto i=axis[0], j=axis[1];
      return tvm::cast(typ, (i*j)) %N / M;
      }),"C");

  tvm::Expr Alpha(1.5);
  tvm::Expr Beta(1.2);

  tvm::Tensor C1 = tvm::compute(shape_C, tvm::FCompute([=](auto axis){
      auto i=axis[0], j=axis[1];
      return C(i,j)*Beta;
      }),"C1");

  tvm::IterVar k = tvm::reduce_axis({0,M},"k");

  tvm::Tensor C2 = tvm::compute(shape_C, tvm::FCompute([=](auto axis){
      auto i=axis[0], j=axis[1];
      return tvm::sum( A(j,k)*Alpha*B(i,k) + B(j,k)*Alpha*A(i,k), {k});
      }),"C2");

  tvm::IterVar C2_i=C2->op->root_iter_vars()[0];
  tvm::IterVar C2_j=C2->op->root_iter_vars()[1];

  tvm::Tensor C3 = C1 + C2;
  tvm::IterVar C3_i=C3->op->root_iter_vars()[0];
  tvm::IterVar C3_j=C3->op->root_iter_vars()[1];

  /* Prepare a lowered func */
  tvm::Schedule s = tvm::create_schedule({C3->op});

#if 0
  {
    tvm::Stage C2_st = s[C2->op];
    // tvm::Stage C3_st = s[C3->op];

    tvm::IterVar i,j;
    i=C2_i; j=C2_j;
    // C3_st.fuse({C3_i,C2_i}, &i);
    // C3_st.fuse({C3_j,C2_j}, &j);

    C2_st.reorder({i,j,k});
    tvm::IterVar i1,i2,j1,j2,k1,k2;
    C2_st.split(i,32,&i1,&i2);
    C2_st.split(j,32,&j1,&j2);
    C2_st.split(k,32,&k1,&k2);
  }
#endif


  /* Output LLVM assembly to stdout */
  if(std::string(argv[1]) == "cuda") {
    tvm::BuildConfig config = tvm::build_config();
    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
    auto lowered = tvm::lower(s, {C3}, "syr2k", binds, config);

    tvm::IterVar block_idx = tvm::thread_axis(tvm::Range(), "blockIdx.x");
    tvm::IterVar thread_idx = tvm::thread_axis(tvm::Range(), "threadIdx.x");

    s[C3].bind(C3_i, block_idx);
    s[C3].bind(C3_j, thread_idx);

    /* Output IR dump to stderr */
    cerr << lowered[0]->body << endl;

    tvm::Target target = tvm::Target::create("cuda");
    tvm::Target target_host = tvm::Target::create("llvm");
    tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);

    mod->SaveToFile(std::string(argv[0]) + ".cuda", "cuda");
  }
  else {
    tvm::BuildConfig config = tvm::build_config();
    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
    auto lowered = tvm::lower(s, {C3}, "syr2k", binds, config);

    /* Output IR dump to stderr */
    cerr << lowered[0]->body << endl;

    tvm::Target target = tvm::Target::create("llvm");
    tvm::Target target_host = tvm::Target::create("llvm");
    tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);

    if(std::string(argv[1]) == "obj") {
      mod->SaveToFile(std::string(argv[0]) + ".obj", "obj");
    }
    else {
      cout << mod->GetSource(argv[1]) << endl;
    }
  }
  return 0;
}


