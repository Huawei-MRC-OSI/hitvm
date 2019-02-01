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

  tvm::Tensor C3 = tvm::compute(shape_C, tvm::FCompute([=](auto axis){
      return C1(axis) + C2(axis);
      }),"C3");

  /* Prepare a lowered func */
  tvm::Schedule s = tvm::create_schedule({C3->op});
  tvm::BuildConfig config = tvm::build_config();
  std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
  auto lowered = tvm::lower(s, {C3}, "syr2k", binds, config);

  /* Output IR dump to stderr */
  cerr << lowered[0]->body << endl;

  auto target = tvm::Target::create("llvm");
  auto target_host = tvm::Target::create("llvm");
  tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);

  /* Output LLVM assembly to stdout */
  cout << mod->GetSource(argv[1]) << endl;
  return 0;
}


