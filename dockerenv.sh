#!/bin/sh
# This file is intended to be sourced from Docker container's interactive shell

CWD=`pwd`
mkdir $HOME/.ipython-profile 2>/dev/null || true
cat >$HOME/.ipython-profile/ipython_config.py <<EOF
print("Enabling autoreload")
c = get_config()
c.InteractiveShellApp.exec_lines = []
c.InteractiveShellApp.exec_lines.append('%load_ext autoreload')
c.InteractiveShellApp.exec_lines.append('%autoreload 2')
EOF

if test -n "$DISPLAY"; then
  alias ipython3='ipython3 --profile-dir=$HOME/.ipython-profile'
fi

export TVM=$CWD/tvm
export PYTHONPATH="$CWD/src/$USER:$TVM/python:$TVM/topi/python:$TVM/nnvm/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$TVM/build-docker:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$TVM/include:$TVM/dmlc-core/include:$TVM/HalideIR/src:$TVM/dlpack/include:$TVM/topi/include:$TVM/nnvm/include"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH=$TVM/build-docker

cdtvm() { cd $TVM ; }
cdex() { cd $TVM/nnvm/examples; }

dclean() {(
  cdtvm
  cd build-docker
  make clean
  rm CMakeCache.txt
  rm -rf CMakeFiles
)}

dmake() {(
  cdtvm
  mkdir build-docker 2>/dev/null
  cat >build/config.cmake <<EOF
    set(USE_CUDA OFF)
    set(USE_ROCM OFF)
    set(USE_OPENCL OFF)
    set(USE_METAL OFF)
    set(USE_VULKAN OFF)
    set(USE_OPENGL OFF)
    set(USE_RPC ON)
    set(USE_GRAPH_RUNTIME ON)
    set(USE_GRAPH_RUNTIME_DEBUG OFF)
    set(USE_LLVM ON)
    set(USE_BLAS openblas)
    set(USE_RANDOM OFF)
    set(USE_NNPACK OFF)
    set(USE_CUDNN OFF)
    set(USE_CUBLAS OFF)
    set(USE_MIOPEN OFF)
    set(USE_MPS OFF)
    set(USE_ROCBLAS OFF)
    set(USE_SORT ON)
EOF
  cdtvm
  ./tests/scripts/task_build.sh build-docker -j6
)}

alias build="dmake"

dtest() {(
  cdtvm
  ./tests/scripts/task_python_nnvm.sh
)}

cdc() {(
  cd $CWD
)}

