#!/bin/bash -xe

export BASE=`pwd`
export LLVM_SRC=${BASE}/llvm
POLLY_DST=${LLVM_SRC}/tools/polly
export POLLY_SRC=${BASE}/polly
CLANG_DST=${LLVM_SRC}/tools/clang
export CLANG_SRC=${BASE}/clang
export LLVM_BUILD=${BASE}/llvm_build

if [ -e /proc/cpuinfo ]; then
    procs=`cat /proc/cpuinfo | grep processor | wc -l`
else
    procs=1
fi

if ! test -d ${LLVM_SRC}; then
    git clone http://llvm.org/git/llvm.git ${LLVM_SRC}
fi

if ! test -d ${POLLY_SRC}; then
    git clone http://llvm.org/git/polly.git ${POLLY_SRC}
fi

if ! test -d ${CLANG_SRC}; then
    git clone http://llvm.org/git/clang.git ${CLANG_SRC}
fi

ln -s $POLLY_SRC $POLLY_DST
ln -s $CLANG_SRC $CLANG_DST

mkdir -p ${LLVM_BUILD}
cd ${LLVM_BUILD}

cmake ${LLVM_SRC}
make -j$procs -l$procs
make check-polly
