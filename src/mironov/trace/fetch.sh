#!/bin/sh
# https://github.com/dmlc/tvm/pull/1973

set -ex

( cd ../tvm
  git branch -D trace || true
  git fetch --force origin pull/1973/head:trace
  git checkout trace
)
