#!/bin/sh

set -e
set -x

WD=./_build
NM=trace0

mkdir $WD || true

# Compile the model code generator
g++ -std=c++14  $NM.cpp -ltvm -o $WD/gen

# Obtain the code of the model
$WD/gen >$WD/$NM.s 2>/dev/null

# Compile the model (`vecadd` function)
g++ -c -o $WD/$NM.o $WD/$NM.s

# Build the shared library
g++ -shared -fPIC -o $WD/$NM.so $WD/$NM.o -ltvm_trace

# Compile the model loader
g++ -std=c++14 -DTVM_SO=\"$WD/$NM.so\" ${NM}run.cpp  -ltvm_runtime -lpthread -ldl -o $WD/run

# Run the model loader
$WD/run

