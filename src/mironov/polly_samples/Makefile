B = _build

.PHONY: all clean
all: \
	$(B)/floyd_warshall.gen \
	$(B)/floyd_warshall.s \
	$(B)/syr2k.gen \
	$(B)/syr2k.s \
	$(B)/syr2k.s.run \
	$(B)/syr2k.ll \
	$(B)/syr2k.ll.run \
	$(B)/syr2k.obj \
	$(B)/syr2k.obj.run \
	$(B)/syr2k.cuda


CLANG=../polly/llvm_build/bin/clang
# CLANG=clang

$(B)/floyd_warshall.gen: floyd_warshall.cpp
	mkdir -p $(@D) || true
	g++ -std=c++14  $^ -ltvm -o $@

$(B)/floyd_warshall.s: $(B)/floyd_warshall.gen
	$^ > $@

$(B)/syr2k.gen: syr2k.cpp syr2k.hpp
	mkdir -p $(@D) || true
	g++ -g -O0 -std=c++14  $< -ltvm -o $@

# ASM

$(B)/syr2k.s: $(B)/syr2k.gen
	$^ asm >$@ 2>$(B)/syr2k.asm.ir

$(B)/syr2k.s.so: $(B)/syr2k.s
	g++ -O3 -c -o $(B)/syr2k.o $^
	g++ -shared -fPIC -o $@ $(B)/syr2k.o

$(B)/syr2k.s.run: syr2k_run.cpp $(B)/syr2k.s.so syr2k.hpp
	mkdir -p $(@D) || true
	g++ -std=c++14 -DTVM_SO=\"$(B)/syr2k.s.so\" $< -ltvm_runtime -lpthread -ldl -o $@


# LL

$(B)/syr2k.ll: $(B)/syr2k.gen
	$^ ll >$@ 2>$(B)/syr2k.ll.ir

$(B)/syr2k.ll.so: $(B)/syr2k.ll
	$(CLANG) -O3 -c -fPIC -o $(B)/syr2k.o $^
	$(CLANG) -shared -fPIC -o $@ $(B)/syr2k.o -ltvm_runtime

$(B)/syr2k.ll.run: syr2k_run.cpp $(B)/syr2k.ll.so syr2k.hpp
	mkdir -p $(@D) || true
	g++ -std=c++14 -DTVM_SO=\"$(B)/syr2k.ll.so\" $< -ltvm_runtime -lpthread -ldl -o $@

# OBJ

$(B)/syr2k.obj: $(B)/syr2k.gen
	$^ obj >$@ 2>$(B)/syr2k.obj.ir
	mv $(B)/syr2k.gen.obj $@

$(B)/syr2k.obj.so: $(B)/syr2k.obj
	g++ -shared -fPIC -o $@ $(B)/syr2k.obj

$(B)/syr2k.obj.run: syr2k_run.cpp $(B)/syr2k.obj.so syr2k.hpp
	mkdir -p $(@D) || true
	g++ -std=c++14 -DTVM_SO=\"$(B)/syr2k.obj.so\" $< -ltvm_runtime -lpthread -ldl -o $@

# CUDA

$(B)/syr2k.cuda: $(B)/syr2k.gen
	$^ cuda >$@ 2>$(B)/syr2k.cuda.ir
	mv $(B)/syr2k.cuda.obj $@

# $(B)/syr2k.obj.so: $(B)/syr2k.cuda
# 	g++ -shared -fPIC -o $@ $(B)/syr2k.obj

# $(B)/syr2k.obj.run: syr2k_run.cpp $(B)/syr2k.obj.so syr2k.hpp
# 	mkdir -p $(@D) || true
# 	g++ -std=c++14 -DTVM_SO=\"$(B)/syr2k.obj.so\" $< -ltvm_runtime -lpthread -ldl -o $@


clean:
	rm -rf $(B)

