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

CLANG=../polly/llvm_build/bin/clang
# CLANG=clang

$(B)/floyd_warshall.gen: floyd_warshall.cpp
	mkdir -p $(@D) || true
	g++ -std=c++14  $^ -ltvm -o $@

$(B)/floyd_warshall.s: $(B)/floyd_warshall.gen
	$^ > $@

$(B)/syr2k.gen: syr2k.cpp syr2k.hpp
	mkdir -p $(@D) || true
	g++ -std=c++14  $< -ltvm -o $@

$(B)/syr2k.s: $(B)/syr2k.gen
	$^ asm >$@ 2>$(B)/syr2k.asm.ir

$(B)/syr2k.ll: $(B)/syr2k.gen
	$^ ll >$@ 2>$(B)/syr2k.ll.ir

$(B)/syr2k.s.so: $(B)/syr2k.s
	g++ -O3 -c -o $(B)/syr2k.o $^
	g++ -shared -fPIC -o $@ $(B)/syr2k.o

$(B)/syr2k.ll.so: $(B)/syr2k.ll
	$(CLANG) -O3 -c -fPIC -o $(B)/syr2k.o $^
	$(CLANG) -shared -fPIC -o $@ $(B)/syr2k.o -ltvm_runtime

$(B)/syr2k.s.run: syr2k_run.cpp $(B)/syr2k.s.so syr2k.hpp
	mkdir -p $(@D) || true
	g++ -std=c++14 -DTVM_SO=\"$(B)/syr2k.s.so\" $< -ltvm_runtime -lpthread -ldl -o $@

$(B)/syr2k.ll.run: syr2k_run.cpp $(B)/syr2k.ll.so syr2k.hpp
	mkdir -p $(@D) || true
	g++ -std=c++14 -DTVM_SO=\"$(B)/syr2k.ll.so\" $< -ltvm_runtime -lpthread -ldl -o $@

clean:
	rm -rf $(B)

