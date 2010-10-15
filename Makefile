CXXFLAGS=-O3 -DNDEBUG -fopenmp
#CXXFLAGS=-g
CXX=g++
#LAPACK=./lapack_LINUX.a ./blas_LINUX.a
LAPACK=-llapack

all: canupo density annotate

canupo:
	$(CXX) $(CXXFLAGS) -Iboost-numeric-bindings canupo.cpp $(LAPACK) -o canupo

density:
	$(CXX) $(CXXFLAGS) density.cpp -o density

annotate:
	$(CXX) $(CXXFLAGS) annotate.cpp -o annotate

make_features:
	$(CXX) $(CXXFLAGS) make_features.cpp $(LAPACK) -o make_features

clean:
	rm -f canupo density

.PHONY: canupo density annotate make_features

