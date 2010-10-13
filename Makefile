CXXFLAGS=-O3 -DNDEBUG -fopenmp
#CXXFLAGS=-g
CXX=g++
LAPACK=./lapack_LINUX.a ./blas_LINUX.a
#LAPACK=-llapack

all: canupo density xyz_to_bintree

multiscale_123D:
	$(CXX) $(CXXFLAGS) multiscale_123D.cpp $(LAPACK) -o multiscale_123D

canupo:
	$(CXX) $(CXXFLAGS) -Iboost-numeric-bindings canupo.cpp $(LAPACK) -o canupo

xyz_to_bintree:
	$(CXX) $(CXXFLAGS) xyz_to_bintree.cpp -o xyz_to_bintree

density: density.cpp
	$(CXX) $(CXXFLAGS) density.cpp -o density

make_features:
	$(CXX) $(CXXFLAGS) make_features.cpp $(LAPACK) -o make_features

clean:
	rm -f canupo density

.PHONY: canupo xyz_to_bintree multiscale_123D make_features

