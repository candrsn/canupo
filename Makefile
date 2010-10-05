CXXFLAGS=-O3 -DNDEBUG -fopenmp
#CXXFLAGS=-g
CXX=g++

all: canupo density xyz_to_bintree

multiscale_123D:
	$(CXX) $(CXXFLAGS) multiscale_123D.cpp -llapack -o multiscale_123D

canupo:
	$(CXX) $(CXXFLAGS) -Iboost-numeric-bindings canupo.cpp -llapack -o canupo

xyz_to_bintree:
	$(CXX) $(CXXFLAGS) xyz_to_bintree.cpp -o xyz_to_bintree

density: density.cpp
	$(CXX) $(CXXFLAGS) density.cpp -o density

clean:
	rm -f canupo density

.PHONY: canupo xyz_to_bintree multiscale_123D

