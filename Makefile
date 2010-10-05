CXXFLAGS=-O3 -DNDEBUG -fopenmp
#CXXFLAGS=-g
CXX=g++

all: canupo density xyz_to_bintree

multiscale_features:
	$(CXX) $(CXXFLAGS) multiscale_features.cpp -llapack -o multiscale_features

canupo:
	$(CXX) $(CXXFLAGS) -Iboost-numeric-bindings canupo.cpp -llapack -o canupo

xyz_to_bintree:
	$(CXX) $(CXXFLAGS) xyz_to_bintree.cpp -o xyz_to_bintree

density: density.cpp
	$(CXX) $(CXXFLAGS) density.cpp -o density

clean:
	rm -f canupo density

.PHONY: canupo xyz_to_bintree multiscale_features

