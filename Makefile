CXXFLAGS=-O3 -DNDEBUG -fopenmp
#CXXFLAGS=-g
CXX=g++

all: canupo density

canupo: canupo.cpp
	$(CXX) $(CXXFLAGS) -Iboost-numeric-bindings canupo.cpp -llapack -o canupo

density: density.cpp
	$(CXX) $(CXXFLAGS) density.cpp -o density

clean:
	rm -f canupo density

