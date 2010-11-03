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

features_least_squares:
	$(CXX) $(CXXFLAGS) features.cpp $(LAPACK) -o features_least_squares

features_linear_svm:
	$(CXX) $(CXXFLAGS) -DLINEAR_SVM features.cpp -o features_linear_svm

features_gaussian_svm:
	$(CXX) $(CXXFLAGS) -DGAUSSIAN_SVM features.cpp -o features_gaussian_svm

features: features_least_squares features_linear_svm features_gaussian_svm
	echo "features generated"

classify:
	$(CXX) $(CXXFLAGS) classify.cpp -o classify

clean:
	rm -f canupo density annotate make_features classify

.PHONY: canupo density annotate classify clean features_least_squares features_linear_svm features_gaussian_svm

