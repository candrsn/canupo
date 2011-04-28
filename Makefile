CXXFLAGS=-O3 -g -DNDEBUG -fopenmp
#CXXFLAGS=-g
CXX=g++
#LAPACK=./lapack_LINUX.a ./blas_LINUX.a
LAPACK=-llapack

all: canupo density annotate

canupo:
	$(CXX) $(CXXFLAGS) canupo.cpp $(LAPACK) -o canupo

normals:
	$(CXX) $(CXXFLAGS) normals.cpp $(LAPACK) -o normals

display_normals:
	$(CXX) $(CXXFLAGS) display_normals.cpp -losg -losgViewer -losgGA -o display_normals

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

features_user_define:
	$(CXX) $(CXXFLAGS) -DPROJ_USER_CLASSIF features.cpp -o features_user_define

features: features_least_squares features_linear_svm features_gaussian_svm features_user_define
	echo "features generated"

suggest_classifier:
	$(CXX) $(CXXFLAGS) suggest_classifier.cpp -lcairo -o suggest_classifier

suggest_classifier_fast:
	$(CXX) $(CXXFLAGS) -DSVM_FAST_MODE suggest_classifier.cpp -lcairo -o suggest_classifier_fast

validate_classifier:
	$(CXX) $(CXXFLAGS) validate_classifier.cpp -o validate_classifier

combine_classifier:
	$(CXX) $(CXXFLAGS) combine_classifier.cpp -o combine_classifier

classify:
	$(CXX) $(CXXFLAGS) classify.cpp -o classify

clean:
	rm -f canupo density annotate make_features classify suggest_classifier 

.PHONY: canupo density annotate classify clean features_least_squares features_linear_svm features_gaussian_svm features_user_define suggest_classifier suggest_classifier_fast validate_classifier combine_classifier normals display_normals

