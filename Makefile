ifndef no_openmp
    CXXFLAGS+=-fopenmp
endif

ifdef debug
    CXXFLAGS+=-g
else
    CXXFLAGS+=-O3 -g -DNDEBUG
endif

CXXFLAGS+=$(CPPFLAGS)

CXX=g++
LAPACK=-llapack

all: canupo density suggest_classifier_svm suggest_classifier_lda msc_tool validate_classifier combine_classifiers classify

canupo:
	$(CXX) $(CXXFLAGS) canupo.cpp $(LAPACK) $(LDFLAGS) -o canupo

normaldiff:
	$(CXX) $(CXXFLAGS) normaldiff.cpp $(LAPACK) -o normaldiff

display_normals:
	$(CXX) $(CXXFLAGS) display_normals.cpp -losg -losgViewer -losgGA -o display_normals

density:
	$(CXX) $(CXXFLAGS) density.cpp -o density

suggest_classifier_svm:
	$(CXX) $(CXXFLAGS) suggest_classifier_svm.cpp -lcairo -o suggest_classifier_svm

suggest_classifier_lda:
	$(CXX) $(CXXFLAGS) suggest_classifier_lda.cpp -lcairo -o suggest_classifier_lda

msc_tool:
	$(CXX) $(CXXFLAGS) msc_tool.cpp -lcairo -o msc_tool

validate_classifier:
	$(CXX) $(CXXFLAGS) validate_classifier.cpp -o validate_classifier

combine_classifiers:
	$(CXX) $(CXXFLAGS) combine_classifiers.cpp -o combine_classifiers

classify:
	$(CXX) $(CXXFLAGS) classify.cpp -o classify

clean:
	rm -f canupo density suggest_classifier msc_tool validate_classifier combine_classifier classify normaldiff display_normals

.PHONY: canupo density suggest_classifier_svm suggest_classifier_lda msc_tool validate_classifier combine_classifiers classify normaldiff display_normals

