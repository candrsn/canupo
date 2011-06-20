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

all: canupo density suggest_classifier msc_tool validate_classifier combine_classifier classify

canupo:
	$(CXX) $(CXXFLAGS) canupo.cpp $(LAPACK) $(LDFLAGS) -o canupo

normals:
	$(CXX) $(CXXFLAGS) normals.cpp $(LAPACK) -o normals

display_normals:
	$(CXX) $(CXXFLAGS) display_normals.cpp -losg -losgViewer -losgGA -o display_normals

density:
	$(CXX) $(CXXFLAGS) density.cpp -o density

suggest_classifier:
	$(CXX) $(CXXFLAGS) suggest_classifier.cpp -lcairo -o suggest_classifier

msc_tool:
	$(CXX) $(CXXFLAGS) msc_tool.cpp -lcairo -o msc_tool

validate_classifier:
	$(CXX) $(CXXFLAGS) validate_classifier.cpp -o validate_classifier

combine_classifier:
	$(CXX) $(CXXFLAGS) combine_classifier.cpp -o combine_classifier

classify:
	$(CXX) $(CXXFLAGS) classify.cpp -o classify

clean:
	rm -f canupo density suggest_classifier msc_tool validate_classifier combine_classifier classify normals display_normals

.PHONY: canupo density suggest_classifier msc_tool validate_classifier combine_classifier classify normals display_normals

