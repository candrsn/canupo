ifndef no_openmp
    CXXFLAGS+=-fopenmp -lpthread
endif

ifdef debug
    CXXFLAGS+=-g
else
    CXXFLAGS+=-O3 -g -DNDEBUG
endif

CXXFLAGS+=$(CPPFLAGS) -std=c++0x -march=native

CXX=g++
LAPACK=-llapack
CAIRO=-lcairo
EXT=
STRIPCMD=strip

ifdef win32
    EXT=.exe
    CXX=i686-w64-mingw32-g++
    STRIPCMD=i686-w64-mingw32-strip
    CXXFLAGS=-std=c++0x -static -pipe -march=corei7 -mfpmath=sse -msse2 -O3 -DNDEBUG -I/usr/local/mingw/include -L/usr/local/mingw/lib -DDEFINE_GETLINE=1 -DNO_MMAP
    LAPACK=-llapack -lblas -lgfortran
    CAIRO=-mwindows -mconsole -lcairo -lpixman-1 -lpng -lz -lfontconfig -liconv -lfreetype -lexpat -lws2_32 -lmsimg32
endif

ifdef strip
    STRIP=echo "Stripping..."; $(STRIPCMD)
else
    STRIP=/bin/true
endif

all: canupo density suggest_classifier_svm suggest_classifier_lda msc_tool validate_classifier combine_classifiers classify filter normaldiff

canupo:
	$(CXX) $(CXXFLAGS) canupo.cpp $(LAPACK) $(LDFLAGS) -o canupo$(EXT)
	@$(STRIP) canupo$(EXT)

normaldiff:
	$(CXX) $(CXXFLAGS) normaldiff.cpp $(LAPACK) -o normaldiff$(EXT)
	@$(STRIP) normaldiff$(EXT)

display_normals:
	$(CXX) $(CXXFLAGS) display_normals.cpp -losg -losgViewer -losgGA -o display_normals$(EXT)
	@$(STRIP) display_normals$(EXT)

density:
	$(CXX) $(CXXFLAGS) density.cpp -o density$(EXT)
	@$(STRIP) density$(EXT)

suggest_classifier_svm:
	$(CXX) $(CXXFLAGS) suggest_classifier_svm.cpp $(CAIRO) -o suggest_classifier_svm$(EXT)
	@$(STRIP) suggest_classifier_svm$(EXT)

suggest_classifier_lda:
	$(CXX) $(CXXFLAGS) suggest_classifier_lda.cpp $(CAIRO) -o suggest_classifier_lda$(EXT)
	@$(STRIP) suggest_classifier_lda$(EXT)

msc_tool:
	$(CXX) $(CXXFLAGS) msc_tool.cpp $(CAIRO) -o msc_tool$(EXT)
	@$(STRIP) msc_tool$(EXT)

validate_classifier:
	$(CXX) $(CXXFLAGS) validate_classifier.cpp -o validate_classifier$(EXT)
	@$(STRIP) validate_classifier$(EXT)

combine_classifiers:
	$(CXX) $(CXXFLAGS) combine_classifiers.cpp -o combine_classifiers$(EXT)
	@$(STRIP) combine_classifiers$(EXT)

classify:
	$(CXX) $(CXXFLAGS) classify.cpp -o classify$(EXT)
	@$(STRIP) classify$(EXT)

filter:
	$(CXX) $(CXXFLAGS) filter.cpp -o filter$(EXT)
	@$(STRIP) filter$(EXT)

resample:
	$(CXX) $(CXXFLAGS) resample.cpp -o resample$(EXT)
	@$(STRIP) resample$(EXT)

clean:
	rm -f canupo$(EXT) normaldiff$(EXT) display_normals$(EXT) density$(EXT) suggest_classifier_svm$(EXT) suggest_classifier_lda$(EXT) msc_tool$(EXT) validate_classifier$(EXT) combine_classifier$(EXT) classify$(EXT) filter$(EXT) resample$(EXT) normaldiff$(EXT)

.PHONY: canupo normaldiff display_normals density suggest_classifier_svm suggest_classifier_lda msc_tool validate_classifier combine_classifier classify filter resample
