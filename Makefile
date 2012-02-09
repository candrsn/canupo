ifndef no_openmp
    CXXFLAGS+=-fopenmp -lpthread
endif

ifdef debug
    CXXFLAGS+=-g
else
    CXXFLAGS+=-O3 -g -DNDEBUG
endif

CXXFLAGS+=$(CPPFLAGS) -std=c++0x -march=corei7 -mfpmath=sse -msse2 -pipe

CXX=g++
LAPACK=-llapack
CAIRO=-lcairo
EXT=
STRIPCMD=strip
PACKCMD=tar cfvz
PACKEXT=.tar.gz
# TODO non-static version for distribution?
PACKDIR=canupo_linux_static_64bits

ifdef static
    CXXFLAGS+=-static
    CAIRO=-lcairo -lpixman-1 -lpng -lz -lfontconfig -lfreetype -lexpat
    LAPACK=./liblapack.a ./libblas.a -lgfortran
    PACKDIR=canupo_linux_static_64bits
endif

ifdef win32
    EXT=.exe
    CXX=i686-w64-mingw32-g++
    STRIPCMD=i686-w64-mingw32-strip
    CXXFLAGS=-std=c++0x -static -pipe -march=corei7 -mfpmath=sse -msse2 -O3 -DNDEBUG -I/usr/local/mingw/include -L/usr/local/mingw/lib -DDEFINE_GETLINE=1 -DNO_MMAP 
    LAPACK=-llapack -lblas -lgfortran
    CAIRO=-mwindows -mconsole -lcairo -lpixman-1 -lpng -lz -lfontconfig -liconv -lfreetype -lexpat -lws2_32 -lmsimg32
    PACKDIR=canupo_windows_static_32bits
    PACKCMD=zip -r
    PACKEXT=.zip
endif

ifdef strip
    STRIP=echo "Stripping..."; $(STRIPCMD)
else
    STRIP=/bin/true
endif

ALL=canupo$(EXT) density$(EXT) suggest_classifier_svm$(EXT) suggest_classifier_lda$(EXT) msc_tool$(EXT) validate_classifier$(EXT) combine_classifiers$(EXT) classify$(EXT) filter$(EXT) normaldiff$(EXT) resample$(EXT)

all: $(ALL)

.PHONY: $(ALL)

pack:
	rm -rf $(PACKDIR)
	mkdir -p $(PACKDIR)
	cp $(ALL) README.txt $(PACKDIR)
	rm -f $(PACKDIR)$(PACKEXT)
	$(PACKCMD) $(PACKDIR)$(PACKEXT) $(PACKDIR)

canupo$(EXT):
	$(CXX) $(CXXFLAGS) canupo.cpp $(LAPACK) $(LDFLAGS) -o canupo$(EXT)
	@$(STRIP) canupo$(EXT)

normaldiff$(EXT):
	$(CXX) $(CXXFLAGS) normaldiff.cpp $(LAPACK) -o normaldiff$(EXT)
	@$(STRIP) normaldiff$(EXT)

display_normals$(EXT):
	$(CXX) $(CXXFLAGS) display_normals.cpp -losg -losgViewer -losgGA -o display_normals$(EXT)
	@$(STRIP) display_normals$(EXT)

density$(EXT):
	$(CXX) $(CXXFLAGS) density.cpp -o density$(EXT)
	@$(STRIP) density$(EXT)

suggest_classifier_svm$(EXT):
	$(CXX) $(CXXFLAGS) suggest_classifier_svm.cpp $(CAIRO) -o suggest_classifier_svm$(EXT)
	@$(STRIP) suggest_classifier_svm$(EXT)

suggest_classifier_lda$(EXT):
	$(CXX) $(CXXFLAGS) suggest_classifier_lda.cpp $(CAIRO) -o suggest_classifier_lda$(EXT)
	@$(STRIP) suggest_classifier_lda$(EXT)

msc_tool$(EXT):
	$(CXX) $(CXXFLAGS) msc_tool.cpp $(CAIRO) -o msc_tool$(EXT)
	@$(STRIP) msc_tool$(EXT)

validate_classifier$(EXT):
	$(CXX) $(CXXFLAGS) validate_classifier.cpp -o validate_classifier$(EXT)
	@$(STRIP) validate_classifier$(EXT)

combine_classifiers$(EXT):
	$(CXX) $(CXXFLAGS) combine_classifiers.cpp -o combine_classifiers$(EXT)
	@$(STRIP) combine_classifiers$(EXT)

classify$(EXT):
	$(CXX) $(CXXFLAGS) classify.cpp -o classify$(EXT)
	@$(STRIP) classify$(EXT)

filter$(EXT):
	$(CXX) $(CXXFLAGS) filter.cpp -o filter$(EXT)
	@$(STRIP) filter$(EXT)

resample$(EXT):
	$(CXX) $(CXXFLAGS) resample.cpp -o resample$(EXT)
	@$(STRIP) resample$(EXT)

clean:
	rm -f $(ALL)
