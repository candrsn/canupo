SRC=$(dir $(CURDIR)/$(lastword $(MAKEFILE_LIST)))
BUILD=$(SRC)../build/

ifdef debug
    CXXFLAGS+=-g
else
    CXXFLAGS+=-O3 -g -DNDEBUG
endif

#CXXFLAGS+=$(CPPFLAGS) -std=c++0x -march=native -mfpmath=sse -msse2 -pipe
CXXFLAGS+=$(CPPFLAGS) -std=c++0x -march=x86-64 -mfpmath=sse -msse2 -pipe

CXX=g++
LAPACK=-llapack
CAIRO=-lcairo -lpng
EXT=
STRIPCMD=strip
PACKCMD=tar cfvz
PACKEXT=.tar.gz
# TODO non-static version for distribution?
PACKDIR=canupo_linux_static_64bits

ifdef static
    CXXFLAGS+=-static -lpthread
    CAIRO=-lcairo -lpixman-1 -lpng -lz -lfontconfig -lfreetype -lexpat -lpthread
    LAPACK=./liblapack.a ./libblas.a -lgfortran -lpthread
    PACKDIR=canupo_linux_static_64bits
    CXX=g++-4.6
    no_openmp=1
endif

ifndef no_openmp
    CXXFLAGS+=-fopenmp -lpthread
endif

ifdef win32
    EXT=.exe
    CXX=i686-w64-mingw32-g++
    STRIPCMD=i686-w64-mingw32-strip
#    CXXFLAGS=-std=c++0x -static -pipe -march=corei7 -mfpmath=sse -msse2 -O3 -DNDEBUG -I/usr/local/mingw/include -L/usr/local/mingw/lib -DDEFINE_GETLINE=1 -DNO_MMAP 
    CXXFLAGS=-std=c++0x -static -pipe -march=i686 -mfpmath=sse -msse2 -O3 -DNDEBUG -I/usr/local/mingw/include -L/usr/local/mingw/lib -DDEFINE_GETLINE=1 -DNO_MMAP 
    LAPACK=-llapack -lblas -lgfortran
    CAIRO=-mwindows -mconsole -lcairo -lpixman-1 -lpng -lz -lfontconfig -liconv -lfreetype -lexpat -lws2_32 -lmsimg32
    PACKDIR=canupo_windows_static_32bits
    PACKCMD=zip -r
    PACKEXT=.zip
endif

ifdef win64
    EXT=.exe
    CXX=x86_64-w64-mingw32-g++
    STRIPCMD=x86_64-w64-mingw32-strip
#    CXXFLAGS=-std=c++0x -static -pipe -march=corei7 -mfpmath=sse -msse2 -O3 -DNDEBUG -I/usr/local/mingw/include -L/usr/local/mingw/lib -DDEFINE_GETLINE=1 -DNO_MMAP 
    CXXFLAGS=-std=c++0x -static -pipe -march=x86-64 -mfpmath=sse -m64 -msse2 -O3 -DNDEBUG -I/usr/local/mingw64/include -L/usr/local/mingw64/lib -DDEFINE_GETLINE=1 -DNO_MMAP 
    LAPACK=-llapack -lblas -lgfortran
    CAIRO=-mwindows -mconsole -lcairo -lpixman-1 -lpng -lz -lfreetype
    PACKDIR=canupo_windows_static_64bits
    PACKCMD=zip -r
    PACKEXT=.zip
endif

ifdef strip
    STRIP=echo "Stripping..."; $(STRIPCMD)
else
    STRIP=/bin/true
endif

PACKED_CANUPO=$(BUILD)canupo$(EXT) $(BUILD)density$(EXT) $(BUILD)suggest_classifier_svm$(EXT) $(BUILD)suggest_classifier_lda$(EXT) $(BUILD)msc_tool$(EXT) $(BUILD)validate_classifier$(EXT) $(BUILD)combine_classifiers$(EXT) $(BUILD)classify$(EXT) $(BUILD)filter$(EXT) $(BUILD)resample$(EXT) $(BUILD)prm_info$(EXT) $(BUILD)set_to_core$(EXT) $(BUILD)m3c2$(EXT)

ALL=$(PACKED_CANUPO)

all: $(ALL)

.PHONY: $(ALL)

pack:
	cd $(BUILD)
	rm -rf $(PACKDIR)
	mkdir -p $(PACKDIR)
	cp $(PACKED_CANUPO) $(SRC)README.txt $(PACKDIR)
	mkdir -p $(PACKDIR)/tutorial
	cp $(SRC)tutorial/floor.xyz $(SRC)tutorial/vegetation.xyz $(SRC)tutorial/scene.xyz $(SRC)tutorial/tutorial.pdf $(PACKDIR)/tutorial/
	inkscape --export-pdf=$(PACKDIR)/tutorial/overview.pdf $(SRC)tutorial/overview.svg
	rm -f $(PACKDIR)$(PACKEXT)
	$(PACKCMD) $(PACKDIR)$(PACKEXT) $(PACKDIR)

$(BUILD)canupo$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)canupo.cpp $(LAPACK) $(LDFLAGS) -o $(BUILD)canupo$(EXT)
	@$(STRIP) canupo$(EXT)

$(BUILD)m3c2$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)m3c2.cpp $(LAPACK) -o $(BUILD)m3c2$(EXT)
	@$(STRIP) m3c2$(EXT)

$(BUILD)display_normals$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)display_normals.cpp -losg -losgViewer -losgGA -o $(BUILD)display_normals$(EXT)
	@$(STRIP) display_normals$(EXT)

$(BUILD)density$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)density.cpp -o $(BUILD)density$(EXT)
	@$(STRIP) density$(EXT)

$(BUILD)suggest_classifier_svm$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)suggest_classifier_svm.cpp $(CAIRO) -o $(BUILD)suggest_classifier_svm$(EXT)
	@$(STRIP) suggest_classifier_svm$(EXT)

$(BUILD)suggest_classifier_lda$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)suggest_classifier_lda.cpp $(CAIRO) -o $(BUILD)suggest_classifier_lda$(EXT)
	@$(STRIP) suggest_classifier_lda$(EXT)

$(BUILD)msc_tool$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)msc_tool.cpp $(CAIRO) -o $(BUILD)msc_tool$(EXT)
	@$(STRIP) msc_tool$(EXT)

$(BUILD)prm_info$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)prm_info.cpp -o $(BUILD)prm_info$(EXT)
	@$(STRIP) prm_info$(EXT)

$(BUILD)validate_classifier$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)validate_classifier.cpp -o $(BUILD)validate_classifier$(EXT)
	@$(STRIP) validate_classifier$(EXT)

$(BUILD)combine_classifiers$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)combine_classifiers.cpp -o $(BUILD)combine_classifiers$(EXT)
	@$(STRIP) combine_classifiers$(EXT)

$(BUILD)classify$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)classify.cpp -o $(BUILD)classify$(EXT)
	@$(STRIP) classify$(EXT)

$(BUILD)filter$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)filter.cpp -o $(BUILD)filter$(EXT)
	@$(STRIP) filter$(EXT)

$(BUILD)set_to_core$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)set_to_core.cpp -o $(BUILD)set_to_core$(EXT)
	@$(STRIP) filter$(EXT)

$(BUILD)resample$(EXT):
	$(CXX) $(CXXFLAGS) $(SRC)resample.cpp -o $(BUILD)resample$(EXT)
	@$(STRIP) resample$(EXT)

clean:
	rm -f $(ALL)
