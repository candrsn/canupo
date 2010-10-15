#include <iostream>
#include <fstream>

#include <math.h>

#include "points.hpp"

using namespace std;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
annotate data.xyz data.msc annotated_file.xyz [some scales]\n\
  input: data.xyz            # Original data file that was used to compute the multiscale parameters\n\
  input: data.msc            # The multiscale parameters computed by canupo\n\
  input: some scales         # Selected scales at which to perform the annotation\n\
                             # The closest match from the parameter file is selected.\n\
                             # All scales in the parameter file are used if not specified, in increasing order.\n\
  output: annotated_file.xyz # The data with RGB columns corresponding to the local 1D/2D/3D\n\
                             # property at each point. There are 3 such colums per selected scale.\n\
  # Note: can be used on the whole scene as well, the dimensionality characterisation is local\n\
"<<endl;
        return 0;
}

int main(int argc, char** argv) {
    
    if (argc<3) return help();
    
    cloud.load_txt(argv[1]);
    
    ifstream mscfile(argv[2], ifstream::binary);
    
    int npts;
    mscfile.read((char*)&npts,sizeof(npts));
    if (npts!=cloud.data.size()) return help("Data file mismatch (xyz/msc)");
    
    int nscales;
    mscfile.read((char*)&nscales, sizeof(nscales));
    vector<FloatType> scales(nscales);
    for (int si=0; si<nscales; ++si) mscfile.read((char*)&scales[si], sizeof(FloatType));

    ofstream annotatedfile(argv[3]);
    
    vector<int> selectedScalesIdx;
    for (int argi=4; argi<argc; ++argi) {
        FloatType selectedScale = atof(argv[argi]);
        if (selectedScale<=0) return help("An invalid scale was specified");
        int scaleFound = -1;
        for (int si=0; si<nscales; ++si) {
            FloatType ratio = scales[si]/selectedScale;
            if (ratio>1-1e-6 && ratio<1+1e-6) {scaleFound = si; break;}
        }
        if (scaleFound<0) return help("scale not found in msc file");
        selectedScalesIdx.push_back(scaleFound);
    }
    if (selectedScalesIdx.empty()) for (int si=nscales-1; si>=0; --si) selectedScalesIdx.push_back(si);

    cout << "Annotating data file with RGB = 1D/2D/3D information for the following scales:";
    for (int seli=0; seli<selectedScalesIdx.size(); ++seli) cout << " " << scales[selectedScalesIdx[seli]];
    cout << endl;
    
    for (int pt=0; pt<npts; ++pt) {
        // points might be shuffled by multi-core parallelism in canupo
        int ptidx;
        mscfile.read((char*)&ptidx, sizeof(ptidx));
        vector<FloatType> mscdata(nscales*2);
        for (int si=0; si<nscales; ++si) {
            mscfile.read((char*)&mscdata[si*2], sizeof(FloatType));
            mscfile.read((char*)&mscdata[si*2+1], sizeof(FloatType));
        }
        // output the point in the annotated file
        annotatedfile << cloud.data[ptidx].x << " " << cloud.data[ptidx].y << " "  << cloud.data[ptidx].z;
        // now process the selected scales
        for (int seli=0; seli<selectedScalesIdx.size(); ++seli) {
            int scaleIdx = selectedScalesIdx[seli];
            FloatType a = mscdata[scaleIdx * 2];
            FloatType b = mscdata[scaleIdx * 2 + 1];
            FloatType c = 1 - a - b; if (c<0) c = 0; if (c>1) c=1;
            int R = int(floor(a * 255.999));
            int G = int(floor(b * 255.999));
            int B = int(floor(c * 255.999));
            annotatedfile << " " << R << " " << G << " " << B;
        }
        annotatedfile << endl;
    }
    
    mscfile.close();
    annotatedfile.close();
    
    return 0;
}
