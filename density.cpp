#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>

#include <stdlib.h>

#include "points.hpp"

using namespace std;

const int svgSize=800;

string hueToRGBstring(FloatType hue) {
    hue = 6.0f * (hue - floorf(hue)); // 0 <= hue < 1
    int r,g,b;
    if (hue < 1.0f) {
        r=255; b=0;
        g = (int)(255.999 * hue);
    }
    else if (hue < 2.0f) {
        g=255; b=0;
        r = (int)(255.999 * (2.0f-hue));
    }
    else if (hue < 3.0f) {
        g=255; r=0;
        b = (int)(255.999 * (hue-2.0f));
    }
    else if (hue < 4.0f) {
        b=255; r=0;
        g = (int)(255.999 * (4.0f-hue));
    }
    else if (hue < 5.0f) {
        b=255; g=0;
        r = (int)(255.999 * (hue-4.0f));
    }
    else {
        r=255; g=0;
        b = (int)(255.999 * (6.0f-hue));
    }
    char ret[8];
    snprintf(ret,8,"#%02X%02X%02X",r,g,b);
    return ret;
}

string scaleColorMap(int density, int mind, int maxd) {
    // log transform. min>=0 by construction, so add 1 to take log
    FloatType fmin = log(mind+1);
    FloatType fmax = log(maxd+1);
    FloatType d = (log(density+1) - fmin) / (fmax - fmin);
    // low value = blue(hue=4/6), high = red(hue=0)
    return hueToRGBstring(FloatType(4)/FloatType(6)*(1-d));
}


int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
density data.msc nsubdiv nametag [some scales]\n\
  input: data.msc              # The multiscale parameters computed by canupo\n\
  input: nsubdiv               # Number of subdivisions on each side of the triangle\n\
  input: nametag               # The base name for the output files\n\
  input: some scales           # Selected scales at which to perform the density plot\n\
                               # All scales in the parameter file are used if not specified.\n\
  output: nametag_scale.svg    # One density plot per selected scale\n\
"<<endl;
        return 0;
}

int main(int argc, char** argv) {

    if (argc<3) return help();

    ifstream mscfile(argv[1], ifstream::binary);
    
    int npts;
    mscfile.read((char*)&npts,sizeof(npts));
    if (npts<=0) help("invalid file");
    
    int nscales;
    mscfile.read((char*)&nscales, sizeof(nscales));
    vector<FloatType> scales(nscales);
    for (int si=0; si<nscales; ++si) mscfile.read((char*)&scales[si], sizeof(FloatType));
    if (nscales<=0) help("invalid file");

    int nsubdiv = 0;

    nsubdiv = atoi(argv[2]);
    if (nsubdiv<=0) {
        // in case of totally uniform distribution, plan for 10 points per cell on average
        // ncells = nsubdiv*nsubdiv and npts = 10 * ncells;
        nsubdiv = sqrt(npts/10);
        if (nsubdiv<2) nsubdiv = 2; // at least 1 subdivision
        if (nsubdiv>150) nsubdiv = 150; // too much isn't visible, better build better stats with more points per cell
        cout << "Using " << nsubdiv << " subdivisions" << endl;
    }

    string nametag = argv[3];
    
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

    // one density entry per selected scale - init all counts to 0
    vector<vector<int> > density(selectedScalesIdx.size(), vector<int>(nsubdiv*(nsubdiv+1), 0));

    for (int pt=0; pt<npts; ++pt) {
        int ptidx; // we do not care of the point order here, just the density
        mscfile.read((char*)&ptidx, sizeof(ptidx));
        vector<FloatType> mscdata(nscales*2);
        for (int si=0; si<nscales; ++si) {
            mscfile.read((char*)&mscdata[si*2], sizeof(FloatType));
            mscfile.read((char*)&mscdata[si*2+1], sizeof(FloatType));
        }
        for (int seli=0; seli<selectedScalesIdx.size(); ++seli) {
            FloatType a = mscdata[selectedScalesIdx[seli] * 2];
            FloatType b = mscdata[selectedScalesIdx[seli] * 2 + 1];
            // Density plot of (a,b) points: discretize the triangle and count how many points are in each cell
            // Barycentric coordinates : a * (0,0) + b * (1,0) + (1-a-b) * (1,1)
            FloatType c = nsubdiv * (1-a);
            FloatType d = nsubdiv * (1-a-b);
            int cellx = (int)floor(c);
            int celly = (int)floor(d);
            int lower = (c - cellx) > (d - celly);
            if (cellx>=nsubdiv) {cellx=nsubdiv-1; lower = 1;}
            if (cellx<0) {cellx=0; lower = 1;}
            if (celly>=nsubdiv) {celly=nsubdiv-1; lower = 1;} // upper triangle cell = lower one
            if (celly<0) {celly=0; lower = 1;}
            if (celly>cellx) {celly=cellx; lower = 1;}
            //int idx = ((cellx * (cellx+1) / 2) + celly) * 2 + lower;
            ++density[seli][((cellx * (cellx+1) / 2) + celly) * 2 + lower];
        }
        
    }

    vector<int> minDensity(selectedScalesIdx.size(), numeric_limits<int>::max());
    vector<int> maxDensity(selectedScalesIdx.size(), 0);
    
    for (int seli=0; seli<selectedScalesIdx.size(); ++seli) {
        for (vector<int>::iterator it = density[seli].begin(); it != density[seli].end(); ++it) {
            if (*it<minDensity[seli]) minDensity[seli] = *it;
            if (*it>maxDensity[seli]) maxDensity[seli] = *it;
        }
    }
    
    for (int seli=0; seli<selectedScalesIdx.size(); ++seli) {
        stringstream filename;
        filename.precision(5);
        filename << nametag << "_" << scales[selectedScalesIdx[seli]] << ".svg";
    
        ofstream densityfile(filename.str().c_str());
        densityfile << "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\""<< svgSize << "\" height=\""<< svgSize*sqrt(3)/2 <<"\" >" << endl;

        FloatType scaleFactor = svgSize / FloatType(nsubdiv+1);
        FloatType top = (nsubdiv+0.5)*scaleFactor*sqrt(3)/2;
        FloatType strokewidth = 0.01 * scaleFactor;
        for (int x=0; x<nsubdiv; ++x) for (int y=0; y<=x; ++y) {
            // lower cell coordinates
            densityfile << "<polygon points=\"";
            densityfile << " " << (x - 0.5*y)*scaleFactor << "," << top-(0.866025403784439 * y)*scaleFactor;
            densityfile << " " << (x+1 - 0.5*y)*scaleFactor << "," << top-(0.866025403784439 * y)*scaleFactor;
            densityfile << " " << (x+1 - 0.5*(y+1))*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
            string color = scaleColorMap(density[seli][(x*(x+1)/2+ y)*2],minDensity[seli],maxDensity[seli]);
            densityfile << "\" style=\"fill:" << color << "; stroke:none;\"/>" << endl;
            if (y<x) { // upper cell
                densityfile << "<polygon points=\"";
                densityfile << " " << (x - 0.5*y)*scaleFactor << "," << top-(0.866025403784439 * y)*scaleFactor;
                densityfile << " " << (x+1 - 0.5*(y+1))*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
                densityfile << " " << (x - 0.5*(y+1)
                )*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
                color = scaleColorMap(density[seli][(x*(x+1)/2+ y)*2+1],minDensity[seli],maxDensity[seli]);
                densityfile << "\" style=\"fill:" << color << "; stroke:none;\"/>" << endl;
            }
        }
        densityfile << "</svg>" << endl;
        densityfile.close();
        cout << "Density plot for scale " << scales[selectedScalesIdx[seli]] << " written in file " << filename.str() << endl;
    }

    return 0;
}
