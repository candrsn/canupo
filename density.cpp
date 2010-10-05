#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>

#include <stdlib.h>

using namespace std;

typedef float FloatType;

const int svgSize=800;

string hueToRGBstring(FloatType hue) {
    hue = 6.0f * (hue - floorf(hue)); // 0 <= hue < 1
    int r,g,b;
    if (hue < 1.0f) {
        r=255; b=0;
        g = (int)(255.99f * hue);
    }
    else if (hue < 2.0f) {
        g=255; b=0;
        r = (int)(255.99f * (2.0f-hue));
    }
    else if (hue < 3.0f) {
        g=255; r=0;
        b = (int)(255.99f * (hue-2.0f));
    }
    else if (hue < 4.0f) {
        b=255; r=0;
        g = (int)(255.99f * (4.0f-hue));
    }
    else if (hue < 5.0f) {
        b=255; g=0;
        r = (int)(255.99f * (hue-4.0f));
    }
    else {
        r=255; g=0;
        b = (int)(255.99f * (6.0f-hue));
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

int main(int argc, char** argv) {

    if (argc<2) {
        cout << "Argument required: ab_file [nsubdiv]" << endl;
        return 0;
    }

    int nsubdiv = 0;

    if (argc>2) nsubdiv = atoi(argv[2]);
    if (nsubdiv<1) {
        ifstream datafile(argv[1]);
        string line;
        int npts = 0;
        while (datafile && !datafile.eof()) {
            getline(datafile, line);
            if (line.empty()) continue;
            ++npts;
        }
        datafile.close();
        // in case of totally uniform distribution, plan for 10 points per cell on average
        // ncells = nsubdiv*nsubdiv and npts = 10 * ncells;
        nsubdiv = sqrt(npts/10);
        if (nsubdiv<2) nsubdiv = 2; // at least 1 subdivision
        if (nsubdiv>150) nsubdiv = 150; // too much isn't visible, better build better stats with more points per cell
        cout << "Using " << nsubdiv << " subdivisions" << endl;
    }
    vector<int> density(nsubdiv*(nsubdiv+1), 0);

    ifstream datafile(argv[1]);
    string line;
    while (datafile && !datafile.eof()) {
        getline(datafile, line);
        if (line.empty()) continue;
        stringstream linereader(line);
        FloatType a, b;
        linereader >> a;
        linereader >> b;

        // Density plot of (a,b) points: discretize the triangle and count how many points are in each cell
        // Transform (a,b) such that equilateral triangle goes to triangle with right angle (0,0) (1,0) (0,1)
        FloatType c = nsubdiv * (a + 0.577350269189626 * b); // sqrt(3)/3
        FloatType d = nsubdiv * (1.154700538379252 * b);     // sqrt(3)*2/3
        int cellx = (int)floor(c);
        int celly = (int)floor(d);
        int lower = (c - cellx) > (d - celly);
        if (cellx>=nsubdiv) {cellx=nsubdiv-1; lower = 1;}
        if (cellx<0) {cellx=0; lower = 1;}
        if (celly>=nsubdiv) {celly=nsubdiv-1; lower = 1;} // upper triangle cell = lower one
        if (celly<0) {celly=0; lower = 1;}
        if (celly>cellx) {celly=cellx; lower = 1;}
        //int idx = ((cellx * (cellx+1) / 2) + celly) * 2 + lower;
        ++density[((cellx * (cellx+1) / 2) + celly) * 2 + lower];
    }
    datafile.close();

    int minDensity = numeric_limits<int>::max(), maxDensity = 0;
    for (vector<int>::iterator it = density.begin(); it != density.end(); ++it) {
        if (*it<minDensity) minDensity = *it;
        if (*it>maxDensity) maxDensity = *it;
    }
    ofstream densityfile("dimdensity.svg");
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
        string color = scaleColorMap(density[(x*(x+1)/2+ y)*2],minDensity,maxDensity);
        densityfile << "\" style=\"fill:" << color << "; stroke:none;\"/>" << endl;
        if (y<x) { // upper cell
            densityfile << "<polygon points=\"";
            densityfile << " " << (x - 0.5*y)*scaleFactor << "," << top-(0.866025403784439 * y)*scaleFactor;
            densityfile << " " << (x+1 - 0.5*(y+1))*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
            densityfile << " " << (x - 0.5*(y+1)
            )*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
            color = scaleColorMap(density[(x*(x+1)/2+ y)*2+1],minDensity,maxDensity);
            densityfile << "\" style=\"fill:" << color << "; stroke:none;\"/>" << endl;
        }
    }
    densityfile << "</svg>" << endl;
    densityfile.close();

    return 0;
}
