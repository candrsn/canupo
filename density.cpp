//**********************************************************************
//* This file is a part of the CANUPO project, a set of programs for   *
//* classifying automatically 3D point clouds according to the local   *
//* multi-scale dimensionality at each point.                          *
//*                                                                    *
//* Author & Copyright: Nicolas Brodu <nicolas.brodu@numerimoire.net>  *
//*                                                                    *
//* This project is free software; you can redistribute it and/or      *
//* modify it under the terms of the GNU Lesser General Public         *
//* License as published by the Free Software Foundation; either       *
//* version 2.1 of the License, or (at your option) any later version. *
//*                                                                    *
//* This library is distributed in the hope that it will be useful,    *
//* but WITHOUT ANY WARRANTY; without even the implied warranty of     *
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU  *
//* Lesser General Public License for more details.                    *
//*                                                                    *
//* You should have received a copy of the GNU Lesser General Public   *
//* License along with this library; if not, write to the Free         *
//* Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,    *
//* MA  02110-1301  USA                                                *
//*                                                                    *
//**********************************************************************/
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <set>
#include <limits>
#include <algorithm>

#include "points.hpp"
#include "predictors.hpp"
#include "helpers.hpp"

using namespace std;
using namespace boost;

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

string scaleColorMap(const int* density, const vector<FloatType>& lmind, const vector<FloatType>& lmaxd, bool hasUnlabelled) {
    // log transform. min>=0 by construction, so add 1 to take log
    if (lmind.size()==1) {
        // single class : from blue to red
        FloatType d = (log(density[0]+1) - lmind[0]) / (lmaxd[0] - lmind[0]);
        // low value = blue(hue=4/6), high = red(hue=0)
        return hueToRGBstring(FloatType(4)/FloatType(6)*(1-d));
    }

    // one color per class, density is the amount of that color
    // pb: null density would be black: better if it is white for graphs in papers
    //     => interpolate in complement space, then take complement again
    static const FloatType classColors[] = {
        0.5f, 0.5f, 0.5f, // unlabeled data
        1,1,0,  // class 0 = blue inverted
        0,1,1,  // class 1 = red inverted
        1,0,1,  // class 2 = green inverted
        0,1,0,  // class 3 = magenta
        1,0,0,  // class 4 = cyan
        0,0,1,  // class 5 = yellow
    };
    int nclasses = lmind.size();
    if (nclasses>6+(int)hasUnlabelled) {
        cerr << "Sorry, displaying more than 6 classes is not supported for now" << endl;
        exit(1);
    }
    FloatType color[] = {0,0,0};
    // convert the densities to log space and between 0 and 1 for each class first
    for (int i=0; i<nclasses; ++i) {
        FloatType coef = (log(density[i]+1) - lmind[i]) / (lmaxd[i] - lmind[i]);
        for (int j=0; j<3; ++j) color[j] += coef * classColors[(i+(1-hasUnlabelled))*3+j];
    }
    // bound check and take complement
    for (int j=0; j<3; ++j) color[j] = 1 - min(1.0, max((double)color[j], 0.0));
    char ret[8];
    snprintf(ret,8,"#%02X%02X%02X",(int)(color[0]*255.99),(int)(color[1]*255.99),(int)(color[2]*255.99));
    return ret;
}

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
density nsubdiv nametag [some scales] [: unlabeled.msc] : data.msc [ - data2.msc ...]\n\
  input: nsubdiv               # Number of subdivisions on each side of the triangle\n\
  input: nametag               # The base name for the output files. One density plot is\n\
                               # generated per selected scale, named \"nametag_scale.svg\"\n\
  input: some scales           # Selected scales at which to perform the density plot\n\
                               # All scales in the parameter file are used if not specified.\n\
  input: data.msc              # The multiscale parameters computed by canupo.\n\
                               # Use - to separate classes. Multiple files per class are allowed.\n\
                               # If no classes are specified (ex: whole scene file) the density is color-coded from blue to red.\n\
                               # If multiple classes are specified the density is coded from light to bright colors with one color per class.\n\
  input: unlabeled.msc         # like the other data, but will be displayed in grey.\n\
"<<endl;
        return 0;
}

int main(int argc, char** argv) {

    if (argc<4) return help();

    int nsubdiv = atoi(argv[1]);
    
    if (nsubdiv<=0) return help();

    string nametag = argv[2];

    int separator = 0;
    for (int i=3; i<argc; ++i) if (!strcmp(":",argv[i])) {
        separator = i;
        break;
    }
    if (!separator) return help();

    // get all unique scales
    typedef set<FloatType> ScaleSet;
    ScaleSet scalesSet;
    for (int i=3; i<separator; ++i) {
        // perhaps it has the minscale:increment:maxscale syntax
        char* col1 = strchr(argv[i],':');
        char* col2 = strrchr(argv[i],':');
        if (col1==0 || col2==0 || col1==col2) {
            FloatType scale = atof(argv[i]);
            if (scale<=0) return help("Invalid scale");
            scalesSet.insert(scale);
        } else {
            *col1++=0;
            FloatType minscale = atof(argv[i]);
            *col2++=0;
            FloatType increment = atof(col1);
            FloatType maxscale = atof(col2);
            if (minscale<=0 || maxscale<=0) return help("Invalid scale range");
            bool validRange = false;
            if ((minscale - maxscale) * increment > 0) return help("Invalid range specification");
            if (minscale<=maxscale) for (FloatType scale = minscale; scale < maxscale*(1-1e-6); scale += increment) {
                validRange = true;
                scalesSet.insert(scale);
            } else for (FloatType scale = minscale; scale > maxscale*(1+1e-6); scale += increment) {
                validRange = true;
                scalesSet.insert(scale);
            }
            // compensate roundoff errors for loop bounds
            scalesSet.insert(minscale); scalesSet.insert(maxscale);
            if (!validRange) return help("Invalid range specification");
        }
    }
    
    // Is there another separator and an unlabelled file in between ?
    bool hasUnlabelled = false;
    for (int i=separator+1; i<argc; ++i) if (!strcmp(":",argv[i])) {
        hasUnlabelled = true; // and this is actually the first class
        break;
    }
    
    // now process the multiscale files and possibly multiple classes
    int nclasses = 1;
    vector<int> classboundaries(1,0);
    int total_pts = 0;
    int ptnparams;

    cout << "reading file headers" << endl;
    
    vector<FloatType> scales;
    // read headers and ensures all files are consistent
    for (int argi = separator+1; argi<argc; ++argi) {
        if (!strcmp("-",argv[argi]) || !strcmp(":",argv[argi])) {
            ++nclasses;
            classboundaries.push_back(total_pts);
            continue;
        }
        
        MSCFile mscfile(argv[argi]);
        // read the file header
        int npts = read_msc_header(mscfile, scales, ptnparams);
        total_pts += npts;
    }
    classboundaries.push_back(total_pts);
    
    int nscales = scales.size();
    // number of features
    int fdim = scales.size() * 2;
    
    //if (scalesSet.empty()) return help();
    if (scalesSet.empty()) {
        cout << "Selecting all scales in the multiscale files" << endl;
        scalesSet.insert(scales.begin(), scales.end());
    }
    else {
        // check for consistency
        for (auto f : scalesSet) {
            bool is_in_file = false;
            for (auto s : scales) {
                if (fpeq(f,s)) {is_in_file = true; break;}
            }
            if (!is_in_file) {
                cout << "Warning: requested scale " << f << " is not present in the multiscale files, ignored" << endl;
            }
        }
        cout << "Selected scales:";
    }
    for (int i=0; i<(int)scales.size(); ++i) cout << " " << scales[i];
    cout << endl;
    
    cout << "reading data in memory" << endl;
    
    // Second pass: store all selected scale data in memory
    vector<FloatType> data(total_pts * fdim);
    int base_pt = 0;
    // and then fill data from the files
    for (int argi = separator+1; argi<argc; ++argi) {
        if (!strcmp("-",argv[argi]) || !strcmp(":",argv[argi])) continue;
        MSCFile mscfile(argv[argi]);
        vector<FloatType> scales_dummy;
        // read the file header again
        int npts = read_msc_header(mscfile, scales_dummy, ptnparams);
        // read data
        read_msc_data(mscfile,nscales,npts,&data[base_pt * fdim], ptnparams);
        base_pt += npts;
    }
    
    // one density entry per selected scale per class - init all counts to 0
    int ncells = nsubdiv*(nsubdiv+1);
    vector<int> density(nscales * ncells * nclasses, 0);
    
    cout << "building the density map" << endl;
    
    for (int ci = 0; ci<nclasses; ++ci) for (int pt=classboundaries[ci]; pt<classboundaries[ci+1]; ++pt) {
        FloatType* mscdata = &data[pt*fdim];
        for (int si=0; si<nscales; ++si) {
            FloatType a = mscdata[si * 2];
            FloatType b = mscdata[si * 2 + 1];
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
            int cellidx = ((cellx * (cellx+1) / 2) + celly) * 2 + lower;
            ++density[si * (ncells * nclasses) + cellidx * nclasses + ci];
        }
        
    }
    
    cout << "outputting result files" << endl;
    for (int si=0; si<nscales; ++si) {
        bool is_selected = false;
        for (auto f : scalesSet) {
            if (fpeq(f,scales[si])) {is_selected = true; break;}
        }
        if (!is_selected) continue;
        
        vector<int> minDensity(nclasses, numeric_limits<int>::max());
        vector<int> maxDensity(nclasses, 0);
    
        for (int cellidx = 0; cellidx < ncells; ++cellidx) {
            for (int ci = 0; ci < nclasses; ++ci) {
                int d = density[si * (ncells * nclasses) + cellidx * nclasses + ci];
                minDensity[ci] = min(minDensity[ci], d);
                maxDensity[ci] = max(maxDensity[ci], d);
            }
        }
        
        vector<FloatType> logMinDensity(nclasses);
        vector<FloatType> logMaxDensity(nclasses);
        for (int ci = 0; ci < nclasses; ++ci) {
            logMinDensity[ci] = log(minDensity[ci] + 1);
            logMaxDensity[ci] = log(maxDensity[ci] + 1);
        }
        
        stringstream filename;
        filename.precision(5);
        filename << nametag << "_" << scales[si] << ".svg";
    
        static const FloatType sqrt3 = sqrt(3);
        
        ofstream densityfile(filename.str().c_str());
        densityfile << "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\""<< svgSize << "\" height=\""<< svgSize*sqrt3/2 <<"\" >" << endl;

        FloatType scaleFactor = svgSize / FloatType(nsubdiv+1);
        FloatType top = (nsubdiv+0.5)*scaleFactor*sqrt3/2;
        FloatType strokewidth = 0.01 * scaleFactor;
        for (int x=0; x<nsubdiv; ++x) for (int y=0; y<=x; ++y) {
            // lower cell coordinates
            densityfile << "<polygon points=\"";
            densityfile << " " << (x - 0.5*y)*scaleFactor << "," << top-(0.866025403784439 * y)*scaleFactor;
            densityfile << " " << (x+1 - 0.5*y)*scaleFactor << "," << top-(0.866025403784439 * y)*scaleFactor;
            densityfile << " " << (x+1 - 0.5*(y+1))*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
            int cellidx = (x*(x+1)/2+ y)*2;
            string color = scaleColorMap(&density[si * (ncells * nclasses) + cellidx * nclasses],logMinDensity,logMaxDensity,hasUnlabelled);
            densityfile << "\" style=\"fill:" << color << "; stroke:none;\"/>" << endl;
            if (y<x) { // upper cell
                densityfile << "<polygon points=\"";
                densityfile << " " << (x - 0.5*y)*scaleFactor << "," << top-(0.866025403784439 * y)*scaleFactor;
                densityfile << " " << (x+1 - 0.5*(y+1))*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
                densityfile << " " << (x - 0.5*(y+1)
                )*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
                int cellidx = (x*(x+1)/2+ y)*2+1;
                color = scaleColorMap(&density[si * (ncells * nclasses) + cellidx * nclasses],logMinDensity,logMaxDensity,hasUnlabelled);
                densityfile << "\" style=\"fill:" << color << "; stroke:none;\"/>" << endl;
            }
        }
        densityfile << "<polygon points=\" 0,"<<top<<" "<<scaleFactor*nsubdiv*0.5<<",0 " << scaleFactor*nsubdiv<<","<<top<<" \" style=\"fill:none;stroke:#000000;stroke-width:1px;\"/>" << endl;
        
        densityfile << "</svg>" << endl;
        densityfile.close();
        cout << "Density plot for scale " << scales[si] << " written in file " << filename.str() << endl;
    }

    return 0;
}
