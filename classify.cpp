#include <iostream>
#include <limits>
#include <fstream>

#include <math.h>

#include "points.hpp"

using namespace std;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
classify features.prm scene.xyz scene_core.msc scene_annotated.xyz\n\
  input: features.prm         # Features computed by the make_features program\n\
  input: scene.xyz            # Point cloud to classify/annotate with each class\n\
  input: scene_core.msc       # Multiscale parameters at core points in the scene\n\
                              # This file need only contain the relevant scales for classification\n\
                              # as reported by the make_features program\n\
  output: scene_annotated.xyz # Output file containing an extra column with the class of each point\n\
                              # Scene points are labelled with the class of the nearest core point.\n\
"<<endl;
        return 0;
}

bool fpeq(FloatType a, FloatType b) {
    static const FloatType epsilon = 1e-6;
    if (b==0) return fabs(a)<epsilon;
    FloatType ratio = a/b;
    return ratio>1-epsilon && ratio<1+epsilon;
}

int main(int argc, char** argv) {

    if (argc<5) return help();

    cout << "Loading parameters and core points" << endl;
    
    ifstream classifparamsfile(argv[1], ifstream::binary);
    int nuniqueScales;
    classifparamsfile.read((char*)&nuniqueScales, sizeof(nuniqueScales));
    vector<FloatType> scales(nuniqueScales);
    for (int s=0; s<nuniqueScales; ++s) classifparamsfile.read((char*)&scales[s], sizeof(FloatType));
    int nclasses;
    classifparamsfile.read((char*)&nclasses, sizeof(nclasses));
    int nclassifiers = nclasses * (nclasses-1) / 2;
    vector<vector<FloatType> > classifierscales(nclassifiers);
    vector<vector<FloatType> > classifierweights(nclassifiers);
    for (int i=0; i<nclassifiers; ++i) {
        int numscales;
        classifparamsfile.read((char*)&numscales, sizeof(numscales));
        classifierscales[i].resize(numscales);
        for (int j=0; j<numscales; ++j) classifparamsfile.read((char*)&classifierscales[i][j], sizeof(FloatType));
        classifierweights[i].resize(2*numscales+1);
        for (int j=0; j<=2*numscales; ++j) classifparamsfile.read((char*)&classifierweights[i][j], sizeof(FloatType));
    }
    
    // reversed situation here compared to canupo:
    // - we load the core points in the cloud so as to perform neighbor searches
    // - the data itself may be unstructured as we only need to loop it in a single pass = not even loaded whole in memory
    ifstream mscfile(argv[3], ifstream::binary);
    // read the file header
    int ncorepoints;
    mscfile.read((char*)&ncorepoints,sizeof(ncorepoints));
    int nscales_msc;
    mscfile.read((char*)&nscales_msc, sizeof(int));
    vector<FloatType> scales_msc(nscales_msc);
    for (int si=0; si<nscales_msc; ++si) mscfile.read((char*)&scales_msc[si], sizeof(FloatType));
    // map classifier scales to indices in multiscale file for later computation of data indices
    // also check that all required scales for the classifiers are present in the msc file
    vector<vector<int> > classifierscalesidx(nclassifiers);
    for (int i=0; i<nclassifiers; ++i) {
        classifierscalesidx[i].resize(classifierscales[i].size());
        for (int j=0; j<classifierscales[i].size(); ++j) {
            int sidx = -1;
            for (int si=0; si<nscales_msc; ++si) if (fpeq(classifierscales[i][j],scales_msc[si])) {sidx=si; break;}
            if (sidx==-1) {
                cerr << "Invalid combination of multiscale file and classifier parameters: scale " << classifierscales[i][j] << " not found." << endl;
                cerr << "Available scales in the multiscale file:";
                for (int si=0; si<nscales_msc; ++si) cerr << " " << scales_msc[si];
                cerr << endl;
                return 1;
            }
            classifierscalesidx[i][j] = sidx;
        }
    }

    // now load the points and multiscale information from the msc file.
    // Put the points in the cloud, keep the multiscale information in a separate vector matched by point index
    vector<FloatType> mscdata(ncorepoints * nscales_msc*2);
    cloud.data.resize(ncorepoints);
    cloud.xmin = numeric_limits<FloatType>::max();
    cloud.xmax = -numeric_limits<FloatType>::max();
    cloud.ymin = numeric_limits<FloatType>::max();
    cloud.ymax = -numeric_limits<FloatType>::max();
    for (int pt=0; pt<ncorepoints; ++pt) {
        mscfile.read((char*)&cloud.data[pt].x, sizeof(FloatType));
        mscfile.read((char*)&cloud.data[pt].y, sizeof(FloatType));
        mscfile.read((char*)&cloud.data[pt].z, sizeof(FloatType));
        cloud.xmin = min(cloud.xmin, cloud.data[pt].x);
        cloud.xmax = max(cloud.xmax, cloud.data[pt].x);
        cloud.ymin = min(cloud.ymin, cloud.data[pt].y);
        cloud.ymax = max(cloud.ymax, cloud.data[pt].y);
        for (int s=0; s<nscales_msc; ++s) {
            FloatType a,b;
            mscfile.read((char*)(&a), sizeof(FloatType));
            mscfile.read((char*)(&b), sizeof(FloatType));
            FloatType c = 1 - a - b;
            // see make_features for this transform
            FloatType x = b + c / 2;
            FloatType y = c * sqrt(3)/2;
            mscdata[pt * nscales_msc*2 + s*2  ] = x;
            mscdata[pt * nscales_msc*2 + s*2+1] = y;
        }
    }
    mscfile.close();
    // complete the cloud structure by setting the grid
    FloatType sizex = cloud.xmax - cloud.xmin;
    FloatType sizey = cloud.ymax - cloud.ymin;
    
    cloud.cellside = sqrt(TargetAveragePointDensityPerGridCell * sizex * sizey / ncorepoints);
    cloud.ncellx = floor(sizex / cloud.cellside) + 1;
    cloud.ncelly = floor(sizey / cloud.cellside) + 1;
    
    cloud.grid.resize(cloud.ncellx * cloud.ncelly);
    for (int i=0; i<cloud.grid.size(); ++i) cloud.grid[i] = 0;
    // setup the grid: list the data points in each cell
    for (int pt=0; pt<ncorepoints; ++pt) {
        int cellx = floor((cloud.data[pt].x - cloud.xmin) / cloud.cellside);
        int celly = floor((cloud.data[pt].y - cloud.ymin) / cloud.cellside);
        cloud.data[pt].next = cloud.grid[celly * cloud.ncellx + cellx];
        cloud.grid[celly * cloud.ncellx + cellx] = &cloud.data[pt];
    }
    
    // store the classes of the core points
    // - the first time a core point is a neighbor of a scene point its class is computed
    // - the class is stored for later use
    // - the core points that are never selected are simply not used.
    vector<int> coreclasses(ncorepoints, -1); // init with class = -1 as marker
    
    cout << "Loading and processing scene data" << endl;
    ofstream scene_annotated(argv[4]);

    // TODO: load in mem then openmp
    
    ifstream datafile(argv[2]);
    string line;
    while (datafile && !datafile.eof()) {
        getline(datafile, line);
        if (line.empty()) continue;
        stringstream linereader(line);
        Point point;
        FloatType value;
        int i = 0;
        while (linereader >> value) {
            point[i] = value;
            if (++i==3) break;
        }
        if (i<3) {
            cerr << "Invalid data file: " << argv[2] << endl;
            return 1;
        }
        // process this point
        // first look for the nearest neighbor in core points
        int neighidx = cloud.findNearest(point);
        if (neighidx==-1) {
            cerr << "Invalid core point file: " << argv[3] << endl;
            return 1;
        }
        // if that core point already has a class, fine, otherwise compute it
        if (coreclasses[neighidx]==-1) {
            FloatType* msc = &mscdata[neighidx*nscales_msc*2];
            // one-against-one process: apply all classifiers and vote for this point class
            vector<FloatType> predictions(nclassifiers);
            vector<int> votes(nclasses, 0);
            for (int j=1; j<nclasses; ++j) for (int i=0; i<j; ++i) {
                int cidx = j*(j-1)/2 + i;
                // multi-scale classifier
                FloatType pred = classifierweights[cidx].back();
                for (int k = 0; k<classifierscalesidx[cidx].size(); ++k) {
                    int sidx = classifierscalesidx[cidx][k];
                    pred += msc[sidx*2] * classifierweights[cidx][k*2];
                    pred += msc[sidx*2+1] * classifierweights[cidx][k*2+1];
                }
                if (pred>=0) ++votes[j];
                else ++votes[i];
                predictions[cidx] = pred;
            }
            // search for max vote, in case equality = use the classifier between both to break the vote
            // TODO: loop breaking, see make_features.cpp
            int maxvote = -1; int selectedclass = 0;
            for (int i=0; i<votes.size(); ++i) {
                if (maxvote < votes[i]) {
                    selectedclass = i;
                    maxvote = votes[i];
                } else if (maxvote == votes[i]) {
                    int iclass = min(selectedclass, i);
                    int jclass = max(selectedclass, i);
                    int idx = jclass*(jclass-1)/2 + iclass;
                    // choose the best between both equal
                    if (predictions[idx]>=0) selectedclass = jclass;
                    else selectedclass = iclass;
                    maxvote = votes[selectedclass];
                }
            }
            coreclasses[neighidx] = selectedclass;
        }
        // assign the scene point to this core point class
        scene_annotated << point.x << " " << point.y << " " << point.z << " " << coreclasses[neighidx] << endl;
    }
    
    return 0;
}
