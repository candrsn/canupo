#include <iostream>
#include <limits>
#include <fstream>
#include <map>

#include <math.h>

#include "points.hpp"

using namespace std;
using namespace boost;

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

struct Classifier {
    
    enum {gridsize = 100};
    
    int class1, class2;
    vector<FloatType> weights_axis1, weights_axis2;
    vector<Point2D> path;
    FloatType absmaxXY;
    
    struct LineDef {
        FloatType wx, wy, c;
    };
    vector<LineDef> pathlines;
    
    Point2D refpt, refpt2;
    
    vector<FloatType> grid;

    void prepare() {
        // compute the lines
        for(int i=0; i<path.size()-1; ++i) {
            LineDef ld;
            FloatType xdelta = path[i+1].x - path[i].x;
            FloatType ydelta = path[i+1].y - path[i].y;
            if (fabs(xdelta) > 1e-6) {
                // y = slope * x + bias
                ld.wy = -1;
                ld.wx = ydelta / xdelta; // slope
                ld.c = path[i].y - path[i].x * ld.wx;
            } else {
                if (fabs(xdelta) < 1e-6) {
                    cerr << "invalid path definition in classifier" << endl;
                    exit(1);
                }
                // just reverse the roles for a quasi-vertical line at x ~ cte
                ld.wx = -1;
                ld.wy = xdelta / ydelta; // is quasi null here, assuming ydelta != 0
                ld.c = path[i].x - path[i].y * ld.wy;
            }
            // normalize so we have unit norm vector
            FloatType n = sqrt(ld.wx*ld.wx + ld.wy*ld.wy);
            ld.wx /= n; ld.wy /= n; ld.c /= n;
            pathlines.push_back(ld);
        }

        // ref points to classify. refpt shall be in class -1, classified as such by both SVM giving the main directions
        refpt = Point2D(-M_PI*absmaxXY, -M_PI*absmaxXY);
        // slighly off so to account for path lines parallel to refpt-somept, but shall still be in the -1 class
        refpt2 = Point2D(-(M_PI+2)*absmaxXY, -(M_PI+1)*absmaxXY);
        
        // compute the classification at each grid point
        vector<FloatType> nodeclassif((gridsize+1)*(gridsize+1));
        for (int j=0; j<=gridsize; ++j) {
            FloatType b = -absmaxXY + j * 2 * absmaxXY / gridsize;
            for (int i=0; i<=gridsize; ++i) {
                FloatType a = -absmaxXY + i * 2 * absmaxXY / gridsize;
                nodeclassif[j*(gridsize+1)+i] = classify2D(a,b,false);
            }
        }

        // some grid cells can be completely classified, speed up further operations
        grid.resize(gridsize*gridsize);
        for (int j=0; j<gridsize; ++j) for (int i=0; i<gridsize; ++i) {
            bool classifiedCell = true;
            if (nodeclassif[j*(gridsize+1)+i] >= 0) {
                classifiedCell &= nodeclassif[j*(gridsize+1)+i+1] >= 0;
                classifiedCell &= nodeclassif[(j+1)*(gridsize+1)+i] >= 0;
                classifiedCell &= nodeclassif[(j+1)*(gridsize+1)+i+1] >= 0;
            }
            if (nodeclassif[j*(gridsize+1)+i] <= 0) {
                classifiedCell &= nodeclassif[j*(gridsize+1)+i+1] <= 0;
                classifiedCell &= nodeclassif[(j+1)*(gridsize+1)+i] <= 0;
                classifiedCell &= nodeclassif[(j+1)*(gridsize+1)+i+1] <= 0;
            }
            FloatType avg = nodeclassif[j*(gridsize+1)+i];
            avg += nodeclassif[j*(gridsize+1)+i+1];
            avg += nodeclassif[(j+1)*(gridsize+1)+i];
            avg += nodeclassif[(j+1)*(gridsize+1)+i+1];
            // if avg is null then consider the cell is not classified in any case
            // otherwise points have avg class boundary within the cell
            // TODO: the min might be more relevant later on when we deal with the laser intensity information
            grid[j*gridsize+i] = classifiedCell ? (avg/4) : 0;
        }
    }

    // classification in the 2D space
    FloatType classify2D(FloatType a, FloatType b, bool usegrid = true) {
        if (usegrid && fabs(a) < absmaxXY && fabs(b) < absmaxXY) {
            int i = (int)floor((a + absmaxXY) * gridsize / (2 * absmaxXY));
            int j = (int)floor((b + absmaxXY) * gridsize / (2 * absmaxXY));
            // return grid classif if it is unique, otherwise compute for this point
            if (grid[j*gridsize+i]!=0) return grid[j*gridsize+i];
        }
        Point2D* refptr = &refpt;
        Point2D* refptr2 = &refpt2;
        if (a == refpt.x) {
            refptr = &refpt2;
            refptr2 = &refpt;
        }
        // line equa from refpt to (a,b).
        // y = refpt.y + (b - refpt.y) * (x - refpt.x) / (a - refpt.x)
        // y = refslope * x + refbias
        FloatType refslope = (b - refptr->y) / (a - refptr->x); // not dividing by 0
        FloatType refbias = refptr->y - refptr->x * refslope;
        // equa for each segment: wx * x + wy * y + wc = 0
        // intersection: wx * x + (wy * refslope) * x + (wc+refbias) = 0
        // x = -(wc+refbias) / (wx + wy * refslope);  and  y = refslope * x + refbias
        // if (wx + wy * refslope) is null : no intersection, parallel lines
        // => use a secondary ref point on a different line
        int crosscount = 0;
        FloatType closestDist = numeric_limits<FloatType>::max();
        for (int i=0; i<pathlines.size(); ++i) {
            FloatType divisor = pathlines[i].wx + pathlines[i] * wy * refslope;
            FloatType intersectx;
            FloatType intersecty;
            if (fabs(divisor)<1e-3) {
                FloatType ref2slope = (b - refptr2->y) / (a - refptr2->x);
                FloatType ref2bias = refptr2->y - refptr2->x * ref2slope;
                divisor = pathlines[i].wx + pathlines[i] * wy * ref2slope;
                intersectx = (ref2bias - pathlines[i].wx) / divisor;
                intersecty = ref2slope * intersectx + ref2bias;
            } else {
                intersectx = (refbias - pathlines[i].wx) / divisor;
                intersecty = refslope * intersectx + refbias;
            }
            bool intersect = true;
            // first and last segments are prolongated to infinity
            if (i==0) {
                FloatType xdelta = path[i+1].x - intersectx;
                FloatType ydelta = path[i+1].y - intersecty;
                // use the more reliable delta
                if (fabs(xdelta)>fabs(ydelta)) {
                    // intersection is valid only if on the half-infinite side of the segment
                    intersect &= (xdelta * (path[i+1].x - path[i].x)) > 0;
                } else {
                    intersect &= (ydelta * (path[i+1].y - path[i].y)) > 0;
                }
            } else if (i==pathlines.size()-1) {
                // idem, just infinite on the other side
                FloatType xdelta = path[i].x - intersectx;
                FloatType ydelta = path[i].y - intersecty;
                if (fabs(xdelta)>fabs(ydelta)) {
                    intersect &= (xdelta * (path[i].x - path[i+1].x)) > 0;
                } else {
                    intersect &= (ydelta * (path[i].y - path[i+1].y)) > 0;
                }
            } else {
                // intersection is valid only within the segment boundaries
                intersect &= (intersectx >= min(path[i].x,path[i+1].x)) && (intersecty >= min(path[i].y,path[i+1].y));
                intersect &= (intersectx <= max(path[i].x,path[i+1].x)) && (intersecty <= max(path[i].y,path[i+1].y));
            }
            // nodes joining segments might be duplicated, but then, they are on the boundary, so we do not care
            // which class they are put into (either odd or even crossings)
            if (intersect) ++crosscount;
            // closest distance from the point to that segment
            // 1. projection of the point of the line
            Point2D p(a,b);
            Point2D n(pathlines[i].wx, pathlines[i].wy);
            p -= n * n.dot(p + n * pathlines[i].c);
            FloatType closestToSeg = numeric_limits<FloatType>::max();
            // 2. Is the projection within the segment limit ? yes => closest
            if (intersect) closestToSeg = dist(Point2D(a,b), p);
            else {
                // 3. otherwise closest is the minimum of the distance to the segment ends
                if (i!=0) closestToSeg = dist(Point2D(a,b), Point2D(path[i].x,path[i].y));
                if (i!=pathlines.size()-1) closestToSeg = min(closestToSeg, dist(Point2D(a,b), Point2D(path[i+1].x,path[i+1].y)));
            }
            closestDist = min(closestDist, closestToSeg);
        }
        // even number of crossings => -1 class, odd = +1.
        // then return closestDist as the confidence in this class
        return ((crosscount&1) * 2 - 1) * closestDist;
    }

    // classification in MSC space
    FloatType classify(FloatType* mscdata) {
        FloatType a = weights_axis1[weights_axis1.size()-1];
        FloatType b = weights_axis2[weights_axis2.size()-1];
        for (int d=0; d<weights_axis1.size()-1; ++d) {
            a += weights_axis1[d] * mscdata[d];
            b += weights_axis2[d] * mscdata[d];
        }
        return classify2D(a,b);
    }
};


int main(int argc, char** argv) {

    if (argc<5) return help();

    cout << "Loading parameters and core points" << endl;
    
    ifstream classifparamsfile(argv[1], ifstream::binary);
    int nscales;
    classifparamsfile.read((char*)&nscales, sizeof(int));
    int fdim = nscales*2;
    vector<FloatType> scales(nscales);
    for (int s=0; s<nscales; ++s) classifparamsfile.read((char*)&scales[s], sizeof(FloatType));
    int nclassifiers; // number of 2-class classifiers
    classifparamsfile.read((char*)&nclassifiers, sizeof(int));
    vector<Classifier> classifiers(nclassifiers);
    for (int ci=0; ci<nclassifiers; ++ci) {
        classifparamsfile.read((char*)&classifiers[ci].class1, sizeof(int));
        classifparamsfile.read((char*)&classifiers[ci].class2, sizeof(int));
        classifiers[ci].weights_axis1.resize(fdim+1);
        classifiers[ci].weights_axis2.resize(fdim+1);
        for (int i=0; i<=fdim; ++i) classifierfile.read((char*)&classifiers[ci].weights_axis1[i],sizeof(FloatType));
        for (int i=0; i<=fdim; ++i) classifierfile.read((char*)&classifiers[ci].weights_axis2[i],sizeof(FloatType));
        int pathsize;
        classifierfile.read((char*)&pathsize,sizeof(int));
        classifiers[ci].path.resize(pathsize);
        for (int i=0; i<pathsize; ++i) {
            classifierfile.read((char*)&classifiers[ci].path[i].x,sizeof(FloatType));
            classifierfile.read((char*)&classifiers[ci].path[i].y,sizeof(FloatType));
        }
        classifierfile.read((char*)&classifiers[ci].absmaxXY,sizeof(FloatType));
        classifiers[ci].prepare();
    }
    classifierfile.close();
    int nclasses = uniqueClasses.size();
    
    // reversed situation here compared to canupo:
    // - we load the core points in the cloud so as to perform neighbor searches
    // - the data itself is unstructured, not even loaded whole in memory
    ifstream mscfile(argv[3], ifstream::binary);
    // read the file header
    int ncorepoints;
    mscfile.read((char*)&ncorepoints,sizeof(ncorepoints));
    int nscales_msc;
    mscfile.read((char*)&nscales_msc, sizeof(int));
    if (nscales_msc!=nscales) {
        cerr << "Inconsistent combination of multiscale file and classifier parameters (wrong number of scales)" << endl;
        return 1;
    }
    for (int si=0; si<nscales; ++si) {
        FloatType scale_msc;
        mscfile.read((char*)&scale_msc, sizeof(FloatType));
        if (!fpeq(scale_msc, scales[si])) {
            cerr << "Inconsistent combination of multiscale file and classifier parameters (not the same scales)" << endl;
            return 1;
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
            
            
            // TODO : update
            
            
            
            
            vector<FloatType> predictions(nclassifiers);
            map<int,int> votes;
            for (int j=1; j<nclasses; ++j) for (int i=0; i<j; ++i) {
                int cidx = j*(j-1)/2 + i;
                // multi-scale classifier: select relevant scales
                for (int k = 0; k<classifierscalesidx[cidx].size(); ++k) {
                    int sidx = classifierscalesidx[cidx][k];
                    selectedScalesData[k*2] = msc[sidx*2];
                    selectedScalesData[k*2+1] = msc[sidx*2+1];
                }
                FloatType pred = predictors[cidx]->predict(&selectedScalesData[0]);
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
