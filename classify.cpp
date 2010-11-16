#include <iostream>
#include <limits>
#include <fstream>
#include <map>

#include <math.h>

#ifdef CHECK_CLASSIFIER
#include <cairo/cairo.h>
#endif

#include "points.hpp"
#include "linearSVM.hpp"

using namespace std;
using namespace boost;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
classify features.prm scene.xyz scene_core.msc scene_annotated.xyz [perr_use4]\n\
  input: features.prm         # Features computed by the make_features program\n\
  input: scene.xyz            # Point cloud to classify/annotate with each class\n\
                              # Text file, lines starting with #,!,;,// or with\n\
                              # less than 3 numeric values are ignored\n\
                              # If a 4rth value is present (ex: laser intensity) it will be used\n\
                              # in order to discriminate points too close to the decision boundary\n\
                              # See also the dbdist parameter\n\
  input: scene_core.msc       # Multiscale parameters at core points in the scene\n\
                              # This file need only contain the relevant scales for classification\n\
                              # as reported by the make_features program\n\
  input: perr_use4            # Distance from the decision boundary below which to use the additional\n\
                              # information (4rth value). The default is 0 (disable the usage\n\
                              # of extra info) and the value is expressed\n\
                              # as the probability to make a mistake in the classification, then internally\n\
                              # converted to the appropriate distance from the decision boundary\n\
                              # This parameter has no effect if there is no 4rth value in the provided file\n\
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
    
    Point2D refpt_pos, refpt_neg;
    
    vector<FloatType> grid;

    void prepare() {
        // exchange refpt_pos and refpt_neg if necessary, the user may have moved them
        // dot product with (+1,+1) vector gives the classification sign
        if (refpt_pos.x + refpt_pos.y < 0) {
            Point2D tmp = refpt_pos;
            refpt_pos = refpt_neg;
            refpt_neg = tmp;
        }
        if (refpt_pos.x + refpt_pos.y < 0) {
            cerr << "Invalid reference points in the classifier" << endl;
            exit(1);
        }

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
            FloatType norm = sqrt(ld.wx*ld.wx + ld.wy*ld.wy);
            ld.wx /= norm; ld.wy /= norm; ld.c /= norm;
            pathlines.push_back(ld);
        }
    }

    FloatType classify2D_checkcondnum(FloatType a, FloatType b, Point2D& refpt, FloatType& condnumber) {
        Point2D pt(a,b);
        // consider each path line as a mini-classifier
        // the segments pt-refpt_pos  and  that segment line cross
        // iff each classifies the end point of the other in different classes
        // we only need a normal vector in each case, not necessary unit 1, as only the sign counts
        // normal vector <=> homogeneous equa
        LineDef ld;
            FloatType xdelta = refpt.x - a;
        FloatType ydelta = refpt.y - b;
        if (fabs(xdelta) > 1e-3) {
            // y = slope * x + bias
            ld.wy = -1;
            ld.wx = ydelta / xdelta; // slope
            ld.c = b - a * ld.wx;
        } else {
            // just reverse the roles for a quasi-vertical line at x ~ cte
            ld.wx = -1;
            ld.wy = xdelta / ydelta; // is quasi null here, assuming ydelta != 0
            ld.c = a - b * ld.wy;
        }
        FloatType norm = sqrt(ld.wx*ld.wx + ld.wy*ld.wy);
        ld.wx /= norm; ld.wy /= norm; ld.c /= norm;
        Point2D refsegn(ld.wx, ld.wy);
        Point2D refshift = refsegn * ld.c;
        FloatType closestDist = numeric_limits<FloatType>::max();
        int selectedSeg = -1;
        int numcross = 0;
        condnumber = -numeric_limits<FloatType>::max();
        for (int i=0; i<pathlines.size(); ++i) {
            Point2D n(pathlines[i].wx, pathlines[i].wy);
            condnumber = max(condnumber, fabs(n.dot(refsegn)));
            Point2D shift = n * pathlines[i].c;
            
            // Compute whether refpt-pt and that segment cross
            // 1. check whether the given pt and the refpt are on different sides of the classifier line
            bool pathseparates = n.dot(pt + shift) * n.dot(refpt + shift) < 0;
            bool refsegseparates;
            // first and last lines are projected to infinity
            if (i==0) {
                // projection of the end point on ref line
                // Point2D p = path[i+1] - refsegn * refsegn.dot(path[i+1] + refshift);
                // path[i+1] - p = refsegn * refsegn.dot(path[i+1] + refshift);
                // compute whether refsegn * refsegn.dot(path[i+1] + refshift); and path[i+1] - path[i]; are in the same dir
                Point2D to_infinity_and_beyond = path[i+1] - path[i];
                refsegseparates = to_infinity_and_beyond.dot(refsegn * refsegn.dot(path[i+1] + refshift))>0;
            } else if (i==pathlines.size()-1) {
                Point2D to_infinity_and_beyond = path[i] - path[i+1];
                refsegseparates = to_infinity_and_beyond.dot(refsegn * refsegn.dot(path[i] + refshift))>0;
            } else refsegseparates = refsegn.dot(path[i] + refshift) * refsegn.dot(path[i+1] + refshift) < 0;
            // crossing iif each segment/line separates the other
            numcross += refsegseparates && pathseparates;
            
            // closest distance from the point to that segment
            // 1. projection of the point of the line
            Point2D p = pt - n * n.dot(pt + n * pathlines[i].c);
            FloatType closestToSeg = numeric_limits<FloatType>::max();
            bool projwithin = true;
            // 2. Is the projection within the segment limit ? yes => closest
            if (i==0) {
                FloatType xdelta = path[i+1].x - p.x;
                FloatType ydelta = path[i+1].y - p.y;
                // use the more reliable delta
                if (fabs(xdelta)>fabs(ydelta)) {
                    // intersection is valid only if on the half-infinite side of the segment
                    projwithin &= (xdelta * (path[i+1].x - path[i].x)) > 0;
                } else {
                    projwithin &= (ydelta * (path[i+1].y - path[i].y)) > 0;
                }
            } else if (i==pathlines.size()-1) {
                // idem, just infinite on the other side
                FloatType xdelta = path[i].x - p.x;
                FloatType ydelta = path[i].y - p.y;
                if (fabs(xdelta)>fabs(ydelta)) {
                    projwithin &= (xdelta * (path[i].x - path[i+1].x)) > 0;
                } else {
                    projwithin &= (ydelta * (path[i].y - path[i+1].y)) > 0;
                }
            } else {
                // intersection is valid only within the segment boundaries
                projwithin &= (p.x >= min(path[i].x,path[i+1].x)) && (p.y >= min(path[i].y,path[i+1].y));
                projwithin &= (p.x <= max(path[i].x,path[i+1].x)) && (p.y <= max(path[i].y,path[i+1].y));
            }
            if (projwithin) closestToSeg = dist(Point2D(a,b), p);
            else {
                // 3. otherwise closest is the minimum of the distance to the segment ends
                if (i!=0) closestToSeg = dist(Point2D(a,b), Point2D(path[i].x,path[i].y));
                if (i!=pathlines.size()-1) closestToSeg = min(closestToSeg, dist(Point2D(a,b), Point2D(path[i+1].x,path[i+1].y)));
            }
            if (closestToSeg < closestDist) {
                selectedSeg = i;
                closestDist = closestToSeg;
            }
        }
        Point2D n(pathlines[selectedSeg].wx, pathlines[selectedSeg].wy);
        Point2D p = pt - n * n.dot(pt + n * pathlines[selectedSeg].c);
        Point2D delta = pt - p;
        if ((numcross&1)==0) return delta.norm();
        else return -delta.norm();
    }


    // classification in the 2D space
    FloatType classify2D(FloatType a, FloatType b) {
        FloatType condpos, condneg;
        FloatType predpos = classify2D_checkcondnum(a,b,refpt_pos,condpos);
        FloatType predneg = classify2D_checkcondnum(a,b,refpt_neg,condneg);
        // normal nearly aligned = bad conditionning, the lower the dot prod the better
        if (condpos<condneg) return predpos;
        return -predneg;
    }
    
    void project(FloatType* mscdata, FloatType& a, FloatType& b) {
        a = weights_axis1[weights_axis1.size()-1];
        b = weights_axis2[weights_axis2.size()-1];
        for (int d=0; d<weights_axis1.size()-1; ++d) {
            a += weights_axis1[d] * mscdata[d];
            b += weights_axis2[d] * mscdata[d];
        }
    }

    // classification in MSC space
    FloatType classify(FloatType* mscdata) {
        FloatType a,b;
        project(mscdata,a,b);
        return classify2D(a,b);
    }
};


struct ClassifInfo {
    bool reliable;
    int classif;
    ClassifInfo() : reliable(false), classif(-1) {}
};
typedef PointTemplate<ClassifInfo> PointClassif;

int main(int argc, char** argv) {

    if (argc<5) return help();

    FloatType duse4 = 0;
    if (argc>=6) {
        FloatType perr_use4 = atof(argv[5]);
        if (perr_use4<=0 || perr_use4>=0.5) {
            cout << "Disabling usage of extra information" << endl;
            duse4 = 0;
        }
        else duse4 = -log(1.0/(1.0 - perr_use4) - 1.0);
    }
    
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
        for (int i=0; i<=fdim; ++i) classifparamsfile.read((char*)&classifiers[ci].weights_axis1[i],sizeof(FloatType));
        for (int i=0; i<=fdim; ++i) classifparamsfile.read((char*)&classifiers[ci].weights_axis2[i],sizeof(FloatType));
        int pathsize;
        classifparamsfile.read((char*)&pathsize,sizeof(int));
        classifiers[ci].path.resize(pathsize);
        for (int i=0; i<pathsize; ++i) {
            classifparamsfile.read((char*)&classifiers[ci].path[i].x,sizeof(FloatType));
            classifparamsfile.read((char*)&classifiers[ci].path[i].y,sizeof(FloatType));
        }
        classifparamsfile.read((char*)&classifiers[ci].refpt_pos.x,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].refpt_pos.y,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].refpt_neg.x,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].refpt_neg.y,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].absmaxXY,sizeof(FloatType));
        classifiers[ci].prepare();
    }
    classifparamsfile.close();

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
    int ptnparams;
    mscfile.read((char*)&ptnparams, sizeof(int));
    if (ptnparams<3) {
        cerr << "Internal error: Multiscale file does not contain point coordinates" << endl;
        return 1;
    }
    vector<FloatType> coreAdditionalInfo;
    if (ptnparams>=4) coreAdditionalInfo.resize(ncorepoints);

    // now load the points and multiscale information from the msc file.
    // Put the points in the cloud, keep the multiscale information in a separate vector matched by point index
    PointCloud<PointClassif> coreCloud;
    vector<FloatType> mscdata(ncorepoints * nscales*2);
    coreCloud.data.resize(ncorepoints);
    coreCloud.xmin = numeric_limits<FloatType>::max();
    coreCloud.xmax = -numeric_limits<FloatType>::max();
    coreCloud.ymin = numeric_limits<FloatType>::max();
    coreCloud.ymax = -numeric_limits<FloatType>::max();
    for (int pt=0; pt<ncorepoints; ++pt) {
        mscfile.read((char*)&coreCloud.data[pt].x, sizeof(FloatType));
        mscfile.read((char*)&coreCloud.data[pt].y, sizeof(FloatType));
        mscfile.read((char*)&coreCloud.data[pt].z, sizeof(FloatType));
        if (ptnparams>=4) mscfile.read((char*)&coreAdditionalInfo[pt], sizeof(FloatType));
        // forward-compatibility: we do not care for possibly extra parameters for now
        for (int i=4; i<ptnparams; ++i) {
            FloatType param;
            mscfile.read((char*)&param, sizeof(FloatType));
        }
        coreCloud.xmin = min(coreCloud.xmin, coreCloud.data[pt].x);
        coreCloud.xmax = max(coreCloud.xmax, coreCloud.data[pt].x);
        coreCloud.ymin = min(coreCloud.ymin, coreCloud.data[pt].y);
        coreCloud.ymax = max(coreCloud.ymax, coreCloud.data[pt].y);
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
    // complete the coreCloud structure by setting the grid
    FloatType sizex = coreCloud.xmax - coreCloud.xmin;
    FloatType sizey = coreCloud.ymax - coreCloud.ymin;
    
    coreCloud.cellside = sqrt(TargetAveragePointDensityPerGridCell * sizex * sizey / ncorepoints);
    coreCloud.ncellx = floor(sizex / coreCloud.cellside) + 1;
    coreCloud.ncelly = floor(sizey / coreCloud.cellside) + 1;
    
    coreCloud.grid.resize(coreCloud.ncellx * coreCloud.ncelly);
    for (int i=0; i<coreCloud.grid.size(); ++i) coreCloud.grid[i] = 0;
    // setup the grid: list the data points in each cell
    for (int pt=0; pt<ncorepoints; ++pt) {
        int cellx = floor((coreCloud.data[pt].x - coreCloud.xmin) / coreCloud.cellside);
        int celly = floor((coreCloud.data[pt].y - coreCloud.ymin) / coreCloud.cellside);
        coreCloud.data[pt].next = coreCloud.grid[celly * coreCloud.ncellx + cellx];
        coreCloud.grid[celly * coreCloud.ncellx + cellx] = &coreCloud.data[pt];
    }
    
    cout << "Loading scene data" << endl;
    PointCloud<Point> sceneCloud;
    vector<FloatType> sceneAdditionalInfo;
    sceneCloud.load_txt(argv[2], &sceneAdditionalInfo);
    
    cout << "Processing scene data" << endl;
    ofstream scene_annotated(argv[4]);

#ifdef CHECK_CLASSIFIER
    static const int svgSize = 800;
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, svgSize, svgSize);
    cairo_t *cr = cairo_create(surface);
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_set_line_width(cr, 0);
    cairo_rectangle(cr, 0, 0, svgSize, svgSize);
    cairo_fill(cr);
    cairo_stroke(cr);
    cairo_set_line_width(cr, 1);
#endif

    if (duse4>0 && sceneAdditionalInfo.empty()) {
        cout << "Warning: perr_use4 argument is ignored as the scene does not have additional information" << endl;
        duse4 = 0;
    }
    if (duse4>0 && coreAdditionalInfo.empty()) {
        cout << "Warning: perr_use4 argument is ignored as the core point file does not have additional information" << endl;
        duse4 = 0;
    }
    if (duse4<=0 && !sceneAdditionalInfo.empty()) {
        cout << "Warning: ignoring extra information in the scene" << endl;
    }
    if (duse4<=0 && coreAdditionalInfo.empty()) {
        cout << "Warning: ignoring extra information at the core points" << endl;
    }

    // 2-step process:
    // - 1. set the class of all core points that are geometrically >dist from hyperplane
    //      note the remaining core points
    //   - loop while some unclassified core points remain, for each point
    //      - train local SVM using the scene points extra info, if such scene points are associated with "sure / OK" core points
    //        - predict the core point using its own extra info. Put it in the "sure/ok" core points and remove from the unclassified list.
    //      - otherwise (core point in a region where there are only bad core points) keep it for later
    //   - the unsure regions shall shrink by construction for a completely connected scene,
    //     but a test to check that the number of remaining points decrease would be nice... otherwise infinite loop
    // - 2. Now that all core points are OK, classify the scene

    vector<int> idxToSearch(coreCloud.data.size());
    for (int ptidx=0; ptidx<coreCloud.data.size(); ++ptidx) idxToSearch[ptidx] = ptidx;
    vector<int> unreliableCoreIdx;
    // just to check we're not in an infinite loop
    int nidxtosearch = idxToSearch.size();
    do {
#pragma omp parallel for
        for (int itsi=0; itsi<idxToSearch.size(); ++itsi) {
            int ptidx = idxToSearch[itsi];
            map<int,int> votes;
            map< pair<int,int>, FloatType > predictions;
            bool unreliable = false;
            // one-against-one process: apply all classifiers and vote for this point class
            for (int ci=0; ci<nclassifiers; ++ci) {
                FloatType pred = classifiers[ci].classify(&mscdata[ptidx*nscales*2]);
                // uniformize the order, pred>0 selects the larger class of both
                if (classifiers[ci].class1 > classifiers[ci].class2) pred = -pred;
                int minclass = min(classifiers[ci].class1, classifiers[ci].class2);
                int maxclass = max(classifiers[ci].class1, classifiers[ci].class2);
                // use extra info when too close to the decision boundary
                if (fabs(pred)<duse4) {
                    // we've made sure above that both core and scene data have the extra info at this point
                    // largest scale is the first by construction in canupo, order was preserved by the other programs
                    FloatType largestScale = scales[0];
                    vector<DistPoint<Point> > neighbors;
                    vector<int> class1sceneidx;
                    vector<int> class2sceneidx;
                    // find all scene data around that core point
                    sceneCloud.findNeighbors(back_inserter(neighbors), sceneCloud.data[ptidx], largestScale);
                    // for each scene data point, find the corresponding core point and check if it is reliable
                    for (int i=0; i<neighbors.size(); ++i) {
                        int neighcoreidx = coreCloud.findNearest(*neighbors[i].pt);
                        if (neighcoreidx==-1) {
                            cerr << "Invalid core point file" << endl;
                            exit(1);
                        }
                        if (coreCloud.data[neighcoreidx].reliable) {
                            if (coreCloud.data[neighcoreidx].classif == minclass) class1sceneidx.push_back(neighbors[i].pt-&sceneCloud.data[0]);
                            if (coreCloud.data[neighcoreidx].classif == maxclass) class2sceneidx.push_back(neighbors[i].pt-&sceneCloud.data[0]);
                            // else the extra info is irrelevant for this classifier pair
                        }
                    }
                    // some local info ? TODO: min size for considering this information is reliable ?
                    int nsamples = class1sceneidx.size() + class2sceneidx.size();
                    if (nsamples>0) {
                        // only one class ?
                        if (class1sceneidx.size()==0) {
                            if (class2sceneidx.size() * 2 > neighbors.size())
                                pred = class2sceneidx.size() / (FloatType)nsamples;
                            else unreliable = true;
//cout << "only class 2" << endl;
                        } else if (class2sceneidx.size()==0) {
                            if (class1sceneidx.size() * 2 > neighbors.size())
                                pred = -(class1sceneidx.size() / (FloatType)nsamples);
                            else unreliable = true;
//cout << "only class 1" << endl;
                        }
                        else {
/*
                            // nearest neighbor in either class
                            FloatType x = coreAdditionalInfo[ptidx];
                            FloatType dmin = numeric_limits<FloatType>::max();
                            for (int i=0; i<class1sceneidx.size(); ++i) {
                                FloatType d = fabs(sceneAdditionalInfo[class1sceneidx[i]]-x);
                                if (d<dmin) d=dmin;
                            }
                            bool isClass1 = true;
                            for (int i=0; i<class2sceneidx.size(); ++i) {
                                FloatType d = fabs(sceneAdditionalInfo[class2sceneidx[i]]-x);
                                if (d<dmin) {
                                    d=dmin; isClass1 = false;
                                    break; // closer points would only improve the decision, now class2
                                }
                            }
                            if (isClass1) pred = -class1sceneidx.size() / (FloatType)nsamples;
                            else pred = class2sceneidx.size() / (FloatType)nsamples;
*/
                            vector<FloatType> info1(class1sceneidx.size());
                            for (int i=0; i<class1sceneidx.size(); ++i) info1[i] = sceneAdditionalInfo[class1sceneidx[i]];
                            vector<FloatType> info2(class2sceneidx.size());
                            for (int i=0; i<class2sceneidx.size(); ++i) info2[i] = sceneAdditionalInfo[class2sceneidx[i]];
                            sort(info1.begin(), info1.end());
                            sort(info2.begin(), info2.end());
                            vector<FloatType>* smallestvec, * largestvec;
                            if (class1sceneidx.size()<class2sceneidx.size()) {
                                smallestvec = &info1;
                                largestvec = &info2;
                            } else {
                                smallestvec = &info2;
                                largestvec = &info1;
                            }
                            vector<FloatType> bestSplit;
                            vector<int> bestSplitDir;
                            FloatType bestclassif = -1;
                            for (int i=0; i<smallestvec->size(); ++i) {
                                int dichofirst = 0;
                                int dicholast = largestvec->size();
                                int dichomed;
                                while (true) {
                                    dichomed = (dichofirst + dicholast) / 2;
                                    if (dichomed==dichofirst) break;
                                    if (info1[i]==info2[dichomed]) break;
                                    if (info1[i]<info2[dichomed]) { dicholast = dichomed; continue;}
                                    dichofirst = dichomed;
                                }
                                // dichomed is now the last index with info2 below or equal to info1[i],
                                int nlabove = largestvec->size() - 1 - dichomed;
                                int nsbelow = i;
                                // or possibly all if info1[i] is too low
                                if ((*smallestvec)[i]<(*largestvec)[dichomed]) {
                                    // shall happen only if dichomed==0, sorted vecs
                                    assert(dichomed==0);
                                    nlabove = largestvec->size();
                                    nsbelow = i+1;
                                }
                                // classification on either side, take largest and reverse roles if necessary
                                FloatType c1 = nlabove / (FloatType)largestvec->size() + nsbelow / (FloatType)smallestvec->size();
                                FloatType c2 = (largestvec->size()-nlabove) / (FloatType)largestvec->size() + (smallestvec->size()-nsbelow)/ (FloatType)smallestvec->size();
                                FloatType classif = max(c1,c2);
                                // no need to average for comparison purpose
                                if (bestclassif < classif) {
                                    bestSplit.clear(); bestSplitDir.clear();
                                    bestclassif = classif;
                                }
                                if (fpeq(bestclassif,classif)) {
                                    bestSplit.push_back(((*smallestvec)[i]+(*largestvec)[dichomed])*0.5);
                                    bestSplitDir.push_back((int)(c1<=c2));
                                }
                            }
                            bestclassif *= 0.5;
                            // see if we're improving estimated probability or not
                            FloatType oriprob = 1 / (1+exp(-fabs(pred)));
// TODO: sometimes (rarely) there are mistakes in the reference core points and we're dealing with similar classes
// => put back these core points in the unreliable pool
//cout << (oriprob < bestclassif?"OK: ":"NO: ") << bestclassif << " vs " << oriprob << endl;
                            if (oriprob < bestclassif) {
                                // take median best split
                                int bsi = bestSplit.size()/2;
                                pred = coreAdditionalInfo[ptidx] - bestSplit[bsi];
                                // reverse if necessary
                                if (bestSplitDir[bsi]==1) pred = -pred;
                                // back to original vectors
                                if (class1sceneidx.size()>=class2sceneidx.size()) pred = -pred;
                            } else unreliable = true;
                            
                        }
                    }
                    else unreliable = true;
                }
                if (unreliable) break;
                if (pred>=0) ++votes[maxclass];
                else ++votes[minclass];
                predictions[make_pair(minclass, maxclass)] = pred;
            }
            if (unreliable) continue; // no classification

            // search for max vote
            vector<int> bestclasses;
            int maxvote = -1;
            for (map<int,int>::iterator it = votes.begin(); it!=votes.end(); ++it) {
                int vclass = it->first;
                int vote = it->second;
                if (maxvote < vote) {
                    bestclasses.clear();
                    bestclasses.push_back(vclass);
                    maxvote = vote;
                } else if (maxvote == vote) {
                    bestclasses.push_back(vclass);
                }
            }
            // only one class => do not bother with tie breaking
            if (bestclasses.size()==1) coreCloud.data[ptidx].classif = bestclasses[0]; 
            else {
                // in case equality = use the distances from the decision boundary
                map<int, FloatType> votepred;
                // process only once each pair of classes
                for (int j=1; j<bestclasses.size(); ++j) for (int i=0; i<j; ++i) {
                    int minc = min(bestclasses[i], bestclasses[j]);
                    int maxc = max(bestclasses[i], bestclasses[j]);
                    FloatType pred = predictions[make_pair(minc,maxc)];
                    if (pred>=0) votepred[maxc] += pred;
                    else votepred[minc] -= pred; // sum positive contributions
                }
                // now look for the class with max total decision boundary
                FloatType maxpred = -numeric_limits<FloatType>::max();
                int selectedclass = 1;
                for (map<int, FloatType>::iterator it = votepred.begin(); it!=votepred.end(); ++it) {
                    if (maxpred < it->second) {
                        maxpred = it->second;
                        selectedclass = it->first;
                    }
                }
                coreCloud.data[ptidx].classif = selectedclass;
            }
        }
        
        // second phase: mark as reliable all searched points where we could find a classification
        for (int itsi=0; itsi<idxToSearch.size(); ++itsi) {
            int ptidx = idxToSearch[itsi];
            if (coreCloud.data[ptidx].classif!=-1) coreCloud.data[ptidx].reliable = true;
            else unreliableCoreIdx.push_back(ptidx);
        }
        // swap to process still unreliable points
        idxToSearch.clear();
        unreliableCoreIdx.swap(idxToSearch);
        // break infinite loop, some core points and scene data are in unconnected zones we have no info for
        // => these points won't be classified below, attributed class 0
        if (nidxtosearch == idxToSearch.size()) {
            for (int itsi=0; itsi<idxToSearch.size(); ++itsi) coreCloud.data[idxToSearch[itsi]].classif = 0;
            break;
        }
        cout << (coreCloud.data.size()-idxToSearch.size()) << " data classified"<< (nidxtosearch==coreCloud.data.size()?" geometrically":"") <<", " << idxToSearch.size() << " remaining" << (nidxtosearch==coreCloud.data.size()?" using extra information":"") << endl;
        nidxtosearch = idxToSearch.size(); // for next loop
    } while (nidxtosearch>0);

    cout << "Core points classified, labelling scene data" << endl;
    
    for (int pt=0; pt<sceneCloud.data.size(); ++pt) {
        Point& point = sceneCloud.data[pt];
        // process this point
        // first look for the nearest neighbor in core points
        int neighidx = coreCloud.findNearest(point);
        if (neighidx==-1) {
            cerr << "Invalid core point file." << endl;
            return 1;
        }
        // assign the scene point to this core point class, which was computed before
        scene_annotated << point.x << " " << point.y << " " << point.z << " " << coreCloud.data[neighidx].classif << endl;
#ifdef CHECK_CLASSIFIER
        FloatType a,b;
        FloatType scaleFactor = svgSize/2 / classifiers[0].absmaxXY;
        classifiers[0].project(&mscdata[neighidx*nscales*2],a,b);
        if (coreCloud.data[neighidx].classif==1) cairo_set_source_rgba(cr, 0, 0, 1, 0.75);
        else if (coreCloud.data[neighidx].classif==2) cairo_set_source_rgba(cr, 1, 0, 0, 0.75);
        else cairo_set_source_rgba(cr, 0, 1, 0, 0.75);
        FloatType x = a*scaleFactor + svgSize/2;
        FloatType y = svgSize/2 - b*scaleFactor;
        cairo_arc(cr, x, y, 0.714, 0, 2*M_PI);
        cairo_stroke(cr);
#endif
    }

#ifdef CHECK_CLASSIFIER
    FloatType scaleFactor = svgSize/2 / classifiers[0].absmaxXY;
    cairo_set_source_rgb(cr, 0,0,0);
    for (int i=0; i<classifiers[0].path.size(); ++i) {
        FloatType x = classifiers[0].path[i].x*scaleFactor + svgSize/2;
        FloatType y = svgSize/2 - classifiers[0].path[i].y*scaleFactor;
        if (i==0) cairo_move_to(cr, x,y);
        else cairo_line_to(cr, x,y);
    }
    cairo_stroke(cr);
    double dashes[2]; 
    int halfSvgSize = svgSize/2;
    dashes[0] = dashes[1] = svgSize*0.01;
    cairo_set_dash(cr, dashes, 2, svgSize*0.005);
    cairo_set_source_rgb(cr, 0.25,0.25,0.25);
    cairo_move_to(cr, 0,halfSvgSize);
    cairo_line_to(cr, svgSize,halfSvgSize);
    cairo_move_to(cr, halfSvgSize,0);
    cairo_line_to(cr, halfSvgSize,svgSize);
    cairo_stroke(cr);
    cairo_surface_write_to_png (surface, "test.png");
#endif

    return 0;
}
