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
    
    Point2D refpt_pos, refpt_neg;
    
    vector<FloatType> grid;

    void prepare() {
        absmaxXY=5.20822;
        
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

    // now load the points and multiscale information from the msc file.
    // Put the points in the cloud, keep the multiscale information in a separate vector matched by point index
    vector<FloatType> mscdata(ncorepoints * nscales*2);
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

    static const int svgSize = 800;
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, svgSize, svgSize);
    cairo_t *cr = cairo_create(surface);
    
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_set_line_width(cr, 0);
    cairo_rectangle(cr, 0, 0, svgSize, svgSize);
    cairo_fill(cr);
    cairo_stroke(cr);
    
    cairo_set_line_width(cr, 1);
    
    
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
            //vector<FloatType> predictions(nclassifiers);
            map<int,int> votes;
            map< pair<int,int>, FloatType > predictions;
            // one-against-one process: apply all classifiers and vote for this point class
            for (int ci=0; ci<nclassifiers; ++ci) {
                FloatType pred = classifiers[ci].classify(&mscdata[neighidx*nscales*2]);
                if (pred>=0) ++votes[classifiers[ci].class2];
                else ++votes[classifiers[ci].class1];
                // uniformize the order, pred>0 selects the larger class of both
                if (classifiers[ci].class1 > classifiers[ci].class2) pred = -pred;
                predictions[make_pair(min(classifiers[ci].class1, classifiers[ci].class2), max(classifiers[ci].class1, classifiers[ci].class2))] = pred;
            }
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
            if (bestclasses.size()==1) coreclasses[neighidx] = bestclasses[0]; 
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
                coreclasses[neighidx] = selectedclass;
            }
        
            FloatType a,b;
            FloatType scaleFactor = svgSize/2 / classifiers[0].absmaxXY;
            classifiers[0].project(&mscdata[neighidx*nscales*2],a,b);
            if (coreclasses[neighidx]==2) cairo_set_source_rgba(cr, 1, 0, 0, 0.75);
            else cairo_set_source_rgba(cr, 0, 0, 1, 0.75);
            FloatType x = a*scaleFactor + svgSize/2;
            FloatType y = svgSize/2 - b*scaleFactor;
            cairo_arc(cr, x, y, 0.714, 0, 2*M_PI);
            cairo_stroke(cr);
        }
        // assign the scene point to this core point class
        scene_annotated << point.x << " " << point.y << " " << point.z << " " << coreclasses[neighidx] << endl;
    }

    FloatType scaleFactor = svgSize/2 / classifiers[0].absmaxXY;
    cairo_set_source_rgb(cr, 0,0,0);
    for (int i=0; i<classifiers[0].path.size(); ++i) {
        FloatType x = classifiers[0].path[i].x*scaleFactor + svgSize/2;
        FloatType y = svgSize/2 - classifiers[0].path[i].y*scaleFactor;
        if (i==0) cairo_move_to(cr, x,y);
        else cairo_line_to(cr, x,y);
    }
    cairo_stroke(cr);

    // draw lines on top of points
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
    
    return 0;
}
