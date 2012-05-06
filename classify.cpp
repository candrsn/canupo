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
#include <limits>
#include <fstream>
#include <map>

#ifdef CHECK_CLASSIFIER
#include <cairo/cairo.h>
#endif

#include "classifier.hpp"
#include "linearSVM.hpp"

using namespace std;
using namespace boost;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
classify features.prm scene.xyz scene_core.msc scene_annotated.xyz [pok [usage_flag]]\n\
  input: features.prm         # Classifier parameters computed by validate_classifier\n\
  input: scene.xyz            # Point cloud to classify/annotate with each class\n\
                              # Text file, lines starting with #,!,;,// or with\n\
                              # less than 3 numeric values are ignored\n\
                              # If a 4rth value is present (ex: laser intensity) it will be used\n\
                              # in order to discriminate points too close to the decision boundary\n\
                              # See also the dbdist parameter\n\
  input: scene_core.msc       # Multiscale parameters at core points in the scene\n\
                              # This file need only contain the relevant scales for classification\n\
                              # as reported by the make_features program\n\
  output: scene_annotated.xyz # Output file containing extra columns: the class\n\
                              # of each point, the confidence in the classification,\n\
                              # the number of neighbors at the min and max scales\n\
                              # Scene points are labelled with the class of the nearest core point.\n\
  input: pok                  # Some threshold, expressed as a probability to make\n\
                              # a correct classification (0.5<pok<1). Use 0\n\
                              # to disable the threshold, which is also the default\n\
                              # Internally this is converted to a distance from\n\
                              # the decision boundary matching that probability\n\
                              # See the usage_flag argument for what pok means.\n\
  input: usage_flag           # What to do with the perr argument if it is valid.\n\
                              # The default is 0:\n\
                              # - 0: mark as unclassified all points with confidence < pok\n\
                              #      or equivalently too close to the decision boundary\n\
                              # - 1: use the 4rth column in the data file as extra information\n\
                              #      and train a local classifier to complement the confidence\n\
                              #      for points < pok\n\
                              #      This parameter has no effect if there is no 4rth value\n\
                              #      in the provided file\n\
"<<endl;
#ifdef CHECK_CLASSIFIER
cout << "\n\
  # Note: A file named \"classification.png\" will display the result of the first classifier\n\
"<<endl;
#endif
        return 0;
}

bool fpeq(FloatType a, FloatType b) {
    static const FloatType epsilon = 1e-6;
    if (b==0) return fabs(a)<epsilon;
    FloatType ratio = a/b;
    return ratio>1-epsilon && ratio<1+epsilon;
}


struct ClassifInfo {
    bool reliable;
    int classif;
    FloatType confidence;
    ClassifInfo() : reliable(false), classif(-1), confidence(0.5) {}
};
typedef PointTemplate<ClassifInfo> PointClassif;

int main(int argc, char** argv) {

    if (argc<5) return help();

    FloatType dist_to_decision_boundary = 0;
    int usage_flag = 0;
    if (argc>=6) {
        FloatType pok = atof(argv[5]);
        if (pok<0.5 || pok>=1) {
            cout << "Invalid pok argument" << endl;
            dist_to_decision_boundary = 0;
        }
        else if (pok!=0) dist_to_decision_boundary = -log(1.0/pok - 1.0);
        if (argc>=7) {
            usage_flag = atoi(argv[6]);
        }
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
        classifparamsfile.read((char*)&classifiers[ci].axis_scale_ratio,sizeof(FloatType));
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
        cerr << "Scales in the classifier file:";
        for (int si=0; si<nscales; ++si) cerr << " " << scales[si];
        cerr << endl << "Scales in the multiscale file:";
        for (int si=0; si<nscales_msc; ++si) {
            FloatType scale_msc;
            mscfile.read((char*)&scale_msc, sizeof(FloatType));
            cerr << " " << scale_msc;
        }
        cerr << endl;
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
    // extract only min/max scales neighbor stats for output file
    vector<int> nneigh_max_scale(ncorepoints);
    vector<int> nneigh_min_scale(ncorepoints);
    //vector<FloatType> avg_ndist_max_scale(ncorepoints);
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
        // we care only for number of neighbors at max and min scales
        mscfile.read((char*)&nneigh_max_scale[pt], sizeof(int));
        int numneigh;
        for (int i=1; i<nscales; ++i) mscfile.read((char*)&numneigh, sizeof(int));
        nneigh_min_scale[pt] = numneigh;
/*        FloatType foof;
        mscfile.read((char*)&avg_ndist_max_scale[pt], sizeof(FloatType));
        for (int i=1; i<nscales; ++i) mscfile.read((char*)&foof, sizeof(FloatType));*/
    }
    mscfile.close();
    // complete the coreCloud structure by setting the grid
    FloatType sizex = coreCloud.xmax - coreCloud.xmin;
    FloatType sizey = coreCloud.ymax - coreCloud.ymin;
    
    coreCloud.cellside = sqrt(TargetAveragePointDensityPerGridCell * sizex * sizey / ncorepoints);
    coreCloud.ncellx = floor(sizex / coreCloud.cellside) + 1;
    coreCloud.ncelly = floor(sizey / coreCloud.cellside) + 1;
    
    coreCloud.grid.resize(coreCloud.ncellx * coreCloud.ncelly);
    coreCloud.links.resize(ncorepoints);
    for (int i=0; i<ncorepoints; ++i) coreCloud.links[i] = IndexType(-1);
    for (int i=0; i<coreCloud.grid.size(); ++i) coreCloud.grid[i] = IndexType(-1);
    // setup the grid: list the data points in each cell
    for (int pt=0; pt<ncorepoints; ++pt) {
        int cellx = floor((coreCloud.data[pt].x - coreCloud.xmin) / coreCloud.cellside);
        int celly = floor((coreCloud.data[pt].y - coreCloud.ymin) / coreCloud.cellside);
        coreCloud.links[pt] = coreCloud.grid[celly * coreCloud.ncellx + cellx];
        coreCloud.grid[celly * coreCloud.ncellx + cellx] = pt;
    }
    
    cout << "Loading scene data" << endl;
    PointCloud<Point> sceneCloud;
    vector<vector<FloatType> > sceneAdditionalInfo;
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
            map<int,FloatType> minconfidences;
            bool unreliable = false;
            // one-against-one process: apply all classifiers and vote for this point class
            for (int ci=0; ci<nclassifiers; ++ci) {
                FloatType pred = classifiers[ci].classify(&mscdata[ptidx*nscales*2]);
                // uniformize the order, pred>0 selects the larger class of both
                if (classifiers[ci].class1 > classifiers[ci].class2) pred = -pred;
                int minclass = min(classifiers[ci].class1, classifiers[ci].class2);
                int maxclass = max(classifiers[ci].class1, classifiers[ci].class2);
                // use extra info when too close to the decision boundary
                if (fabs(pred)<dist_to_decision_boundary && usage_flag==0) {
                    unreliable = true;
                }
                else if (fabs(pred)<dist_to_decision_boundary && usage_flag==1) {
                    // we've made sure above that both core and scene data have the extra info at this point
                    // largest scale is the first by construction in canupo, order was preserved by the other programs
                    FloatType largestScale = scales[0];
                    vector<DistPoint<Point> > neighbors;
                    vector<int> class1sceneidx;
                    vector<int> class2sceneidx;
                    // find all scene data around that core point
                    sceneCloud.findNeighbors(back_inserter(neighbors), sceneCloud.data[ptidx], largestScale * 0.5); // take radius, not diameter
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
                            for (int i=0; i<class1sceneidx.size(); ++i) info1[i] = sceneAdditionalInfo[class1sceneidx[i]][0];
                            vector<FloatType> info2(class2sceneidx.size());
                            for (int i=0; i<class2sceneidx.size(); ++i) info2[i] = sceneAdditionalInfo[class2sceneidx[i]][0];
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
                FloatType confidence = 1.0 / (exp(-fabs(pred))+1.0);
                int theclass = minclass;
                if (pred>=0) theclass = maxclass;
                ++votes[theclass];
                // simply maintain the min confidence for each class
                // and we'll use that for the best vote below
                // for our application this is enough
                if (minconfidences.find(theclass)==minconfidences.end()) {
                    minconfidences[theclass] = confidence;
                } else {
                    if (confidence<minconfidences[theclass]) minconfidences[theclass] = confidence;
                }
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
            if (bestclasses.size()==1) {
                coreCloud.data[ptidx].classif = bestclasses[0]; 
                coreCloud.data[ptidx].confidence = minconfidences[bestclasses[0]];
            }
            else {
                // in case equality = use the distances from the decision boundary
                // take the max vote class that has also farthest min dist
                FloatType max_minc = minconfidences[bestclasses[0]];
                int selectedclass = bestclasses[0];
                for (int j=1; j<bestclasses.size(); ++j) {
                    if (minconfidences[bestclasses[j]]>max_minc) {
                        max_minc = minconfidences[bestclasses[j]];
                        selectedclass = bestclasses[j];
                    }
                }
                coreCloud.data[ptidx].classif = selectedclass;
                coreCloud.data[ptidx].confidence = max_minc;
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
        // also break when explicitly marking close points are unreliable
        if (nidxtosearch == idxToSearch.size() || usage_flag==0) {
            for (int itsi=0; itsi<idxToSearch.size(); ++itsi) coreCloud.data[idxToSearch[itsi]].classif = 0;
            break;
        }
        cout << (coreCloud.data.size()-idxToSearch.size()) << " data classified"<< (nidxtosearch==coreCloud.data.size()?" geometrically":"") <<", " << idxToSearch.size() << " remaining" << (nidxtosearch==coreCloud.data.size()?" using extra information":"") << endl;
        nidxtosearch = idxToSearch.size(); // for next loop
    } while (nidxtosearch>0);

    cout << "Core points classified, labelling scene data" << endl;
    cout << "Output file contains for each point a line with the following values:" << endl;
    cout << "x  y  z  class  confidence num_neighbors_min_scale num_neighbors_max_scale ";// avg_dist_nearest_neighbors_max_scale";
    if (!coreAdditionalInfo.empty()) cout << " extra_info";
    cout << endl;
    cout << "The first 3 values are those of the scene point (x,y,z), the other values are taken from the nearest core point to this scene point" << endl;

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
        scene_annotated << point.x << " " << point.y << " " << point.z;
        scene_annotated << " " << coreCloud.data[neighidx].classif;
        scene_annotated << " " << coreCloud.data[neighidx].confidence;
        scene_annotated << " " << nneigh_min_scale[neighidx];
        scene_annotated << " " << nneigh_max_scale[neighidx];
        //scene_annotated << " " << avg_ndist_max_scale[neighidx];
        if (!coreAdditionalInfo.empty()) scene_annotated << " " << coreAdditionalInfo[neighidx];
        scene_annotated << endl;
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
    scene_annotated.close();

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
    cairo_surface_write_to_png (surface, "classification.png");
#endif

    return 0;
}
