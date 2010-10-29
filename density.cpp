#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <set>
#include <limits>
#include <algorithm>

#include <stdlib.h>
#include <string.h>

#include "points.hpp"
#include "predictors.hpp"

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

string scaleColorMap(const vector<int>& density, const vector<FloatType>& lmind, const vector<FloatType>& lmaxd) {
    // log transform. min>=0 by construction, so add 1 to take log
    if (density.size()==1) {
        // single class : from blue to red
        FloatType d = (log(density[0]+1) - lmind[0]) / (lmaxd[0] - lmind[0]);
        // low value = blue(hue=4/6), high = red(hue=0)
        return hueToRGBstring(FloatType(4)/FloatType(6)*(1-d));
    }

    // one color per class, density is the amount of that color
    // pb: null density would be black: better if it is white for graphs in papers
    //     => interpolate in complement space, then take complement again
    static const FloatType classColors[] = {
        1,1,0,  // class 0 = blue complement
        0,1,1,  // class 1 = red complement
        1,0,1,  // class 2 = green complement
        0,1,0,  // class 3 = magenta complement
        0,0,1,  // class 4 = yellow complement
        1,0,0   // class 5 = cyan complement
    };
    int nclasses = density.size();
    if (nclasses>6) {
        cerr << "Sorry, displaying more than 6 classes is not supported for now" << endl;
        exit(1);
    }
    FloatType color[] = {0,0,0};
    // convert the densities to log space and between 0 and 1 for each class first
    for (int i=0; i<nclasses; ++i) {
        FloatType coef = (log(density[i]+1) - lmind[i]) / (lmaxd[i] - lmind[i]);
        for (int j=0; j<3; ++j) color[j] += coef * classColors[i*3+j];
    }
    // bound check and take complement
    for (int j=0; j<3; ++j) color[j] = 1 - min(1.0, max(color[j], 0.0));
    char ret[8];
    snprintf(ret,8,"#%02X%02X%02X",(int)(color[0]*255.99),(int)(color[1]*255.99),(int)(color[2]*255.99));
    return ret;
}


int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
density nsubdiv nametag [some scales] : [features.prm : ] data.msc [ - data2.msc ...]\n\
  input: nsubdiv               # Number of subdivisions on each side of the triangle\n\
  input: nametag               # The base name for the output files. One density plot is\n\
                               # generated per selected scale, named \"nametag_scale.svg\"\n\
  input: some scales           # Selected scales at which to perform the density plot\n\
                               # All scales in the parameter file are used if not specified.\n\
  input: data.msc              # The multiscale parameters computed by canupo.\n\
                               # Use - to separate classes. Multiple files per class are allowed.\n\
                               # If no classes are specified (ex: whole scene file) the density is color-coded from blue to red.\n\
                               # If multiple classes are specified the density is coded from light to bright colors with one color per class.\n\
  input: features.prm          # An optional classifier definition file. If it is specified the decision\n\
                               # boundaries at each scale will be displayed in the generated graphs.\n\
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
    
    if (scalesSet.empty()) return help();
    vector<FloatType> scales(scalesSet.begin(), scalesSet.end());
    int nscales = scales.size();

    cout << "Selected scales:";
    for (int i=0; i<nscales; ++i) cout << " " << scales[i];
    cout << endl;

    // Is there another separator and a feature file in between ?
    int separator2 = 0;
    for (int i=separator+1; i<argc; ++i) if (!strcmp(":",argv[i])) {
        separator2 = i;
        break;
    }
    string classifierfilename;
    if (separator2) {
        if (separator2 - separator != 2) return help();
        classifierfilename = argv[separator+1];
        separator = separator2;        
    }

    // now process the multiscale files and possibly multiple classes
    int nclasses = 1;
    vector<int> classboundaries(1,0);
    int total_pts = 0;

    // read headers and ensures all files are consistent
    for (int argi = separator+1; argi<argc; ++argi) {
        if (!strcmp("-",argv[argi])) {
            ++nclasses;
            classboundaries.push_back(total_pts);
            continue;
        }
        ifstream mscfile(argv[argi], ifstream::binary);
        // read the file header
        int npts;
        mscfile.read((char*)&npts,sizeof(npts));
        if (npts<=0) help("invalid file");
        
        int nscales_thisfile;
        mscfile.read((char*)&nscales_thisfile, sizeof(nscales_thisfile));
        vector<FloatType> scales_thisfile(nscales_thisfile);
        for (int si=0; si<nscales_thisfile; ++si) mscfile.read((char*)&scales_thisfile[si], sizeof(FloatType));
        if (nscales_thisfile<=0) help("invalid file");
        
        // all files must be contain at least the selected scales
        for (int si=0; si<nscales; ++si) {
            bool scalefound = false;
            for (int sitf=0; sitf<nscales_thisfile; ++sitf) if (fpeq(scales[si],scales_thisfile[sitf])) {
                scalefound = true;
                break;
            }
            if (!scalefound) {
                cerr<<"input file mismatch: "<<argv[argi]<< " does not contain the selected scale " << scales[si] << endl; 
                return 1;
            }
        }
        mscfile.close();
        total_pts += npts;
    }
    classboundaries.push_back(total_pts);
    
    // number of features
    int fdim = nscales * 2;

    // if a classifier was specified, load it and ensure it is consistent
    int nclassifiers = 0;
    vector<vector<FloatType> > classifierscales;
    vector<shared_ptr<Predictor> > predictors;
    
    if (!classifierfilename.empty()) {
        ifstream classifparamsfile(classifierfilename.c_str(), ifstream::binary);
        int nuniqueScales;
        classifparamsfile.read((char*)&nuniqueScales, sizeof(nuniqueScales));
        vector<FloatType> uniquescales(nuniqueScales);
        for (int s=0; s<nuniqueScales; ++s) classifparamsfile.read((char*)&uniquescales[s], sizeof(FloatType));
        int nclasses_classifier_file;
        classifparamsfile.read((char*)&nclasses_classifier_file, sizeof(nclasses));
        if (nclasses_classifier_file != nclasses) {
            cerr << "The classifier file was designed to handle " << nclasses_classifier_file << " classes, but " << nclasses << " classes were provided." << endl;
            return 1;
        }
        nclassifiers = nclasses * (nclasses-1) / 2;
        int classifierID;
        classifparamsfile.read((char*)&classifierID, sizeof(classifierID));        
        classifierscales.resize(nclassifiers);
        predictors.resize(nclassifiers);
        for (int i=0; i<nclassifiers; ++i) {
            int numscales;
            classifparamsfile.read((char*)&numscales, sizeof(numscales));
            classifierscales[i].resize(numscales);
            for (int j=0; j<numscales; ++j) classifparamsfile.read((char*)&classifierscales[i][j], sizeof(FloatType));
            predictors[i] = getPredictorFromClassifierID(classifierID);
            predictors[i]->load(classifparamsfile);
        }
    }
    
    // Second pass: store all selected scale data in memory
    vector<FloatType> data(total_pts * fdim);
    int base_pt = 0;
    // and then fill data from the files
    for (int argi = separator+1; argi<argc; ++argi) {
        if (!strcmp("-",argv[argi])) continue;
        ifstream mscfile(argv[argi], ifstream::binary);
        // read the file header (again)
        int npts;
        mscfile.read((char*)&npts,sizeof(npts));
        int nscales_thisfile;
        mscfile.read((char*)&nscales_thisfile, sizeof(nscales_thisfile));
        vector<FloatType> scales_thisfile(nscales_thisfile);
        for (int si=0; si<nscales_thisfile; ++si) mscfile.read((char*)&scales_thisfile[si], sizeof(FloatType));
        
        // now fill in the big data storage with the points
        for (int pt=0; pt<npts; ++pt) {
            FloatType coord; // we do not care for the point coordinates
            mscfile.read((char*)&coord, sizeof(coord));
            mscfile.read((char*)&coord, sizeof(coord));
            mscfile.read((char*)&coord, sizeof(coord));
            // must read all scales contained in the file
            vector<FloatType> mscdata(nscales_thisfile*2);
            for (int s=0; s<nscales_thisfile; ++s) {
                FloatType a,b;
                mscfile.read((char*)(&a), sizeof(FloatType));
                mscfile.read((char*)(&b), sizeof(FloatType));
                mscdata[s*2] = a;
                mscdata[s*2+1] = b;
            }
            // now retain only selected scales
            for (int si=0; si<nscales; ++si) {
                int scalefound = -1;
                for (int sitf=0; sitf<nscales_thisfile; ++sitf) if (fpeq(scales[si],scales_thisfile[sitf])) {
                    scalefound = sitf;
                    break;
                }
                //assert(scalefound!=-1);
                data[(base_pt+pt)*fdim + si*2] = mscdata[scalefound*2];
                data[(base_pt+pt)*fdim + si*2+1] = mscdata[scalefound*2+1];
            }
        }
        mscfile.close();
        base_pt += npts;
    }
    

    // one density entry per selected scale per class - init all counts to 0
    vector<vector<vector<int> > > density(nscales, vector<vector<int> >(nsubdiv*(nsubdiv+1), vector<int>(nclasses,0) ));
    
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
            ++density[si][((cellx * (cellx+1) / 2) + celly) * 2 + lower][ci];
        }
        
    }

    
    for (int si=0; si<nscales; ++si) {
        vector<int> minDensity(nclasses, numeric_limits<int>::max());
        vector<int> maxDensity(nclasses, 0);
    
        for (vector<vector<int> >::iterator it = density[si].begin(); it != density[si].end(); ++it) {
            for (int ci = 0; ci < nclasses; ++ci) {
                minDensity[ci] = min(minDensity[ci], (*it)[ci]);
                maxDensity[ci] = max(maxDensity[ci], (*it)[ci]);
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
            string color = scaleColorMap(density[si][(x*(x+1)/2+ y)*2],logMinDensity,logMaxDensity);
            densityfile << "\" style=\"fill:" << color << "; stroke:none;\"/>" << endl;
            if (y<x) { // upper cell
                densityfile << "<polygon points=\"";
                densityfile << " " << (x - 0.5*y)*scaleFactor << "," << top-(0.866025403784439 * y)*scaleFactor;
                densityfile << " " << (x+1 - 0.5*(y+1))*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
                densityfile << " " << (x - 0.5*(y+1)
                )*scaleFactor << "," << top-(0.866025403784439 * (y+1))*scaleFactor;
                color = scaleColorMap(density[si][(x*(x+1)/2+ y)*2+1],logMinDensity,logMaxDensity);
                densityfile << "\" style=\"fill:" << color << "; stroke:none;\"/>" << endl;
            }
        }
        densityfile << "<polygon points=\" 0,"<<top<<" "<<scaleFactor*nsubdiv*0.5<<",0 " << scaleFactor*nsubdiv<<","<<top<<" \" style=\"fill:none;stroke:#000000;stroke-width:1px;\"/>" << endl;
        
        // plot classifiers decision boundaries, if any
        if (nclassifiers!=0) for (int class2=1; class2<nclasses; ++class2) for (int class1=0; class1<class2; ++class1) {
            int cli = class2*(class2-1)/2 + class1;
            int cliscaleidx = -1;
            for (int i=0; i<classifierscales[cli].size(); ++i) if (fpeq(scales[si],classifierscales[cli][i])) cliscaleidx = i;
            if (cliscaleidx==-1) continue; // scale not used for this classifier
            const FloatType sf = scaleFactor*nsubdiv;
            LinearPredictor* lp = dynamic_cast<LinearPredictor*>(predictors[cli].get());
            if (lp) {
                FloatType bias = lp->weights[lp->dim];
                FloatType xw = lp->weights[cliscaleidx*2];
                FloatType yw = lp->weights[cliscaleidx*2+1];
                // Compute first the intersections in xy space, then scale and reverse top/down
                // Intersection with triangle bottom line (1D-2D axis)
                FloatType x12 = numeric_limits<FloatType>::max();
                FloatType y12 = 0;
                if (xw!=0) x12 = -bias / xw;
                bool use12 = x12>=0 && x12<=1;
                FloatType x13 = numeric_limits<FloatType>::max();
                FloatType y13 = numeric_limits<FloatType>::max();
                FloatType tmp = xw + sqrt3 * yw;
                if (tmp!=0) {
                    x13 = -bias / tmp;
                    y13 = sqrt3 * x13;
                }
                bool use13 = x13>=0 && x13<=0.5 && y13>=0 && y13<=sqrt3*0.5;
                FloatType x23 = numeric_limits<FloatType>::max();
                FloatType y23 = numeric_limits<FloatType>::max();
                tmp = sqrt3 * yw - xw;
                if (tmp!=0) {
                    x23 = (bias + sqrt3 * yw) / tmp;
                    y23 = sqrt3 - sqrt3 * x23;
                }
                bool use23 = x23>=0.5 && x23<=1 && y23>=0 && y23<=sqrt3*0.5;
                // the coordinates of the line we'll retain
                FloatType xl1, yl1, xl2, yl2;
                if (use12 && use23) {xl1 = x12; yl1 = y12; xl2 = x23; yl2 = y23; cout << "12-23" << endl;}
                else if (use13 && use23) {xl1 = x13; yl1 = y13; xl2 = x23; yl2 = y23; cout << "13-23" << endl;}
                else if (use12 && use13) {xl1 = x12; yl1 = y12; xl2 = x13; yl2 = y13; cout << "12-13" << endl;}
                else {
                    cout << "hyperplane not intersecting scale " << scales[si] << " relevant discriminant area" << endl;
                    continue;
                }
                // TODO: color per classifier
                densityfile << "<path style=\"fill:none;stroke:#000000;stroke-width:1px;\" ";
                densityfile << "d=\"M " << sf*xl1 << "," << top - sf*yl1;
                densityfile << " L " << sf*xl2 << "," << top - sf*yl2;
                densityfile << "\" id=\"classif-" << (class1+1) << "-" << (class2+1) << "\" />";
                continue;
            }
            // TODO: user-defined, piecewise-linear, classifiers
            
            // generic classifier type (including gaussian)
            // interpolate between cells
            // choose a better resolution than the cells
            int segres = nsubdiv * 4; // 16 times more elements in 2D
            vector<FloatType> mscdata(classifierscales[cli].size()*2, 0);
            bool scaleUsed = false;
            for (int segi = 1; segi <= segres; ++segi) {
                vector<FloatType> preds(segi+1);
                vector<FloatType> x(segi+1);
                vector<FloatType> y(segi+1);
                for (int segj = 0; segj <= segi; ++segj) {
                    // horizontal, start from (1/2, sqrt3/2) down to (0,0)
                    mscdata[cliscaleidx*2] = x[segj] = 0.5 - 0.5 * segi / (FloatType) segres + segj / (FloatType) segres;
                    mscdata[cliscaleidx*2+1] = y[segj] = 0.5 * sqrt3 - 0.5 * sqrt3 * segi / (FloatType) segres;
                    preds[segj] = predictors[cli]->predict(&mscdata[0]);
                }
                for (int segj = 0; segj < segi; ++segj) {
                    // changing sign indicates this segment crosses the decision boundary
                    if (preds[segj] * preds[segj+1] < 0) {
                        scaleUsed = true;
                        densityfile << "<path style=\"fill:none;stroke:#000000;stroke-width:1px;\" ";
                        densityfile << "d=\"M " << sf*x[segj] << "," << top - sf*y[segj];
                        densityfile << " L " << sf*x[segj+1] << "," << top - sf*y[segj+1];
                        densityfile << "\" id=\"classif-" << (class1+1) << "-" << (class2+1) << "-h" << segi<<"-"<<segj<<"\" />";
                    }
                }
                for (int segj = 0; segj <= segi; ++segj) {
                    // slanted right, start from (1,0) to (0,0) then up-right-wards
                    mscdata[cliscaleidx*2] = x[segj] = 1 - segi / (FloatType) segres + segj * 0.5 / (FloatType) segres;
                    mscdata[cliscaleidx*2+1] = y[segj] = 0.5 * sqrt3 * segj / (FloatType) segres;
                    preds[segj] = predictors[cli]->predict(&mscdata[0]);
                }
                for (int segj = 0; segj < segi; ++segj) {
                    // changing sign indicates this segment crosses the decision boundary
                    if (preds[segj] * preds[segj+1] < 0) {
                        scaleUsed = true;
                        densityfile << "<path style=\"fill:none;stroke:#000000;stroke-width:1px;\" ";
                        densityfile << "d=\"M " << sf*x[segj] << "," << top - sf*y[segj];
                        densityfile << " L " << sf*x[segj+1] << "," << top - sf*y[segj+1];
                        densityfile << "\" id=\"classif-" << (class1+1) << "-" << (class2+1) << "-sr" << segi<<"-"<<segj<<"\" />";
                    }
                }
                for (int segj = 0; segj <= segi; ++segj) {
                    // slanted left, start from (0,0) to (1,0) then up-left-wards
                    mscdata[cliscaleidx*2] = x[segj] = segi / (FloatType) segres - segj * 0.5 / (FloatType) segres;
                    mscdata[cliscaleidx*2+1] = y[segj] = 0.5 * sqrt3 * segj / (FloatType) segres;
                    preds[segj] = predictors[cli]->predict(&mscdata[0]);
                }
                for (int segj = 0; segj < segi; ++segj) {
                    // changing sign indicates this segment crosses the decision boundary
                    if (preds[segj] * preds[segj+1] < 0) {
                        scaleUsed = true;
                        densityfile << "<path style=\"fill:none;stroke:#000000;stroke-width:1px;\" ";
                        densityfile << "d=\"M " << sf*x[segj] << "," << top - sf*y[segj];
                        densityfile << " L " << sf*x[segj+1] << "," << top - sf*y[segj+1];
                        densityfile << "\" id=\"classif-" << (class1+1) << "-" << (class2+1) << "-sl" << segi<<"-"<<segj<<"\" />";
                    }
                }
            }
            if (!scaleUsed) {
                cout << "scale " << scales[si] << " is not a relevant discriminant scale for the given classifier" << endl;
                continue;
            }
        }
        
        densityfile << "</svg>" << endl;
        densityfile.close();
        cout << "Density plot for scale " << scales[si] << " written in file " << filename.str() << endl;
    }

    return 0;
}
