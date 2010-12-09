#include <iostream>
#include <fstream>
#include <vector>

#include "points.hpp"

using namespace std;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
combine_classifiers  multiclass.prm  classifier1.prm  classifier2.prm [...] \n\
    produce a multiclass prm file by combining the two-class\n\
    classifiers that were previously validated for the class\n\
    numbers that were specified in validate_classifier\n\
"<<endl;
        return 0;
}

bool fpeq(FloatType a, FloatType b) {
    static const FloatType epsilon = 1e-6;
    if (b==0) return fabs(a)<epsilon;
    FloatType ratio = a/b;
    return ratio>1-epsilon && ratio<1+epsilon;
}

struct BinaryClassifier {
    vector<FloatType> scales;
    int class_num_1, class_num_2;
    vector<FloatType> weights_axis1, weights_axis2;
    vector<Point2D> path;
    Point2D refpt1, refpt2;
    FloatType absmaxXY;
};

int main(int argc, char** argv) {
    
    if (argc<2) return help();

    ofstream classifierfile(argv[1], ofstream::binary);
    
    vector<BinaryClassifier> classifiers;
    
    vector<FloatType> scales;

    for (int argi=2; argi<argc; ++argi) {
        ifstream inputclassifier(argv[argi], ifstream::binary);
        
        int nscales = 0;
        inputclassifier.read((char*)&nscales,sizeof(int));
        vector<FloatType> this_scales(nscales);
        for (int i=0; i<nscales; ++i) inputclassifier.read((char*)&this_scales[i],sizeof(FloatType));
        if (scales.empty()) scales = this_scales;
        else if (scales.size() != this_scales.size()) {
            cerr << "Inconsistent number of classifier scales (argument " << argi << ")" << endl;
            return 1;
        }
        for (int i=0; i<nscales; ++i) if (!fpeq(this_scales[i],scales[i])) {
            cerr << "Inconsistent classifier scale values (argument " << argi << ")" << endl;
            return 1;
        }        

        // number of classifiers embedded in this parameter file
        int nclassifiers;
        inputclassifier.read((char*)&nclassifiers,sizeof(int));
        int fdim = nscales * 2;

        for (int cidx = 0; cidx < nclassifiers; ++cidx) {
            BinaryClassifier bc;
            bc.scales = scales;
            
            inputclassifier.read((char*)&bc.class_num_1,sizeof(int));
            inputclassifier.read((char*)&bc.class_num_2,sizeof(int));
            bc.weights_axis1.resize(fdim+1);
            bc.weights_axis2.resize(fdim+1);
            // the first two directions maximizing the separability of the data
            for (int i=0; i<=fdim; ++i) inputclassifier.read((char*)&bc.weights_axis1[i],sizeof(FloatType));
            for (int i=0; i<=fdim; ++i) inputclassifier.read((char*)&bc.weights_axis2[i],sizeof(FloatType));
            // the list of points in the path (scaled in the 2D space)
            int pathsize;
            inputclassifier.read((char*)&pathsize,sizeof(int));
            bc.path.resize(pathsize);
            for(int i=0; i<bc.path.size(); ++i) {
                inputclassifier.read((char*)&bc.path[i].x,sizeof(FloatType));
                inputclassifier.read((char*)&bc.path[i].y,sizeof(FloatType));
            }
            // the reference points for each class
            inputclassifier.read((char*)&bc.refpt1.x,sizeof(FloatType));
            inputclassifier.read((char*)&bc.refpt1.y,sizeof(FloatType));
            inputclassifier.read((char*)&bc.refpt2.x,sizeof(FloatType));
            inputclassifier.read((char*)&bc.refpt2.y,sizeof(FloatType));
            // some information useful for debugging
            inputclassifier.read((char*)&bc.absmaxXY,sizeof(FloatType));
            classifiers.push_back(bc);
        }
        
        inputclassifier.close();
    }
    
    // New classifier format
    int nscales = scales.size();
    int fdim = nscales * 2;
    // scales are for checking compatibility of the scene file
    classifierfile.write((char*)&nscales,sizeof(int));
    for (int i=0; i<nscales; ++i) classifierfile.write((char*)&scales[i],sizeof(FloatType));
    // number of classifiers embedded in this parameter file
    int nclassifiers=classifiers.size();
    classifierfile.write((char*)&nclassifiers,sizeof(int));
    for (int cidx = 0; cidx < nclassifiers; ++cidx) {
        // classes handled by this partial classifier, as numbered by the user
        classifierfile.write((char*)&classifiers[cidx].class_num_1,sizeof(int));
        classifierfile.write((char*)&classifiers[cidx].class_num_2,sizeof(int));
        // the first two directions maximizing the separability of the data
        for (int i=0; i<=fdim; ++i) classifierfile.write((char*)&classifiers[cidx].weights_axis1[i],sizeof(FloatType));
        for (int i=0; i<=fdim; ++i) classifierfile.write((char*)&classifiers[cidx].weights_axis2[i],sizeof(FloatType));
        // the list of points in the path (scaled in the 2D space)
        int pathsize = classifiers[cidx].path.size();
        classifierfile.write((char*)&pathsize,sizeof(int));
        for(int i=0; i<classifiers[cidx].path.size(); ++i) {
            classifierfile.write((char*)&classifiers[cidx].path[i].x,sizeof(FloatType));
            classifierfile.write((char*)&classifiers[cidx].path[i].y,sizeof(FloatType));
        }
        // the reference points for each class
        classifierfile.write((char*)&classifiers[cidx].refpt1.x,sizeof(FloatType));
        classifierfile.write((char*)&classifiers[cidx].refpt1.y,sizeof(FloatType));
        classifierfile.write((char*)&classifiers[cidx].refpt2.x,sizeof(FloatType));
        classifierfile.write((char*)&classifiers[cidx].refpt2.y,sizeof(FloatType));
        // some information useful for debugging
        classifierfile.write((char*)&classifiers[cidx].absmaxXY,sizeof(FloatType));
    }
    classifierfile.close();

    return 0;
}
