#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <limits>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <stdint.h>

#include "points.hpp"

#include "base64.hpp"

using namespace std;
using namespace boost;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
validate_classifier  user_modified_svg  classifier_file.prm  [ class_num_1  class_num_2 ] \n\
    produce biclass prm file\n\
    from SVG, use path (predef path if not changed or user-defined)\n\
    class_num_1  and  class_num_2  can be specified for producing multiclass classifiers\n\
    otherwise they have value 1 and 2 by default\n\
"<<endl;
        return 0;
}

bool fpeq(FloatType a, FloatType b) {
    static const FloatType epsilon = 1e-6;
    if (b==0) return fabs(a)<epsilon;
    FloatType ratio = a/b;
    return ratio>1-epsilon && ratio<1+epsilon;
}

struct LineDef {
    FloatType wx, wy, c;
};

int main(int argc, char** argv) {
    
    if (argc!=3 && argc!=5) return help();

    ifstream svgfile(argv[1]);
    
    ofstream classifierfile(argv[2], ofstream::binary);
    
    int class_num_1 = 1;
    int class_num_2 = 2;
    if (argc==5) {
        class_num_1 = atoi(argv[3]);
        class_num_2 = atoi(argv[4]);
//        if (class_num_1<=0 || class_num_2 <=0) return help();
    }
    
    string line;
    bool incomment = false, inparams = false, inpath = false, inpathdef = false;
    string params;
    vector<Point2D> path;
    while (svgfile && !svgfile.eof()) {
        getline(svgfile, line);
        if (line.empty()) continue;
        trim(line);
        vector<string> tokens;
        split(tokens, line, is_any_of(" \t"));
        for (vector<string>::iterator it = tokens.begin(); it!=tokens.end(); ++it) {
            string& token = *it;
            if (token == "<!--") {incomment = true; continue;}
            if (token == "-->") {incomment = false; inparams = false; continue;}
            if (inparams) params += token;
            if (token == "params" && incomment) {inparams = true; continue;}
            if (token == "<path") {inpath = true; continue;}
            if (inpath && token=="/>") {inpath = false; inpathdef = false; continue;}
            if (inpath && token=="d=\"M") { inpathdef = true; path.clear(); continue;}
            if (inpathdef) {
                if (token=="L") continue;
                if (token=="\"") {inpathdef = false; continue;}
                if (ends_with(token, "\"")) {
                    token = token.substr(0, token.length()-1);
                    inpathdef = false;
                }
                int commapos = token.find(',');
                if (commapos==-1) {
                    cerr << "invalid path definition in svg (invalid token)" << endl;
                    return 0;
                }
                try {
                    Point2D point(
                        lexical_cast<FloatType>(token.substr(0,commapos)),
                        lexical_cast<FloatType>(token.substr(commapos+1))
                    );
                    path.push_back(point);
                } catch(bad_lexical_cast) {
                    cerr << "invalid path definition in svg (bad value)" << endl;
                    return 0;
                }
            }
        }
    }

    if (path.size()<2) {
        cerr << "invalid path definition in svg (less than 2 nodes)" << endl;
        return 0;
    }

//    cout << params << endl;
    
    base64 codec;
    vector<char> binary_params(codec.get_max_decoded_size(params.size()));
    // no need for terminating 0, base64 has its own termination marker
    codec.decode(params.c_str(), params.size(), &binary_params[0]);

    int bpidx = 0;
    int nscales;
    memcpy(&nscales,&binary_params[bpidx],sizeof(int)); bpidx += sizeof(int);
    if (nscales<=0 || nscales>=1000) {
        cerr << "Invalid parameters in SVG file" << endl;
        return 1;
    }
    vector<FloatType> scales(nscales);
    for (int i=0; i<nscales; ++i) {
        memcpy(&scales[i],&binary_params[bpidx],sizeof(FloatType));
        bpidx += sizeof(FloatType);
    }
    
    int fdim = nscales * 2;
    vector<FloatType> weights_axis1(fdim+1);
    for (int i=0; i<=fdim; ++i) {
        memcpy(&weights_axis1[i],&binary_params[bpidx],sizeof(FloatType));
        bpidx += sizeof(FloatType);
    }
    vector<FloatType> weights_axis2(fdim+1);
    for (int i=0; i<=fdim; ++i) {
        memcpy(&weights_axis2[i],&binary_params[bpidx],sizeof(FloatType));
        bpidx += sizeof(FloatType);
    }
    FloatType absmaxXY;
    memcpy(&absmaxXY,&binary_params[bpidx],sizeof(FloatType)); bpidx += sizeof(FloatType);
    FloatType scaleFactor;
    memcpy(&scaleFactor,&binary_params[bpidx],sizeof(FloatType)); bpidx += sizeof(FloatType);
    int halfSvgSize;
    memcpy(&halfSvgSize,&binary_params[bpidx],sizeof(int)); bpidx += sizeof(int);
    
    // convert the path from SVG to 2D space
    for(int i=0; i<path.size(); ++i) {
        // xsvg = x2d * scaleFactor + halfSvgSize
        // ysvg = halfSvgSize - y2d * scaleFactor
        path[i].x = (path[i].x - halfSvgSize) / scaleFactor;
        path[i].y = (halfSvgSize - path[i].y) / scaleFactor;
    }
    
    // New classifier format
    // scales are for checking compatibility of the scene file
    classifierfile.write((char*)&nscales,sizeof(int));
    for (int i=0; i<nscales; ++i) classifierfile.write((char*)&scales[i],sizeof(FloatType));
    // number of classifiers embedded in this parameter file
    int nclassifiers=1;
    classifierfile.write((char*)&nclassifiers,sizeof(int));
    // classes handled by this partial classifier, as numbered by the user
    // the first class is classified to -1, the second is to +1
    // further classifiers in the same file may handle different user classes
    // and we deal with multi-class classification by voting
    classifierfile.write((char*)&class_num_1,sizeof(int));
    classifierfile.write((char*)&class_num_2,sizeof(int));
    // the first two directions maximizing the separability of the data
    for (int i=0; i<=fdim; ++i) classifierfile.write((char*)&weights_axis1[i],sizeof(FloatType));
    for (int i=0; i<=fdim; ++i) classifierfile.write((char*)&weights_axis2[i],sizeof(FloatType));
    // the list of points in the path (scaled in the 2D space)
    int pathsize = path.size();
    classifierfile.write((char*)&pathsize,sizeof(int));
    for(int i=0; i<path.size(); ++i) {
        classifierfile.write((char*)&path[i].x,sizeof(FloatType));
        classifierfile.write((char*)&path[i].y,sizeof(FloatType));
    }
    // helper to get max grid size in classify
    classifierfile.write((char*)&absmaxXY,sizeof(FloatType));
    classifierfile.close();

    return 0;
}
