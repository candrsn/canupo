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

#include "classifier.hpp"

#include "base64.hpp"

using namespace std;
using namespace boost;

int help(const char* errmsg = 0) {
cout << "\
validate_classifier  user_modified.svg  classifier_file.prm  [ class_num_1  class_num_2 ] [: class1.msc ... - class2.msc ...] \n\
    input: user_modified.svg      # a svg file produced by suggest_classifier, possibly updated by the user\n\
    output: classifier_file.prm   # produces a two-class classifier parameter file for use by \"classify\"\n\
    \n\
    input(optional): class_num_1 class_num_2  # the class number of the first and second classes of this binary classifier\n\
                                              # These can be specified for producing multiclass classifiers by \"combine_classifier\"\n\
                                              # otherwise they have value 1 and 2 by default.\n\
                                              # Class <=0 is reserved for points that cannot be classified.\n\
    input(optional): class1.msc ... - class2.msc ...  # If these are given the classifier performance can be estimated\n\
"<<endl;

    if (errmsg) cout << "Error: " << errmsg << endl;
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

typedef vector<FloatType> sample_type;

// if vector is empty, fill it
// otherwise check the vectors match
int read_msc_header(ifstream& mscfile, vector<FloatType>& scales, int& ptnparams) {
    int npts;
    mscfile.read((char*)&npts,sizeof(npts));
    if (npts<=0) help("invalid file");
    
    int nscales_thisfile;
    mscfile.read((char*)&nscales_thisfile, sizeof(nscales_thisfile));
    vector<FloatType> scales_thisfile(nscales_thisfile);
    for (int si=0; si<nscales_thisfile; ++si) mscfile.read((char*)&scales_thisfile[si], sizeof(FloatType));
    if (nscales_thisfile<=0) help("invalid file");
    
    // all files must be consistant
    if (scales.size() == 0) {
        scales = scales_thisfile;
    } else {
        if (scales.size() != nscales_thisfile) {cerr<<"input file mismatch: "<<endl; return 1;}
        for (int si=0; si<scales.size(); ++si) if (!fpeq(scales[si],scales_thisfile[si])) {cerr<<"input file mismatch: "<<endl; return 1;}
    }
    mscfile.read((char*)&ptnparams, sizeof(int));

    return npts;
}

void read_msc_data(ifstream& mscfile, int nscales, int npts, sample_type* data, int ptnparams) {
    for (int pt=0; pt<npts; ++pt) {
        for (int i=0; i<ptnparams; ++i) {
            FloatType param;
            mscfile.read((char*)&param, sizeof(FloatType));
        }
        for (int s=0; s<nscales; ++s) {
            FloatType a,b;
            mscfile.read((char*)(&a), sizeof(FloatType));
            mscfile.read((char*)(&b), sizeof(FloatType));
            FloatType c = 1 - a - b;
            FloatType x = b + c / 2;
            FloatType y = c * sqrt(3)/2;
            (*data)[s*2] = x;
            (*data)[s*2+1] = y;
        }
        int fooi;
        for (int i=0; i<nscales; ++i) mscfile.read((char*)&fooi, sizeof(int));
        ++data;
    }
}

int main(int argc, char** argv) {

    if (argc<7 && argc!=3 && argc!=5) return help();

    ifstream svgfile(argv[1]);

    ofstream classifierfile(argv[2], ofstream::binary);

    int class_num_1 = 1;
    int class_num_2 = 2;
    if (argc==5) {
        class_num_1 = atoi(argv[3]);
        class_num_2 = atoi(argv[4]);
        if (class_num_1<=0 || class_num_2 <=0) return help();
    }
    
    string line;
    bool incomment = false, inparams = false, inpath = false, inpathdef = false;
    bool incircle = false, hasrefpt1 = false, relative_path = false;
    bool intext = false;
    bool insubparams=false;
    string params;
    vector<Point2D> path;
    Point2D refpt1, refpt2;
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
            if (token == "<text") {intext = true; continue;}
            if (token == "<path") {inpath = true; continue;}
            if (token=="/>") {
                if (incircle) {hasrefpt1 = true; incircle=false;}
                if (inpath) {inpath=false; inpathdef=false;}
                if (intext) {intext=false;}
                continue;
            }
            if (intext && ((token.find("params=")!=-1) || insubparams)) {
                vector<string> subtokens;
                split(subtokens, token, is_any_of("< \t>"));
                for (vector<string>::iterator subit = subtokens.begin(); subit!=subtokens.end(); ++subit) {
                    string& subtoken = *subit;
                    if (subtoken.find("params=")==0) {
                        params+=subtoken.substr(7);
                        insubparams = true;
                        continue;
                    }
                    if (subtoken=="/text") {intext=false; continue;}
                    if (insubparams) params+=subtoken;
                }
            }
            if (inpath && (token=="d=\"M" || token=="d=\"m")) { inpathdef = true; path.clear(); relative_path = (token=="d=\"m"); continue;}
            if (inpathdef) {
                if (token=="L") {relative_path = false; continue;}
                if (token=="l") {relative_path = true; continue;}
                if (token=="\"") {inpathdef = false; continue;}
                if (ends_with(token, "\"")) {
                    token = token.substr(0, token.length()-1);
                    inpathdef = false;
                }
                int commapos = token.find(',');
                if (commapos==-1) {
                    cerr << "unsupported path definition in svg (no comma in the token other than L or l)" << endl;
                    return 0;
                }
                try {
                    Point2D point(
                        lexical_cast<FloatType>(token.substr(0,commapos)),
                        lexical_cast<FloatType>(token.substr(commapos+1))
                    );
                    if (relative_path && path.size()>0) point += path.back();
                    path.push_back(point);
                } catch(bad_lexical_cast) {
                    cerr << "unsupported path definition in svg (bad numeric value)" << endl;
                    return 1;
                }
            }
            if (token == "<circle") {incircle=true; continue;}
            if (incircle && (starts_with(token, "cx=") || starts_with(token, "cy="))) {
                bool iscx = starts_with(token, "cx=");
                token = token.substr(3);
                if (starts_with(token, "\"")) token = token.substr(1);
                if (ends_with(token, "\"")) token = token.substr(0, token.length()-1);
                FloatType value;
                try {
                    value = lexical_cast<FloatType>(token);
                } catch(bad_lexical_cast) {
                    cerr << "invalid circle definition in svg (bad cx or cy value)" << endl;
                    return 1;
                }
                if (hasrefpt1) {
                    if (iscx) refpt2.x = value;
                    else refpt2.y = value;
                }
                else {
                    if (iscx) refpt1.x = value;
                    else refpt1.y = value;
                }
            }
        }
    }

    if (path.size()<2) {
        cerr << "invalid path definition in svg (less than 2 nodes)" << endl;
        return 1;
    }

    if (params.empty()) {
        cout << "Invalid SVG file: initial parameters were not preserved" << endl;
        return 2;
    }
    
    //cout << params << endl; return 0;

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
    FloatType axis_scale_ratio;
    memcpy(&axis_scale_ratio,&binary_params[bpidx],sizeof(int)); bpidx += sizeof(int);
    
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
    // the reference points for each class
    refpt1.x = (refpt1.x - halfSvgSize) / scaleFactor;
    refpt1.y = (halfSvgSize - refpt1.y) / scaleFactor;
    classifierfile.write((char*)&refpt1.x,sizeof(FloatType));
    classifierfile.write((char*)&refpt1.y,sizeof(FloatType));
    refpt2.x = (refpt2.x - halfSvgSize) / scaleFactor;
    refpt2.y = (halfSvgSize - refpt2.y) / scaleFactor;
    classifierfile.write((char*)&refpt2.x,sizeof(FloatType));
    classifierfile.write((char*)&refpt2.y,sizeof(FloatType));
    // some information useful for debugging
    classifierfile.write((char*)&absmaxXY,sizeof(FloatType));
    classifierfile.write((char*)&axis_scale_ratio,sizeof(FloatType));
    classifierfile.close();

    int arg_class1 = argc;
    for (int argi = 3; argi<argc; ++argi) if (!strcmp(argv[argi],":")) {
        arg_class1 = argi+1;
        break;
    }
    if (arg_class1>=argc) return 0; // done, no estimation required
    
    int arg_class2 = argc;
    for (int argi = arg_class1+1; argi<argc; ++argi) if (!strcmp(argv[argi],"-")) {
        arg_class2 = argi+1;
        break;
    }
    if (arg_class2>=argc) {
        return help("Classifier parameters were computed, but we cannot estimate the performance with only one class !");
    }

    // estimate the performance of the classifier if required
    sample_type undefsample;
    int ptnparams;
    // class1 files
    int ndata_class1 = 0;
    for (int argi = arg_class1; argi<arg_class2-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales, ptnparams);
        mscfile.close();
        ndata_class1 += npts;
    }
    // class2 files
    int ndata_class2 = 0;
    for (int argi = arg_class2; argi<argc; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales, ptnparams);
        mscfile.close();
        ndata_class2 += npts;
    }
    undefsample.resize(fdim,FloatType(0));
    int nsamples = ndata_class1+ndata_class2;
    vector<sample_type> samples(nsamples, undefsample);

    int base_pt = 0;
    for (int argi = arg_class1; argi<arg_class2-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales, ptnparams);
        read_msc_data(mscfile,nscales,npts,&samples[base_pt], ptnparams);
        mscfile.close();
        base_pt += npts;
    }
    for (int argi = arg_class2; argi<argc; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales, ptnparams);
        read_msc_data(mscfile,nscales,npts,&samples[base_pt], ptnparams);
        mscfile.close();
        base_pt += npts;
    }

    Classifier classifier;
    classifier.weights_axis1 = weights_axis1;
    classifier.weights_axis2 = weights_axis2;
    classifier.path = path;
    classifier.refpt_pos = refpt1;
    classifier.refpt_neg = refpt2;
    classifier.absmaxXY = absmaxXY;
    classifier.prepare();

    // true/false positive/negative counts
    FloatType TP=0, TN=0, FP=0, FN=0;
    double m1 = 0, m2 = 0, v1 = 0, v2=0;
    for (int i=0; i<ndata_class1; ++i) {
        FloatType pred = classifier.classify(&samples[i][0]);
        if (pred<0) ++TP;
        else ++FN;
        m1+=pred; v1+=pred*pred;
    }
    for (int i=ndata_class1; i<nsamples; ++i) {
        FloatType pred = classifier.classify(&samples[i][0]);
        if (pred>0) ++TN;
        else ++FP;
        m2+=pred; v2+=pred*pred;
    }
    m1/=ndata_class1; m2/=ndata_class2;
    v1 = (v1 - m1 * m1 * ndata_class1) / (ndata_class1 - 1);
    v2 = (v2 - m2 * m2 * ndata_class2) / (ndata_class2 - 1);
    
    // display scores, at last...
    // http://en.wikipedia.org/wiki/Binary_classification
    // http://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    cout << "True/False Positive/Negative counts: TP=" << TP << ", FN=" << FN << ", TN=" << TN << ", FP=" << FP << endl;
    cout << "Class 1 count (=TP+FN): " << ndata_class1 << ", " << "Class 2 count (=TN+FP): " << ndata_class2 << endl;
    cout << "Sensitivity (true positive rate): " << TP / ndata_class1 << endl;
    cout << "Specificity (true negative rate): " << TN / ndata_class2 << endl;
    FloatType mcc = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN);
    if (mcc<=0) mcc = 0;
    else mcc = ((TP*TN) - (FP*FN)) / sqrt(mcc);
    cout << "Matthews correlation coefficient (note: -1 ≤ mcc ≤ 1): " << mcc << endl;
    cout << "Accuracy: " << (TP+TN) / nsamples << endl;
    cout << "Fisher's discriminant ratio: " << (m1-m2)*(m1-m2)/(v1+v2) << endl;
    cout << "Balanced Accuracy: " << 0.5 * (TP / ndata_class1 + TN / ndata_class2) << endl;
    return 0;
}
