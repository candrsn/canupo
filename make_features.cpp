#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <string.h>

#include <assert.h>
#include <stdlib.h>

#include "points.hpp"
#include "svd.hpp"
#include "leastSquares.hpp"

using namespace std;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
make_features features.prm data1.msc data2.msc - data3.msc - data4.msc...\n\
  inputs: dataX.msc     # The multiscale parameters for the samples the user wishes to discriminate\n\
                        # Use - separators to indicate each class, one or more samples allowed per class\n\
  output: features.prm  # The resulting parameters for feature extraction and classification of the whole scene\n\
"<<endl;
        return 0;
}

// simple linear fit trying to map data to -1,+1 classes with least squared error
struct LeastSquareModel {
    vector<FloatType> weights;
    vector<FloatType> A, B; // matrices for lapack
    int ndata, dim, dataidx;
    
    void prepareTraining(int _ndata, int _dim) {
        ndata = _ndata; dim = _dim;
        assert(ndata>dim);
        // LAPACK matrices are column-major and destroyed in the process => allocate new matrices now
        // add a column of 1 for allowing hyperplanes not going through the origin
        A.resize(ndata * (dim+1));
        B.resize(ndata);
        dataidx = 0;
    }
    
    // data shall point to an array of dim floats. label shall be either +1 or -1
    void addTrainData(const FloatType* data, FloatType label) {
        for (int d=0; d<dim; ++d) A[d*ndata + dataidx] = data[d];
        A[dim*ndata + dataidx] = 1;
        B[dataidx] = label;
        ++dataidx;
    }
    
    void train() {
        // now the least squares hyperplane fit
        leastSquares(&A[0], ndata, dim+1, &B[0], 1);
        weights.resize(dim+1);
        for (int d=0; d<=dim; ++d) weights[d] = B[d];
    }
    
    FloatType predict(const FloatType* data) {
        FloatType ret = weights[dim];
        for (int d=0; d<dim; ++d) ret += weights[d] * data[d];
        return ret;
    }

    // shuffling the data order is necessary so classes are equally distributed in each fold region
    void shuffle() {
        for (int i=ndata-1; i>=1; --i) {
            // may select i itself for the permutation if the element does not change place
            int ridx = random() % (i+1);
            for (int d=0; d<dim; ++d) swap(A[d * ndata + ridx], A[d * ndata + i]);
            swap(B[ridx], B[i]);
        }
    }
    
    void crossValidate(int nfolds, FloatType &accuracy, FloatType &perf) {
        // shall shuffle first if necessary
        int ncorrectpos = 0, ncorrectneg = 0;
        // stats on each class
        int npos = 0, nneg = 0;
        //FloatType perfpos = 1e6, perfneg = 1e6;
        FloatType perfpos = 0, perfneg = 0;
        for (int pt=0; pt<ndata; ++pt) {
            if (B[pt]>=0) ++npos;
            if (B[pt]<0) ++nneg;
        }
        for (int fold = 0; fold < nfolds; ++fold) {
            int fbeg = ndata * fold / nfolds;
            int fend = ndata * (fold+1) / nfolds;
            int ncv = fend - fbeg;
            int nremdata = ndata - ncv;
            vector<FloatType> Acv(nremdata * (dim+1));
            vector<FloatType> Bcv(nremdata);
            for (int d=0; d<dim; ++d) {
                const int baseremdata = d * nremdata;
                const int basedata = d * ndata;
                for (int pt=0; pt<fbeg; ++pt) {
                    Acv[baseremdata + pt] = A[basedata + pt];
                    Bcv[pt] = B[pt];
                }
                for (int pt=fend; pt<ndata; ++pt) {
                    Acv[baseremdata + pt - ncv] = A[basedata + pt];
                    Bcv[pt-ncv] = B[pt];
                }
            }
            // column of 1
            for (unsigned int i = dim * nremdata; i<Acv.size(); ++i) Acv[i] = 1.0;
            // train only on partial data
            leastSquares(&Acv[0], nremdata, dim+1, &Bcv[0], 1);
            // predict on the cv region
            for (int pt=fbeg; pt<fend; ++pt) {
                FloatType pred = Bcv[dim];
                for (int d=0; d<dim; ++d) pred += Bcv[d] * A[d * ndata + pt];
                if (B[pt] >= 0) {if (pred >= 0) ++ncorrectpos; perfpos += pred*fabs(pred);}
                if (B[pt] < 0) {if (pred < 0) ++ncorrectneg; perfneg -= pred*fabs(pred);}
            }
        }
        // return the equilibrated performance
        accuracy = 0.5 * (ncorrectpos / (FloatType)npos + ncorrectneg / (FloatType)nneg);
        perf = 0.5 * (perfpos / npos + perfneg / nneg);
    }
        
};


int main(int argc, char** argv) {
    
    if (argc<4) return help();

    int nclasses = 1;
    vector<int> classboundaries(1,0);
    
    int total_pts = 0;

    int nscales = 0;
    vector<FloatType> scales;
    
    ofstream classifparamsfile(argv[1], ofstream::binary);

    // read headers and ensures all files are consistent
    for (int argi = 2; argi<argc; ++argi) {
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
        
        // all files must be consistant
        if (nscales == 0) {
            nscales = nscales_thisfile;
            scales = scales_thisfile;
        } else {
            if (nscales != nscales_thisfile) {cerr<<"input file mismatch: "<<argv[argi]<<endl; return 1;}
            for (int si=0; si<nscales; ++si) if (scales[si]!=scales_thisfile[si]) {cerr<<"input file mismatch: "<<argv[argi]<<endl; return 1;}
        }
        mscfile.close();
        total_pts += npts;
    }
    classboundaries.push_back(total_pts);

    if (nclasses==1) {
        cerr << "Only one class! Please provide several classes to distinguish." << endl;
        return 1;
    }
    
    // number of features
    int fdim = nscales * 2;

    // store all data in memory, keep overhead to a minimum as we plan for a lot of points...
    vector<FloatType> data(total_pts * fdim);
    int base_pt = 0;
    // and then fill data from the files
    for (int argi = 2; argi<argc; ++argi) {
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
            for (int s=0; s<nscales; ++s) {
                FloatType a,b;
                mscfile.read((char*)(&a), sizeof(FloatType));
                mscfile.read((char*)(&b), sizeof(FloatType));
                FloatType c = 1 - a - b;
                // project in the equilateral triangle a*(0,0) + b*(1,0) + c*(1/2,sqrt3/2)
                // equivalently to the triangle formed by the three components unit vector
                // (1,0,0), (0,1,0) and (0,0,1) when considering a,b,c in 3D
                // so each a,b,c = dimensionality of the data is given equal weight
                // is this necessary ? => not for linear classifiers, but plan ahead...
                FloatType x = b + c / 2;
                FloatType y = c * sqrt(3)/2;
                data[(base_pt+pt)*fdim + s*2] = x;
                data[(base_pt+pt)*fdim + s*2+1] = y;
            }
        }
        mscfile.close();
        base_pt += npts;
    }

    // Now the one-against-one fitting of hyperplanes
    // there will be (nclasses * (nclasses-1) / 2) classifiers
    // let i,j and i<j be two classes to compare (hence j>=1 and i>=0)
    // the classifier for these two classes is stored idx = j*(j-1)/2 + i
    // each classifier is a set of fdim+1 weights = one hyperplane that may be shifted from origin
    vector<vector<FloatType> > weights(nclasses * (nclasses-1) / 2);
    vector<int> selectedScales(nclasses * (nclasses-1) / 2, -1);
    
    // train all one-against-one classifiers
    for (int jclass = 1; jclass < nclasses; ++jclass) {
        int jbeg = classboundaries[jclass];
        int jend = classboundaries[jclass+1];
        int nj = jend - jbeg;
        for (int iclass = 0; iclass < jclass; ++iclass) {
            int ibeg = classboundaries[iclass];
            int iend = classboundaries[iclass+1];
            int ni = iend - ibeg;
            
            int ntotal = ni + nj;
            
            cout << "Classifier for classes " << iclass << " vs " << jclass << endl;
            FloatType bestaccuracy = -1;
            FloatType bestperf = -1e6;
            int bestscale = -1;
            
            // scale by scale classif, look for characteristic scale and add more scales if necessary
            for(int s=0; s<nscales; ++s) {
            
                // 10-fold CV for estimating generalisation ability instead of train classif
                // train a classifier per scale
                LeastSquareModel classifier;
                classifier.prepareTraining(ntotal, 2);
                
                // fill the training data
                for (int pt=0; pt<ni; ++pt) classifier.addTrainData(&data[(ibeg+pt)*fdim + s*2], -1);
                for (int pt=0; pt<nj; ++pt) classifier.addTrainData(&data[(jbeg+pt)*fdim + s*2], 1);

                classifier.shuffle();                
                FloatType accuracy, perf;
                classifier.crossValidate(10, accuracy, perf);
                cout << "cv accuracy using scale " << scales[s] << " only: " << accuracy << " perf " << perf << endl;
                
                if (bestaccuracy < accuracy || (bestaccuracy == accuracy && bestperf < perf) ) {
                    bestaccuracy = accuracy;
                    bestperf = perf;
                    bestscale = s;
                }
                
/*                FloatType accuracy = 0.5 * (ncorrecti / (FloatType)ni + ncorrectj / (FloatType)nj);
                FloatType perfclassif = 0.5 * (perfclassifi / ni + perfclassifj / nj);
                cout << "accuracy using scale " << scales[s] << " only: " << accuracy << " perf " << perfclassif << endl;
*/
            }
            cout << "Selected scale " << scales[bestscale] << endl;
            
            
            // TODO: multiple scales
            // - sort scales by accuracy
            // - select best accuracy + next best, etc.
            // - OR select all combo : too many combinations...
            // - select best + run through (best + all others) = N-1 tests
            // if some two-element combo (best + one other) is > best alone, then take it
            // - go to three element combo (best 2 combo + one other) = N-2 tests...
            // until no more improvement (stop before if accuracy 1 is reached)
            
            // train a classifier at this scale on the whole data set
            int idx = jclass*(jclass-1)/2 + iclass;
            LeastSquareModel classifier;
            classifier.prepareTraining(ntotal, 2);
            // fill the training data
            for (int pt=0; pt<ni; ++pt) classifier.addTrainData(&data[(ibeg+pt)*fdim + bestscale*2], -1);
            for (int pt=0; pt<nj; ++pt) classifier.addTrainData(&data[(jbeg+pt)*fdim + bestscale*2], 1);
            // no need to shuffle for whole data set training
            classifier.train();
            // store the weights and best scale for later use
            weights[idx] = classifier.weights;
            selectedScales[idx] = bestscale;
            
            // TODO: Select more than one scale
            //       process combination of best scales so long as it increases the performance
            //       here we found 100% CV classif... but wait till we do n-class voting
        }
    }

    // compute error of one-against-one on the training set
    // TODO: split and keep a test set here
    vector<int> ncorrectclassif(nclasses,0);
    for (int pt=0; pt<total_pts; ++pt) {
        FloatType* ptdata = &data[pt*fdim];
        // one-against-one process: apply all classifiers and vote for this point class
        vector<FloatType> predictions(nclasses * (nclasses-1) / 2);
        vector<int> votes(nclasses, 0);
        for (int j=1; j<nclasses; ++j) for (int i=0; i<j; ++i) {
            int idx = j*(j-1)/2 + i;
            // single scale classifier for now: TODO: multi-scale
            FloatType pred = weights[idx][2];
            pred += ptdata[selectedScales[idx]*2] * weights[idx][0];
            pred += ptdata[selectedScales[idx]*2+1] * weights[idx][1];
            if (pred>=0) ++votes[j];
            else ++votes[i];
            predictions[idx] = pred;
        }
        // search for max vote, in case equality = use the classifier between both to break the vote
        // TODO: in case of loops (ex: all 3 ex-aequo amongst 3) and all classifiers say a > b > c > a
        //       currently the loop is broken by the order in which comparisons are made, but another
        //       criterion like distance from hyperplanes would be better
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
        // look at the real class for checking
        int ptclass = -1;
        for (int c=0; c<nclasses; ++c) if (pt>=classboundaries[c] && pt<classboundaries[c+1]) {ptclass=c; break;}
        assert(ptclass!=-1);
        
        if (selectedclass == ptclass) ++ncorrectclassif[ptclass];
    }
    
    // compute accuracy that a given point in the data set is correctly classified: weight all classes 
    FloatType accuracy = 0;
    for (int c=0; c<nclasses; ++c) accuracy += ncorrectclassif[c] / (FloatType)(classboundaries[c+1] - classboundaries[c]);
    accuracy /= nclasses;
    cout << "One-against-one voting accuracy: " << accuracy << endl;

    // parameters in the classifier definition file :
    // write nunique selected scales first
    // write these scales numeric values (so reduced msc files containing only these are OK too)
    // write num of classes
    // write for each of the n(n-1)/2 classifiers 
    // - num of scales used by (here 1 unique scale for now, but plan for extension)
    // - the scales used 
    // - weights (2*n_used_scales + 1 coefs per classifier)
    set<int> uniqueScales(selectedScales.begin(), selectedScales.end());
    int nuniqueScales = uniqueScales.size();
    classifparamsfile.write((char*)&nuniqueScales, sizeof(nuniqueScales));
    cout << "Selected scales (for extraction of the whole scene):";
    for (set<int>::iterator it = uniqueScales.begin(); it!=uniqueScales.end(); ++it) {
        cout << " " << scales[*it];
        classifparamsfile.write((char*)&scales[*it], sizeof(FloatType));
    }
    cout << endl;
    classifparamsfile.write((char*)&nclasses, sizeof(nclasses));
    int nclassifiers = nclasses * (nclasses-1) / 2;
    for (int i=0; i<nclassifiers; ++i) {
        int numscales = 1; // TODO: implement multi-scale classifiers
        classifparamsfile.write((char*)&numscales, sizeof(numscales));
        //for (int j=0; j<numscales; ++j)
        classifparamsfile.write((char*)&scales[selectedScales[i]], sizeof(FloatType));
        // (2*n_used_scales + 1 coefs per classifier)
        for (int j=0; j<=2*numscales; ++j) classifparamsfile.write((char*)&weights[i][j], sizeof(FloatType));
    }
    
    return 0;
}


