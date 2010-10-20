#include <iostream>
#include <fstream>
#include <string.h>

#include <assert.h>

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
            int ptidx; // we do not care for the point order here
            mscfile.read((char*)&ptidx, sizeof(ptidx));
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

/*                // keep 3 coordinates for later interpretation of the weights !
                // nop, not well conditionned...
                data[(base_pt+pt)*fdim + s*3  ] = a;
                data[(base_pt+pt)*fdim + s*3+1] = b;
                data[(base_pt+pt)*fdim + s*3+2] = c;
*/
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
    vector<vector<FloatType> > weights(nclasses * (nclasses-1) / 2, vector<FloatType>(fdim+1));
    
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
            // scale by scale classif, look for characteristic scale and add more scales if necessary
            for(int s=0; s<nscales; ++s) {
                // train a classifier only at this scale
                vector<FloatType> A(ntotal * 3);
                vector<FloatType> B(ntotal);
                // fill the matrices with the classes data
                for (int pt=0; pt<ni; ++pt) {
                    for(int f=0; f<2; ++f) A[f * ntotal + pt] = data[(ibeg+pt)*fdim + s*2+f];
                    A[2*ntotal + pt] = 1;
                    B[pt] = -1;
                }
                for (int pt=0; pt<nj; ++pt) {
                    for(int f=0; f<2; ++f) A[f * ntotal + (pt+ni)] = data[(jbeg+pt)*fdim + s*2+f];
                    A[2*ntotal + (pt+ni)] = 1;
                    B[pt+ni] = 1;
                }
                // now the least squares hyperplane fit
                leastSquares(&A[0], ntotal, 3, &B[0], 1);
                // percent classif obtained by this scale alone ?
                int ncorrecti = 0; FloatType perfclassifi = 0;
                for (int pt=0; pt<ni; ++pt) {
                    FloatType pred = B[2];
                    pred += data[(ibeg+pt)*fdim + s*2] * B[0];
                    pred += data[(ibeg+pt)*fdim + s*2+1] * B[1];
                    if (pred<0) ++ncorrecti;
                    // go through sigmoid function, seek for negative class
                    perfclassifi += 1.0 / (1.0 + exp(pred));
                }
                int ncorrectj = 0; FloatType perfclassifj = 0;
                for (int pt=0; pt<nj; ++pt) {
                    FloatType pred = B[2];
                    pred += data[(jbeg+pt)*fdim + s*2] * B[0];
                    pred += data[(jbeg+pt)*fdim + s*2+1] * B[1];
                    if (pred>=0) ++ncorrectj;
                    // go through sigmoid function, seek for positive class
                    perfclassifj += 1.0 / (1.0 + exp(-pred));
//                    perfclassifj += tanh(pred);
                }
                FloatType accuracy = 0.5 * (ncorrecti / (FloatType)ni + ncorrectj / (FloatType)nj);
                FloatType perfclassif = 0.5 * (perfclassifi / ni + perfclassifj / nj);
                cout << "accuracy using scale " << scales[s] << " only: " << accuracy << " perf " << perfclassif << endl;
                
            }
            
            
            
#if 0
            // need to allocate a new matrix as the content is destroyed by the algorithm
            // Add a column of 1 so as to allow hyperplanes not necessarily going through the origin
            // A is column-major... and each row of A is an instance
            vector<FloatType> A(ntotal * (fdim+1));
            // Prepare B = the classes indicator. class i = -1, j = +1
            vector<FloatType> B(ntotal);
            // fill the matrices with the classes data
            for (int pt=0; pt<ni; ++pt) {
                for(int f=0; f<fdim; ++f) A[f * ntotal + pt] = data[(ibeg+pt)*fdim + f];
                A[fdim*ntotal + pt] = 1;
                B[pt] = -1;
            }
            for (int pt=0; pt<nj; ++pt) {
                for(int f=0; f<fdim; ++f) A[f * ntotal + (pt+ni)] = data[(jbeg+pt)*fdim + f];
                A[fdim*ntotal + (pt+ni)] = 1;
                B[pt+ni] = 1;
            }
            // now the least squares hyperplane fit
            leastSquares(&A[0], ntotal, fdim+1, &B[0], 1);
            // result is the fdim+1 first entries of B
            // compute the classification error
            int ncorrecti = 0;
            for (int pt=0; pt<ni; ++pt) {
                FloatType pred = 0;
                for(int f=0; f<fdim; ++f) pred += data[(ibeg+pt)*fdim + f] * B[f];
                pred += B[fdim]; // bias at origin
                if (pred<0) ++ncorrecti;
            }
            int ncorrectj = 0;
            for (int pt=0; pt<nj; ++pt) {
                FloatType pred = 0;
                for(int f=0; f<fdim; ++f) pred += data[(jbeg+pt)*fdim + f] * B[f];
                pred += B[fdim]; // bias at origin
                if (pred>=0) ++ncorrectj;
            }
            // accuracy that gives equal weight to each class
            FloatType accuracy = 0.5 * (ncorrecti / (FloatType)ni + ncorrectj / (FloatType)nj);
            // that accuracy will be the voting weight for this classifer when predicting new points
            cout << "accuracy for discriminating classes " << iclass << " and " << jclass << " is: " << accuracy << endl;
            cout << "set of weights + bias:";
            for(int f=0; f<fdim; ++f) cout << " " << B[f];
            cout << " " << B[fdim] << endl;
    
            // TODO: write weights in param file
            
            // TODO: interpretation of weights according to scales
            for(int s=0; s<nscales; ++s) {
                FloatType xw = B[s*2];
                FloatType yw = B[s*2+1];
                FloatType cte = B[fdim];
                // percent classif obtained by this scale alone ?
                int ncorrecti = 0;
                for (int pt=0; pt<ni; ++pt) {
                    FloatType pred = cte;
                    pred += data[(ibeg+pt)*fdim + s*2] * xw;
                    pred += data[(ibeg+pt)*fdim + s*2+1] * yw;
                    if (pred<0) ++ncorrecti;
                }
                int ncorrectj = 0;
                for (int pt=0; pt<nj; ++pt) {
                    FloatType pred = cte;
                    pred += data[(jbeg+pt)*fdim + s*2] * xw;
                    pred += data[(jbeg+pt)*fdim + s*2+1] * yw;
                    if (pred>=0) ++ncorrectj;
                }
                FloatType accuracy = 0.5 * (ncorrecti / (FloatType)ni + ncorrectj / (FloatType)nj);
                cout << "accuracy using scale " << scales[s] << " only: " << accuracy << endl;
                
                // Line equa of projected hyperplane on this plane : xw * x + yw * y + cte = 0
                // all other components are 0 due to projection
                // but x = b + c / 2  and   y = c * sqrt(3)/2;  and  a + b + c = 1
                // so:  xw * (b + c / 2) + yw * (c * sqrt(3)/2) + cte * (a + b + c) = 0
                //      (xw + cte) * b + (xw/2 + yw *sqrt(3)/2 + cte) * c + cte * a = 0;
            }
#endif
        }
    }
        
    return 0;
}


