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
        // this allows to pre-allocate the big matrix for the svd
        total_pts += npts;
    }
    classboundaries.push_back(total_pts);

    if (nclasses==1) {
        cerr << "Only one class! Please provide several classes to distinguish." << endl;
        return 1;
    }
    
    // allocate the big matrix. it will be column-major,
    int fdim = nscales * 2;
    vector<FloatType> A(total_pts * fdim);
    vector<FloatType> avg_feat(fdim, 0);
    
    int base_pt = 0;
    // and then fill it with the files
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
        
        // now fill in the matrix with the points
        for (int pt=0; pt<npts; ++pt) {
            int ptidx; // we do not care for the point order here
            mscfile.read((char*)&ptidx, sizeof(ptidx));
            
/*
            for (int f=0; f<fdim; ++f) {
                // matrix is column-major, but input file is row-major...
                FloatType value;
                mscfile.read((char*)(&value), sizeof(FloatType));
                A[f*total_pts+base_pt+pt] = value;
                avg_feat[f] += value;
            }
*/
            for (int f=0; f<nscales; ++f) {
                // matrix is column-major, but input file is row-major...
                FloatType a,b;
                mscfile.read((char*)(&a), sizeof(FloatType));
                mscfile.read((char*)(&b), sizeof(FloatType));
                FloatType c = 1 - a - b;
                // project in the equilateral triangle a*(0,0) + b*(1,0) + c*(1/2,sqrt3/2)
                // so each dimension is given equal weight during the PCA
                // is this necessary ?
                FloatType x = b + c / 2;
                FloatType y = c * sqrt(3)/2;                
                A[(f*2)*total_pts+base_pt+pt] = x;
                avg_feat[f*2] += x;
                A[(f*2+1)*total_pts+base_pt+pt] = y;
                avg_feat[f*2+1] += y;
            }
        }
        mscfile.close();
        base_pt += npts;
    }
    
    // now center the matrix to properly do the PCA on the covariance matrix using SVD
    for (int f=0; f<fdim; ++f) {
        avg_feat[f] /= total_pts;
        for (int pt=0; pt<total_pts; ++pt) A[f*total_pts+pt] -= avg_feat[f];
    }
    
    // singular values
    vector<FloatType> S(fdim);
    // principal component vector basis
    vector<FloatType> B(fdim*fdim);

    // make a copy, svd destroys the original matrix
    vector<FloatType> Aori = A;
    
    // now do the big svd, keeping all projected series in A for further classification
    svd(total_pts, fdim, &A[0], &S[0], true, &B[0]);

/*    // display first observation
    cout << "s=[ "; for (int comp=0; comp<fdim; ++comp) cout << S[comp] << " "; cout << "]"<<endl;
    cout << "aori=[ "; for (int comp=0; comp<fdim; ++comp) cout << Aori[comp*total_pts] << " "; cout << "]"<<endl;
    cout << "acmp=[ "; for (int comp=0; comp<fdim; ++comp) cout << A[comp*total_pts] << " "; cout << "]"<<endl;
    cout << "b=["; for (int comp=0; comp<fdim; ++comp) {
        for (int f=0; f<fdim; ++f) cout << B[f*fdim+comp] << " ";
        if (comp<fdim-1) cout << "; "; else cout << "]";
    } cout << endl;
*/  

    // sanity check: project the original data on the basis vectors, shall find the svd series
    // first scale the B matrix by the singular values so as to get the projected series directly
    for (int comp=0; comp<fdim; ++comp) {
        for (int f=0; f<fdim; ++f) B[f*fdim+comp] /= S[comp];
    }
    vector<FloatType> avgdiff(fdim,0), avgscale(fdim,0);
    for (int obs = 0; obs<total_pts; ++obs) {
        // for each component
        for (int comp=0; comp<fdim; ++comp) {
            double res = 0;
            // that component is the observation dot the unit vector, times S[comp]
            for (int f=0; f<fdim; ++f) res += Aori[f*total_pts + obs] * B[f*fdim+comp];
            //cout << res << " shall be equal to " << A[comp * total_pts + obs] << endl;
            avgdiff[comp] += fabs(A[comp * total_pts + obs] - res);
            avgscale[comp] += fabs(A[comp * total_pts + obs] + res) * 0.5;
        }
    }
    cout << "Reliability of each component : average difference between calculated series and projected one, compared to data scale" << endl;
    int nreliablecomp = 0, hasunreliable = 0;
    for (int comp=0; comp<fdim; ++comp) {
        cout << (avgdiff[comp]/total_pts) << " / " << (avgscale[comp]/total_pts);
        if (avgdiff[comp]/avgscale[comp] < 1e-6) {
            if (hasunreliable) cout << " would be OK except for previous components" << endl;
            else {
                cout << " OK (<1e-6)" << endl;
                nreliablecomp++;
            }
        }
        else cout << " unreliable" << endl;
    }
    cout << endl;

    // Output file contains all valid scaled projection vectors and classif params
    classifparamsfile.write((char*)&nreliablecomp, sizeof(int));
    for (int comp=0; comp<nreliablecomp; ++comp) {
        // write the B vector for this component, row-major to simplify later reuse...
        for (int f=0; f<fdim; ++f) B[f*fdim+comp];
        
        // TODO: here
        
    }

    
    // informative output, may help select a number of features
    cout << "Total variance explained by each reliable feature" << endl;
    FloatType totalvar = 0;
    for (int f=0; f<fdim; ++f) totalvar += S[f] * S[f];
    for (int f=0; f<nreliablecomp; ++f) {
        cout << " " << (S[f] * S[f] / totalvar) << endl;
    }
    cout << endl;
    cout << "Cumulated variance explained by up to each feature" << endl;
    FloatType cumvar = 0;
    for (int f=0; f<nreliablecomp; ++f) {
        cumvar += S[f] * S[f];
        cout << " " << (cumvar / totalvar) << endl;
    }
    cout << endl;

    // set of weights for classifying each class, at each level of retained components
    vector<vector<FloatType> > weights[nclasses];
    
    // std technique : ssq to separate one class from all others => 1 set of weights for each class
    // apply that to each subset of features
    for (int f=0; f<nreliablecomp; ++f) {
        int ncomp = f+1;
        FloatType accuracy = 0;
        for (int c = 0; c < nclasses; ++c) {
            // ncomp+1 coefficients for the hyperplane def (aka col of 1, shifting coef, etc)
            weights[c].push_back(vector<FloatType>(ncomp+1,0));
            // need to allocate a new matrix as the content is destroyed by the algorithm
            // Add a column of 1 to the projected information so that we can properly do the least-square fitting
            vector<FloatType> Asel(total_pts * (ncomp+1));
            // Copy the matrix, column-major = just take the first portion of memory
            memcpy(&Asel[0], &A[0], ncomp * total_pts * sizeof(FloatType));
            // Now add the column of 1
            for (int pt=0; pt<total_pts; ++pt) Asel[ncomp * total_pts + pt] = 1;
            // Prepare B = the classes indicator. within class = +1, outside = -1
            vector<FloatType> Bsel(total_pts);
            int beg = classboundaries[c];
            int end = classboundaries[c+1]; // was filled with total_pts at nclasses
            for (int pt=0; pt<beg; ++pt) Bsel[pt] = -1;
            for (int pt=beg; pt<end; ++pt) Bsel[pt] = 1;
            for (int pt=end; pt<total_pts; ++pt) Bsel[pt] = -1;
            // Do the least square fit
            leastSquares(&Asel[0], total_pts, ncomp+1, &Bsel[0], 1);
            // result is the ncomp+1 first entries of B
            for (int i=0; i<=ncomp; ++i) weights[c].back()[i] = Bsel[i];
            // compute the classification error for this class
            int ncorrectclasses = 0;
            for (int pt=beg; pt<end; ++pt) {
                FloatType pred = 0;
                for (int i=0; i<ncomp; ++i) pred += A[i * total_pts + pt] * Bsel[i];
                pred += Bsel[ncomp]; // factor for the column of 1
                if (pred>0) ++ncorrectclasses;
            }
            // accuracy that gives equal weight to each class
            accuracy += (FloatType)ncorrectclasses / (FloatType)(end-beg);
        }
        accuracy /= nclasses;
        cout << "Classification accuracy for " << (f+1) << " retained component(s): " << (accuracy*100) << "%" << endl;
        //if (accuracy==1) break;
    }

    // TODO: write matrix B for projection
    // trick : scale B, not A, so that the projections of unknown points from the full data set are compatible
        
    return 0;
}


