#include <iostream>
#include <fstream>
#include <string.h>

#include <assert.h>

#include "points.hpp"
#include "svd.hpp"

using namespace std;


extern "C" {
    // least square solving from lapack
    void dgelsd_ (const int* m, const int* n, const int* nrhs, double* A, const int* lda, double* B, const int* ldb, double* S, const double* rcond, int* rank, double* work, const int* lwork, int* iwork, int* info);
    void sgelsd_ (const int* m, const int* n, const int* nrhs, float* A, const int* lda, float* B, const int* ldb, float* S, const float* rcond, int* rank, float* work, const int* lwork, int* iwork, int* info);
}

// Our own somewhat simplified wrapper.
void solveLeastSquares(double* A, int nrows, int ncols, double* B, int nrhs) {
    int info = 0;
    int ldb = max(nrows,ncols);
    double *sval = new double[min(nrows,ncols)];
    double rcond = -1;
    int rank;
    // first, workspace query
    int lwork = -1;
    double tmpwork;
    int tmpiwork;
    dgelsd_(&nrows, &ncols, &nrhs, A, &nrows, B, &ldb, sval, &rcond, &rank, &tmpwork, &lwork, &tmpiwork, &info);
    if (info) {
        cerr << "Could not retrieve the work array size for lapack" << endl;
        exit(1);
    }
    lwork = (int)tmpwork;
    double* work = new double[lwork];
    // doc says read size of the int array in tmpiwork, but debugging says it is irrelevant
    // => doc says min required sizes and formula for work is > formula for iwork in all cases: reuse lwork
    int* iwork = new int[lwork]; 
    dgelsd_(&nrows, &ncols, &nrhs, A, &nrows, B, &ldb, sval, &rcond, &rank, work, &lwork, iwork, &info);
    if (info) {
        cerr << "Error in dgelsd: " << info << endl;
        exit(1);
    }
    delete [] iwork;
    delete [] work;
    delete [] sval;
}
void solveLeastSquares(float* A, int nrows, int ncols, float* B, int nrhs) {
    int info = 0;
    int ldb = max(nrows,ncols);
    float *sval = new float[min(nrows,ncols)];
    float rcond = -1;
    int rank;
    // first, workspace query
    int lwork = -1;
    float tmpwork;
    int tmpiwork;
    sgelsd_(&nrows, &ncols, &nrhs, A, &nrows, B, &ldb, sval, &rcond, &rank, &tmpwork, &lwork, &tmpiwork, &info);
    if (info) {
        cerr << "Could not retrieve the work array size for lapack" << endl;
        exit(1);
    }
    lwork = (int)tmpwork;
    float* work = new float[lwork];
    // doc says read size of the int array in tmpiwork, but debugging says it is irrelevant
    // => doc says min required sizes and formula for work is > formula for iwork in all cases: reuse lwork
    int* iwork = new int[lwork]; 
    sgelsd_(&nrows, &ncols, &nrhs, A, &nrows, B, &ldb, sval, &rcond, &rank, work, &lwork, iwork, &info);
    if (info) {
        cerr << "Error in sgelsd: " << info << endl;
        exit(1);
    }
    delete [] iwork;
    delete [] work;
    delete [] sval;
}


int main(int argc, char** argv) {
    if (argc<5 || strcmp(argv[2],":")) {
        cout << "Arguments required: output_classification_definition followed by : then several multiscale files from class 1 followed by - then for class 2, etc" << endl;
        cout << "Ex: " << argv[0] << " classifparams.prm : class1_ex1.msc class1_ex2.msc - class2_ex.msc - class3_ex1.msc class3_ex2.msc class3_ex3.msc" << endl;
        cout << "The classification parameters would here be stored in the file 'classifparams.prm'. They can be used later on to process and classify a full scene, based on the training done on the given examples." << endl;
        return 0;
    }

    int nclasses = 1;
    vector<int> classboundaries(1,0);
    
    int fdim = 0;
    int total_pts = 0;
    
    ofstream classifparamsfile(argv[1], ofstream::binary);

    for (int argi = 3; argi<argc; ++argi) {
        if (!strcmp("-",argv[argi])) {
            ++nclasses;
            classboundaries.push_back(total_pts);
            continue;
        }
        ifstream multiscalefile(argv[argi], ifstream::binary);
        // read the file header
        int npts;
        multiscalefile.read(reinterpret_cast<char*>(&npts), sizeof(int));
        int fdim_local;
        multiscalefile.read(reinterpret_cast<char*>(&fdim_local), sizeof(int));
        if (fdim==0) fdim = fdim_local;
        if (fdim==0) {cerr<<"input file error: "<<argv[argi]<<endl; return 1;}
        if (fdim!=fdim_local) {cerr<<"input file mismatch: "<<argv[argi]<<endl; return 1;}
        multiscalefile.close();
        // this allows to pre-allocate the big matrix for the svd
        total_pts += npts;
    }
    classboundaries.push_back(total_pts);

    //for (int c=0; c<(int)classboundaries.size(); ++c) cout << classboundaries[c] << endl;

    if (nclasses==1) {
        cerr << "Only one class! Please provide several classes to distinguish." << endl;
        return 0;
    }
    
    // allocate the big matrix. it will be column-major,
    vector<FloatType> A(total_pts * fdim);
    vector<FloatType> avg_feat(fdim, 0);
    
    int base_pt = 0;
    // and then fill it with the files
    for (int argi = 3; argi<argc; ++argi) {
        if (!strcmp("-",argv[argi])) continue;
        ifstream multiscalefile(argv[argi], ifstream::binary);
        // read but ignore the file header
        int npts;
        multiscalefile.read(reinterpret_cast<char*>(&npts), sizeof(int));
        int fdim_local;
        multiscalefile.read(reinterpret_cast<char*>(&fdim_local), sizeof(int));
        if (fdim_local!=fdim) {cerr << "internal error" << endl; return 1;}
        // now fill in the matrix with the points
        for (int pt=0; pt<npts; ++pt) {
            FloatType borderStat;
            multiscalefile.read(reinterpret_cast<char*>(&borderStat), sizeof(FloatType));
            
            // TODO: option to eliminate points with bad borderStat
            
            for (int f=0; f<fdim; ++f) {
                // matrix is column-major, but input file is row-major...
                FloatType value;
                multiscalefile.read((char*)(&value), sizeof(FloatType));
assert(f*total_pts+base_pt+pt<A.size());
                A[f*total_pts+base_pt+pt] = value;
                avg_feat[f] += value;
            }
        }
        multiscalefile.close();
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

    // display first observation
    cout << "s=[ "; for (int comp=0; comp<fdim; ++comp) cout << S[comp] << " "; cout << "]"<<endl;
    cout << "aori=[ "; for (int comp=0; comp<fdim; ++comp) cout << Aori[comp*total_pts] << " "; cout << "]"<<endl;
    cout << "acmp=[ "; for (int comp=0; comp<fdim; ++comp) cout << A[comp*total_pts] << " "; cout << "]"<<endl;
    cout << "b=["; for (int comp=0; comp<fdim; ++comp) {
        for (int f=0; f<fdim; ++f) cout << B[f*fdim+comp] << " ";
        if (comp<fdim-1) cout << "; "; else cout << "]";
    } cout << endl;
  
    // sanity check: project the original data on the basis vectors, shall find the series
    vector<FloatType> avgdiff(fdim,0), avgscale(fdim,0);
    for (int obs = 0; obs<total_pts; ++obs) {
        // for each component
        for (int comp=0; comp<fdim; ++comp) {
            double res = 0;
            // that component is the observation dot the unit vector, times S[comp]
            for (int f=0; f<fdim; ++f) res += Aori[f*total_pts + obs] * B[f*fdim+comp]; // take the transpose
            res /= S[comp];
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

    //classifparamsfile.write
    
    // std technique : ssq to separate one class from all others => 1 set of weights for each class
    // apply that to each subset of features
    for (int f=0; f<nreliablecomp; ++f) {
        int ncomp = f+1;
        FloatType accuracy = 0;
        for (int c = 0; c < nclasses; ++c) {
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
            solveLeastSquares(&Asel[0], total_pts, ncomp+1, &Bsel[0], 1);
            // result is the ncomp+1 first entries of B
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
        // TODO: display
        cout << "Classification accuracy for " << (f+1) << " retained component(s): " << (accuracy*100) << "%" << endl;
    }
    
    // TODO: write matrix B for projection
    // trick : scale B, not A, so that the projections of unknown points from the full data set are compatible
        
    return 0;
}


