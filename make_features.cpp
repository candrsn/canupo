#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>

#include <boost/algorithm/string.hpp>

#include <string.h>
#include <assert.h>
#include <stdlib.h>

#include "dlib/svm.h"

#include "points.hpp"
#include "svd.hpp"
#include "leastSquares.hpp"

using namespace std;
using namespace boost;

static const bool useSVM = true;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
make_features features.prm [scales] : data1.msc data2.msc - data3.msc - data4.msc...\n\
  output: features.prm  # The resulting parameters for feature extraction and classification of the whole scene\n\
  input:  scales        # Optional, a set of scales to compute the features on.\n\
                        # If this is not specified an automated procedure will find the scales that\n\
                        # best discriminates the given data. You will then be prompted for which\n\
                        # scales to use. This also allows you to input different scales for the\n\
                        # different classifiers.\n\
                        # If the scales are specified on the command line there is no interaction and\n\
                        # no automated search, and all classifiers use the same scales.\n\
  inputs: dataX.msc     # The multiscale parameters for the samples the user wishes to discriminate\n\
                        # Use - separators to indicate each class, one or more samples allowed per class\n\
                        # The data file lists start after the : separator on the command line\n\
"<<endl;
        return 0;
}

struct Classifier {
    virtual ~Classifier() {}
    virtual void prepareTraining(int _ndata, int _dim) = 0;    
    virtual void addTrainData(const FloatType* data, FloatType label) = 0;
    virtual void train() = 0;
    virtual FloatType predict(const FloatType* data) = 0;
    virtual void shuffle() = 0;
    virtual void crossValidate(int nfolds, FloatType &accuracy, FloatType &perf) = 0;
};

// simple linear fit trying to map data to -1,+1 classes with least squared error
struct LeastSquareModel : public Classifier {
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

struct SVM_Model : public Classifier {

    // see dlib examples
    template <typename sample_type>
    struct cross_validation_objective {
        cross_validation_objective (
            const vector<sample_type>& samples_,
            const vector<double>& labels_,
            int _nfolds
        ) : samples(samples_), labels(labels_), nfolds(_nfolds) {}

//        double operator() (const dlib::matrix<double>& params) const {
        double operator() (FloatType lognu) const {
            using namespace dlib;
            // see below for changes from dlib examples
            //const FloatType nu    = exp(params(0));
            //const double gamma = exp(params(1));
            
            const FloatType nu    = exp(lognu);

            // Make an SVM trainer and tell it what the parameters are supposed to be.
            svm_nu_trainer<kernel_type> trainer;
            //trainer.set_kernel(kernel_type(gamma));
            trainer.set_kernel(kernel_type());
            trainer.set_nu(nu);

            // Finally, perform 10-fold cross validation and then print and return the results.
            matrix<double> result = cross_validate_trainer(trainer, samples, labels, nfolds);

            return sum(result);
        }

        const vector<sample_type>& samples;
        const vector<double>& labels;
        int nfolds;
    };
    
    // Use the SVM from dlib to find an appropriate max-margin hyperplane
    // declare dlib types for invoking the SVMs : samples are column vectors
    // dlib allows both static and dynamic lengths... perfect !
    typedef dlib::matrix<FloatType, 0, 1> sample_type;
    int ndata, dim, dataidx;
    
    // gaussian kernel
    //typedef radial_basis_kernel<sample_type> kernel_type;
    // linear kernel to begin with
    typedef dlib::linear_kernel<sample_type> kernel_type;

    vector<sample_type> samples;
    vector<FloatType> labels;
    
    dlib::matrix<FloatType, 2, 1> params;
    
    virtual void prepareTraining(int _ndata, int _dim) {
        ndata = _ndata; dim = _dim;
        // prepare one sample with the right dimensions as an argument to the vector init
        sample_type undefsample; undefsample.set_size(dim,1);
        // use a vector of samples as in the dlib examples.
        samples.clear();
        samples.resize(ndata, undefsample);
        labels.resize(ndata);
        dataidx = 0;
    }
    
    virtual void addTrainData(const FloatType* data, FloatType label) {
        for (int i=0; i<dim; ++i) samples[dataidx](i) = data[i];
        labels[dataidx] = label;
        ++dataidx;
    }
    
    virtual void shuffle() {
        dlib::randomize_samples(samples, labels);
    }
    
    virtual void crossValidate(int nfolds, FloatType &accuracy, FloatType &perf) {
        using namespace dlib;
        // taken from dlib examples
        
        // largest allowed nu: strictly below what's returned by maximum_nu
        FloatType max_nu = 0.999*maximum_nu(labels);
        
/*        matrix<FloatType> gridsearchspace = cartesian_product(
            logspace(log10(max_nu), log10(1e-5), 4), // nu parameter
            logspace(log10(5.0), log10(1e-5), 4)     // gamma parameter
        );
*/
        matrix<FloatType> gridsearchspace = logspace(log10(max_nu), log10(1e-5), 16); // nu parameter

        matrix<FloatType> best_result(2,1);
        best_result = 0;
        FloatType best_nu = 1;
        FloatType best_gamma = 0.1; // unused for linear kernel
        
        // grid search
        for (int col = 0; col < gridsearchspace.nc(); ++col) {
            // pull out the current set of model parameters
            const FloatType nu    = gridsearchspace(0, col);
            //const FloatType gamma = gridsearchspace(1, col);

            // setup a training object using our current parameters
            svm_nu_trainer<kernel_type> trainer;
            //trainer.set_kernel(kernel_type(gamma));
            trainer.set_kernel(kernel_type()); // no gamma parameter for the linear kernel
            trainer.set_nu(nu);

            // Finally, do 10 fold cross validation and then check if the results are the best we have seen so far.
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);
            //cout << "gamma: " << setw(11) << gamma << "  nu: " << setw(11) << nu <<  "  cross validation accuracy: " << result;
            //cout << "nu: " << setw(11) << nu <<  "  cross validation accuracy: " << result;

            // save the best results
            if (sum(result) > sum(best_result))
            {
                best_result = result;
                best_nu = nu;
//                best_gamma = gamma;
            }
        }
        //cout << "best result of grid search: " << sum(best_result) << endl;
        //cout << "best gamma: " << best_gamma << "   best nu: " << best_nu << endl;
        //cout << "best nu: " << best_nu << endl;
        
        //params.resize(2,1);
        params = best_nu, best_gamma;
        
        // We also need to supply lower and upper bounds for the search.  
        matrix<FloatType> lower_bound(2,1), upper_bound(2,1);
        lower_bound = 1e-7,   // smallest allowed nu
                      1e-7;   // smallest allowed gamma
        upper_bound = max_nu, // largest allowed nu
                      100;    // largest allowed gamma
        
        // convert to log space
        params = log(params);
        lower_bound = log(lower_bound);
        upper_bound = log(upper_bound);

        // Finally, ask BOBYQA to look for the best set of parameters
/*        double best_score = find_max_bobyqa(
            cross_validation_objective(samples, labels), // Function to maximize
            params,                                      // starting point
            params.size()*2 + 1,                         // See BOBYQA docs, generally size*2+1 is a good setting for this
            lower_bound,                                 // lower bound 
            upper_bound,                                 // upper bound
            min(upper_bound-lower_bound)/10,             // search radius
            0.01,                                        // desired accuracy
            100                                          // max number of allowable calls to cross_validation_objective()
            );
*/

        FloatType best_score = find_max_single_variable(
            cross_validation_objective<sample_type>(samples, labels, nfolds), // Function to maximize
            params(0),                                   // starting point
            lower_bound(0),
            upper_bound(0),
            1e-2,
            100
        );

        // Don't forget to convert back from log scale to normal scale
        params = exp(params);

//        cout << " best result of BOBYQA: " << best_score << endl;
//        cout << " best gamma: " << params(1) << "   best nu: " << params(0) << endl;
//        cout << " best result of find_max_single_variable: " << best_score << endl;
//        cout << " best nu: " << params(0) << endl;

        accuracy = best_score * 0.5;
        perf = best_score * 0.5; // TODO: find better measure
    }
    
    virtual void train() {
        using namespace dlib;
        
        // we first need to select a good nu => crossvalidation no matter what
        shuffle();
        FloatType dummy0, dummy1;
        crossValidate(10, dummy0, dummy1);
        
        const FloatType nu    = params(0);
        //const FloatType gamma = params(1);
        
        svm_nu_trainer<kernel_type> trainer;
        //trainer.set_kernel(kernel_type(gamma));
        trainer.set_kernel(kernel_type()); // no gamma parameter for the linear kernel
        trainer.set_nu(nu);
        
        // dlib returns a decision function
        decfun = trainer.train(samples, labels);
        
        // linear kernel: convert it to proper weights for the hyperplane rather than support vectors
        weights.clear();
        weights.resize(dim+1, 0);
        
        matrix<FloatType> w(dim,1);
        w = 0;
        for (int i=0; i<decfun.alpha.nr(); ++i) {
            w += decfun.alpha(i) * decfun.basis_vectors(i);
        }
        for (int i=0; i<dim; ++i) weights[i] = w(i);
        weights[dim] = -decfun.b;
        

        // test validity of the method on a few samples
        for (int s=0; s<10; ++s) {
            // by hand
            FloatType p = 0;
            for (int i=0; i<dim; ++i) p += weights[i] * samples[s](i);
            p += weights[dim];
            cout << "with weights: " << p << ", with dfun: " << decfun(samples[s]) << endl;
        }

    }
    
    virtual FloatType predict(const FloatType* data) {}
    
    dlib::decision_function<kernel_type> decfun;
    vector<FloatType> weights;
};

bool fpeq(FloatType a, FloatType b) {
    static const FloatType epsilon = 1e-6;
    if (b==0) return fabs(a)<epsilon;
    FloatType ratio = a/b;
    return ratio>1-epsilon && ratio<1+epsilon;
}

int main(int argc, char** argv) {
    
    if (argc<5) return help();

    int nclasses = 1;
    vector<int> classboundaries(1,0);
    
    int total_pts = 0;

    int nscales = 0;
    vector<FloatType> scales;
    vector<FloatType> specifiedScales;
    
    ofstream classifparamsfile(argv[1], ofstream::binary);

    int arg_firstdata = argc;
    for (int argi = 2; argi<argc; ++argi) if (!strcmp(argv[argi],":")) {
        arg_firstdata = argi+1;
        break;
    }
    if (arg_firstdata>=argc) return help();
    
    for (int argi = 2; argi<arg_firstdata-1; ++argi) {
        FloatType scale = atof(argv[argi]);
        if (scale<=0) return help("invalid scale");
        specifiedScales.push_back(scale);
    }
    
    // read headers and ensures all files are consistent
    for (int argi = arg_firstdata; argi<argc; ++argi) {
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

    vector<int> specifiedScalesIdx(specifiedScales.size(), -1);
    for (int i=0; i<specifiedScales.size(); ++i) {
        for (int s=0; s<nscales; ++s) if (fpeq(specifiedScales[i],scales[s])) {
            specifiedScalesIdx[i] = s;
            break;
        }
        if (specifiedScalesIdx[i]<0) return help("Specified scale not found in the msc files");
    }

    // number of features
    int fdim = nscales * 2;

    // store all data in memory, keep overhead to a minimum as we plan for a lot of points...
    vector<FloatType> data(total_pts * fdim);
    int base_pt = 0;
    // and then fill data from the files
    for (int argi = arg_firstdata; argi<argc; ++argi) {
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
    vector<vector<int> > selectedScales(nclasses * (nclasses-1) / 2);
    set<int> uniqueScales;
    
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
            
            vector<int> bestscales;
            vector<int> usersidx;

            if (!specifiedScalesIdx.empty()) usersidx = specifiedScalesIdx;
            else {
                // scale by scale classif, look for scale leading to best classification and add more scales if necessary
                for (int nusedscales = 0; nusedscales < nscales; ++nusedscales) {
                    FloatType bestaccuracy_without_new_scale = bestaccuracy;
                    FloatType bestperf_without_new_scale = bestperf;
                    int bestscale = -1;
                    
                    for(int s=0; s<nscales; ++s) {
                        bool usedscale = false;
                        for (int i=0; i<bestscales.size(); ++i) if (bestscales[i]==s) {usedscale = true; break;}
                        if (usedscale) continue;

                        int nfeatures = (bestscales.size()+1)*2;
#ifdef USE_SVM
                        SVM_Model classifier;
#else
                        LeastSquareModel classifier;
#endif
                        classifier.prepareTraining(ntotal, nfeatures);

                        // fill the training data
                        vector<FloatType> mscdata(nfeatures);
                        for (int pt=0; pt<ni; ++pt) {
                            for (int i=0; i<bestscales.size(); ++i) {
                                mscdata[i*2] = data[(ibeg+pt)*fdim + bestscales[i]*2];
                                mscdata[i*2+1] = data[(ibeg+pt)*fdim + bestscales[i]*2+1];
                            }
                            mscdata[bestscales.size()*2] = data[(ibeg+pt)*fdim + s*2];
                            mscdata[bestscales.size()*2+1] = data[(ibeg+pt)*fdim + s*2+1];
                            classifier.addTrainData(&mscdata[0], -1);
                        }
                        for (int pt=0; pt<nj; ++pt) {
                            for (int i=0; i<bestscales.size(); ++i) {
                                mscdata[i*2] = data[(jbeg+pt)*fdim + bestscales[i]*2];
                                mscdata[i*2+1] = data[(jbeg+pt)*fdim + bestscales[i]*2+1];
                            }
                            mscdata[bestscales.size()*2] = data[(jbeg+pt)*fdim + s*2];
                            mscdata[bestscales.size()*2+1] = data[(jbeg+pt)*fdim + s*2+1];
                            classifier.addTrainData(&mscdata[0], 1);
                        }

                        classifier.shuffle();                

                        FloatType accuracy, perf;
                        classifier.crossValidate(10, accuracy, perf);
                        cout << "cv accuracy using scale";
                        if (bestscales.empty()) cout << " " << scales[s] << " only: ";
                        else {
                            cout << "s";
                            for (int i=0; i<bestscales.size(); ++i) cout << " " << scales[bestscales[i]];
                            cout << " and " << scales[s] << ": ";
                        }
                        cout << accuracy << " perf " << perf << endl;
                        
                        if (bestaccuracy < accuracy || (bestaccuracy == accuracy && bestperf < perf) ) {
                            bestaccuracy = accuracy;
                            bestperf = perf;
                            bestscale = s;
                        }
                    }
                    
                    if (bestaccuracy==bestaccuracy_without_new_scale && bestperf == bestperf_without_new_scale) break;
                    bestscales.push_back(bestscale);
                    // sort for display
                    sort(bestscales.begin(), bestscales.end());
                }
                
                cout << "Please enter one or more selected scales (default =";
                for (int i=0; i<bestscales.size(); ++i) cout << " " << scales[bestscales[i]];
                cout << "): " << endl;
            
                do {
                    string userscalestring;
                    getline(cin, userscalestring);
                    trim(userscalestring);
                    if (userscalestring.empty()) {usersidx = bestscales; break;}
                    vector<string> tokens;
                    split(tokens, userscalestring, is_any_of(", \t"));
                    for (vector<string>::iterator it = tokens.begin(); it!=tokens.end(); ++it) {
                        FloatType userscale = atof(it->c_str());
                        int scaleidx = -1;
                        for(int s=0; s<nscales; ++s) if (fpeq(scales[s],userscale)) {scaleidx = s; break;}
                        if (scaleidx==-1) {usersidx.clear(); break;}
                        usersidx.push_back(scaleidx);
                    }
                    if (usersidx.empty()) cout << "Invalid scale, please try again: " << endl;
                } while (usersidx.empty());
            }
            
            for (int i=0; i<usersidx.size(); ++i) uniqueScales.insert(usersidx[i]);
            
            // train a classifier at the selected scales on the whole data set
            int idx = jclass*(jclass-1)/2 + iclass;
#ifdef USE_SVM
            SVM_Model classifier;
#else
            LeastSquareModel classifier;
#endif
            classifier.prepareTraining(ntotal, usersidx.size()*2);
            // fill the training data
            vector<FloatType> mscdata(usersidx.size()*2);
            for (int pt=0; pt<ni; ++pt) {
                for (int i=0; i<usersidx.size(); ++i) {
                    mscdata[i*2] = data[(ibeg+pt)*fdim + usersidx[i]*2];
                    mscdata[i*2+1] = data[(ibeg+pt)*fdim + usersidx[i]*2+1];
                }
                classifier.addTrainData(&mscdata[0], -1);
            }
            for (int pt=0; pt<nj; ++pt) {
                for (int i=0; i<usersidx.size(); ++i) {
                    mscdata[i*2] = data[(jbeg+pt)*fdim + usersidx[i]*2];
                    mscdata[i*2+1] = data[(jbeg+pt)*fdim + usersidx[i]*2+1];
                }
                classifier.addTrainData(&mscdata[0], 1);
            }
            // no need to shuffle for training on the whole data set
            classifier.train();
            // store the weights and selected scale indices for later use
            weights[idx] = classifier.weights;
            selectedScales[idx] = usersidx;
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
            FloatType pred = weights[idx][selectedScales[idx].size()*2];
            for (int ssi = 0; ssi < selectedScales[idx].size(); ++ssi) {
                pred += ptdata[selectedScales[idx][ssi]*2] * weights[idx][ssi*2];
                pred += ptdata[selectedScales[idx][ssi]*2+1] * weights[idx][ssi*2+1];
            }
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
        int numscales = selectedScales[i].size(); // TODO: implement multi-scale classifiers
        classifparamsfile.write((char*)&numscales, sizeof(numscales));
        for (int j=0; j<numscales; ++j) classifparamsfile.write((char*)&scales[selectedScales[i][j]], sizeof(FloatType));
        // (2*n_used_scales + 1 coefs per classifier)
        for (int j=0; j<=2*numscales; ++j) classifparamsfile.write((char*)&weights[i][j], sizeof(FloatType));
    }
    
    return 0;
}


