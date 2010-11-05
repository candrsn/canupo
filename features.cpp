#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <string.h>
#include <assert.h>
#include <stdlib.h>

#ifdef PROJ_USER_CLASSIF
#define LINEAR_SVM 1
#endif

#if defined(LINEAR_SVM) || defined(GAUSSIAN_SVM)
#define USE_SVM 1
#endif

#include "points.hpp"

#ifdef USE_SVM
#include "dlib/svm.h"
#else
#include "leastSquares.hpp"
#endif

using namespace std;
using namespace boost;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << ""
#if defined(LINEAR_SVM)
"features_linear_svm"
#elif defined(GAUSSIAN_SVM)
"features_gaussian_svm"
#else
"features_least_squares"
#endif
" features.prm [scales] : data1.msc data2.msc - data3.msc - data4.msc...\n\
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

#ifndef USE_SVM
// simple linear fit trying to map data to -1,+1 classes with least squared error
struct LeastSquareModel {
    
    // parameters are all that's needed to make a prediction
    struct Parameters {
        vector<FloatType> weights;
        int dim; // redundant but handy, as weights.size() == dim+1
    };
    
    Parameters parameters; // weights
    
    vector<FloatType> A, B; // matrices for lapack
    int ndata, dataidx;
    
    void prepareTraining(int _ndata, int _dim) {
        ndata = _ndata; parameters.dim = _dim;
        assert(ndata>parameters.dim);
        // LAPACK matrices are column-major and destroyed in the process => allocate new matrices now
        // add a column of 1 for allowing hyperplanes not going through the origin
        A.resize(ndata * (parameters.dim+1));
        B.resize(ndata);
        dataidx = 0;
    }
    
    // data shall point to an array of dim floats. label shall be either +1 or -1
    void addTrainData(const FloatType* data, FloatType label) {
        for (int d=0; d<parameters.dim; ++d) A[d*ndata + dataidx] = data[d];
        A[parameters.dim*ndata + dataidx] = 1;
        B[dataidx] = label;
        ++dataidx;
    }
    
    void train() {
        // now the least squares hyperplane fit
        leastSquares(&A[0], ndata, parameters.dim+1, &B[0], 1);
        parameters.weights.resize(parameters.dim+1);
        for (int d=0; d<=parameters.dim; ++d) parameters.weights[d] = B[d];
    }
    
    FloatType predict(const FloatType* data) {
        FloatType ret = parameters.weights[parameters.dim];
        for (int d=0; d<parameters.dim; ++d) ret += parameters.weights[d] * data[d];
        return ret;
    }

    // shuffling the data order is necessary so classes are equally distributed in each fold region
    void shuffle() {
        for (int i=ndata-1; i>=1; --i) {
            // may select i itself for the permutation if the element does not change place
            int ridx = random() % (i+1);
            for (int d=0; d<parameters.dim; ++d) swap(A[d * ndata + ridx], A[d * ndata + i]);
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
            vector<FloatType> Acv(nremdata * (parameters.dim+1));
            vector<FloatType> Bcv(nremdata);
            for (int d=0; d<parameters.dim; ++d) {
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
            for (unsigned int i = parameters.dim * nremdata; i<Acv.size(); ++i) Acv[i] = 1.0;
            // train only on partial data
            leastSquares(&Acv[0], nremdata, parameters.dim+1, &Bcv[0], 1);
            // predict on the cv region
            for (int pt=fbeg; pt<fend; ++pt) {
                FloatType pred = Bcv[parameters.dim];
                for (int d=0; d<parameters.dim; ++d) pred += Bcv[d] * A[d * ndata + pt];
                if (B[pt] >= 0) {if (pred >= 0) ++ncorrectpos; perfpos += pred*fabs(pred);}
                if (B[pt] < 0) {if (pred < 0) ++ncorrectneg; perfneg -= pred*fabs(pred);}
            }
        }
        // return the equilibrated performance
        accuracy = 0.5 * (ncorrectpos / (FloatType)npos + ncorrectneg / (FloatType)nneg);
        perf = 0.5 * (perfpos / npos + perfneg / nneg);
    }

    enum {ClassifierID = 0};
    
    void saveParameters(ostream& os) {
        os.write((char*)&parameters.dim, sizeof(int));
        for (int d=0; d<=parameters.dim; ++d) os.write((char*)&parameters.weights[d],sizeof(FloatType));
    }
};
#endif

#ifdef USE_SVM
struct SVM_Model {

#ifdef GAUSSIAN_SVM
    enum {ClassifierID = 2};
#else
    enum {ClassifierID = 1};
#endif

    // see dlib examples
    template <typename sample_type>
    struct cross_validation_objective {
        cross_validation_objective (
            const vector<sample_type>& samples_,
            const vector<double>& labels_,
            int _nfolds
        ) : samples(samples_), labels(labels_), nfolds(_nfolds) {}

#ifdef GAUSSIAN_SVM
        double operator() (const dlib::matrix<double>& params) const {
#else
        double operator() (FloatType lognu) const {
#endif
            using namespace dlib;
            // see below for changes from dlib examples
#ifdef GAUSSIAN_SVM
            const FloatType nu    = exp(params(0));
            const FloatType gamma = exp(params(1));
#else
            const FloatType nu    = exp(lognu);
#endif

            // Make an SVM trainer and tell it what the parameters are supposed to be.
            svm_nu_trainer<kernel_type> trainer;
#ifdef GAUSSIAN_SVM
            trainer.set_kernel(kernel_type(gamma));
#else
            trainer.set_kernel(kernel_type());
#endif
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
    int ndata, dataidx;
    
#ifdef GAUSSIAN_SVM
    typedef dlib::radial_basis_kernel<sample_type> kernel_type;
#else
    typedef dlib::linear_kernel<sample_type> kernel_type;
#endif

    vector<sample_type> samples;
    vector<FloatType> labels;
    
    void prepareTraining(int _ndata, int _dim) {
        ndata = _ndata; parameters.dim = _dim;
        // prepare one sample with the right dimensions as an argument to the vector init
        sample_type undefsample; undefsample.set_size(parameters.dim,1);
        // use a vector of samples as in the dlib examples.
        samples.clear();
        samples.resize(ndata, undefsample);
        labels.resize(ndata);
        dataidx = 0;
    }
    
    void addTrainData(const FloatType* data, FloatType label) {
        for (int i=0; i<parameters.dim; ++i) samples[dataidx](i) = data[i];
        labels[dataidx] = label;
        ++dataidx;
    }
    
    void shuffle() {
        dlib::randomize_samples(samples, labels);
    }
    
    void crossValidate(int nfolds, FloatType &accuracy, FloatType &perf) {
        using namespace dlib;
        // taken from dlib examples
        
        // largest allowed nu: strictly below what's returned by maximum_nu
        FloatType max_nu = 0.999*maximum_nu(labels);
        
#ifdef GAUSSIAN_SVM
        matrix<FloatType> gridsearchspace = cartesian_product(
            logspace(log10(max_nu), log10(1e-5), 5), // nu parameter
            logspace(log10(5.0), log10(1e-5), 5)     // gamma parameter
        );
#else
        matrix<FloatType> gridsearchspace = logspace(log10(max_nu), log10(1e-5), 16); // nu parameter
#endif


        matrix<FloatType> best_result(2,1);
        best_result = 0;
        FloatType best_nu = 1;
#ifdef GAUSSIAN_SVM
        FloatType best_gamma = 0.1;
#endif
        
        // grid search
        for (int col = 0; col < gridsearchspace.nc(); ++col) {
            // pull out the current set of model parameters
            const FloatType nu    = gridsearchspace(0, col);
#ifdef GAUSSIAN_SVM
            const FloatType gamma = gridsearchspace(1, col);
#endif

            // setup a training object using our current parameters
            svm_nu_trainer<kernel_type> trainer;
#ifdef GAUSSIAN_SVM
            trainer.set_kernel(kernel_type(gamma));
#else
            trainer.set_kernel(kernel_type()); // no gamma parameter for the linear kernel
#endif
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
#ifdef GAUSSIAN_SVM
                best_gamma = gamma;
#endif
            }
        }
        //cout << "best result of grid search: " << sum(best_result) << endl;
        //cout << "best gamma: " << best_gamma << "   best nu: " << best_nu << endl;
        //cout << "best nu: " << best_nu << endl;
        
#ifdef GAUSSIAN_SVM
        matrix<FloatType,2,1> params;
        params = best_nu, best_gamma;
        // We also need to supply lower and upper bounds for the search.  
        matrix<FloatType,2,1> lower_bound, upper_bound;
        lower_bound = 1e-7,   // smallest allowed nu
                      1e-7;   // smallest allowed gamma
        upper_bound = max_nu, // largest allowed nu
                      100;    // largest allowed gamma
        // convert to log space
        params = log(params);
        lower_bound = log(lower_bound);
        upper_bound = log(upper_bound);
        // Start from the best grid point and launch the appropriate optimiser
        FloatType best_score = find_max_bobyqa(
            cross_validation_objective<sample_type>(samples, labels, nfolds), // Function to maximize
            params,                                      // starting point and result
            params.size()*2 + 1,                         // See BOBYQA docs, generally size*2+1 is a good setting for this
            lower_bound,                                 // lower bound 
            upper_bound,                                 // upper bound
            min(upper_bound-lower_bound)/10,             // search radius
            0.01,                                        // desired accuracy
            100                                          // max number of allowable calls to cross_validation_objective()
            );
        // Don't forget to convert back from log scale to normal scale
        parameters.nu = exp(params(0));
        parameters.gamma = exp(params(1));
#else
        parameters.nu = log(best_nu);
        FloatType best_score = find_max_single_variable(
            cross_validation_objective<sample_type>(samples, labels, nfolds), // Function to maximize
            parameters.nu,              // starting point and result
            log(1e-7),                  // lower bound
            log(max_nu),                // upper bound
            1e-2,
            100
        );
        parameters.nu = exp(parameters.nu);
#endif


//        cout << " best result of BOBYQA: " << best_score << endl;
//        cout << " best gamma: " << params(1) << "   best nu: " << params(0) << endl;
//        cout << " best result of find_max_single_variable: " << best_score << endl;
//        cout << " best nu: " << params(0) << endl;

        accuracy = best_score * 0.5;
        perf = best_score * 0.5; // TODO: find better measure
    }
    
    void train() {
        using namespace dlib;
        
        // we first need to select a good set of SVM parameters => crossvalidation no matter what
        shuffle();
        FloatType dummy0, dummy1;
        crossValidate(10, dummy0, dummy1);
        
        svm_nu_trainer<kernel_type> trainer;
        trainer.set_nu(parameters.nu);
        
#ifdef GAUSSIAN_SVM
        trainer.set_kernel(kernel_type(parameters.gamma));
        // dlib returns a decision function
        parameters.decfun = trainer.train(samples, labels);
#else
        trainer.set_kernel(kernel_type());
        // linear kernel: convert the decision function to an hyperplane rather than support vectors
        // This is equivalent but way more efficient for later scene classification
        dlib::decision_function<kernel_type> decfun = trainer.train(samples, labels);
        parameters.weights.clear();
        parameters.weights.resize(parameters.dim+1, 0);
        matrix<FloatType> w(parameters.dim,1);
        w = 0;
        for (int i=0; i<decfun.alpha.nr(); ++i) {
            w += decfun.alpha(i) * decfun.basis_vectors(i);
        }
        for (int i=0; i<parameters.dim; ++i) parameters.weights[i] = w(i);
        parameters.weights[parameters.dim] = -decfun.b;
#endif

    }
    
    struct Parameters {
        int dim;
        FloatType nu;
#ifdef GAUSSIAN_SVM
        FloatType gamma;
        dlib::decision_function<kernel_type> decfun;
#else
        vector<FloatType> weights;
#ifdef PROJ_USER_CLASSIF
        vector<FloatType> orthoweights;
#endif
#endif
    };
    
    Parameters parameters;
    
    FloatType predict(const FloatType* data) {
#ifdef GAUSSIAN_SVM
        dlib::matrix<FloatType> x(parameters.dim,1);
        for (int d=0; d<parameters.dim; ++d) x(d) = data[d];
        return parameters.decfun(x);
#else
        FloatType ret = parameters.weights[parameters.dim];
        for (int d=0; d<parameters.dim; ++d) ret += parameters.weights[d] * data[d];
        return ret;
#endif
    }
    
    void saveParameters(ostream& os) {
        os.write((char*)&parameters.dim, sizeof(int));
#ifdef GAUSSIAN_SVM
        // dlib::serialize(parameters.decfun, os);
        // the serialize function is buggy !
        // do it tediously
        os.write((char*)&parameters.gamma, sizeof(FloatType));
        FloatType bias = parameters.decfun.b;
        os.write((char*)&bias, sizeof(FloatType));
        int nalpha = parameters.decfun.alpha.nr();
        os.write((char*)&nalpha, sizeof(int));
        for (int i=0; i<nalpha; ++i) {
            FloatType alpha = parameters.decfun.alpha(i);
            os.write((char*)&alpha, sizeof(FloatType));
            for (int j=0; j<parameters.dim; ++j) {
                FloatType v = parameters.decfun.basis_vectors(i)(j);
                os.write((char*)&v, sizeof(FloatType));
            }
        }        
#else
        for (int d=0; d<=parameters.dim; ++d) os.write((char*)&parameters.weights[d],sizeof(FloatType));
#endif
    }
};
#endif

bool fpeq(FloatType a, FloatType b) {
    static const FloatType epsilon = 1e-6;
    if (b==0) return fabs(a)<epsilon;
    FloatType ratio = a/b;
    return ratio>1-epsilon && ratio<1+epsilon;
}

#ifdef USE_SVM
typedef SVM_Model Classifier;
#else
typedef LeastSquareModel Classifier;
#endif

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
    vector<Classifier::Parameters > params(nclasses * (nclasses-1) / 2);
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

#ifndef PROJ_USER_CLASSIF
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
                        
                        Classifier classifier;
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
#else
            usersidx.resize(nscales);
            for(int s=0; s<nscales; ++s) usersidx[s] = s;
#endif

            for (int i=0; i<usersidx.size(); ++i) uniqueScales.insert(usersidx[i]);
            
            // train a classifier at the selected scales on the whole data set
            int idx = jclass*(jclass-1)/2 + iclass;

            Classifier classifier;

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
            // store the parameters and selected scale indices for later use
            params[idx] = classifier.parameters;
            selectedScales[idx] = usersidx;
            
#ifdef PROJ_USER_CLASSIF
            // projection onto orthogonal subspace and repeat SVM
            
            // projection parameters
            vector<FloatType> nvec(classifier.parameters.dim); // normal vector
            FloatType norm = 0;
            for (int i=0; i<nvec.size(); ++i) {
                nvec[i] = classifier.parameters.weights[i];
                norm += nvec[i] * nvec[i];
            }
            norm = sqrt(norm);
            for (int i=0; i<nvec.size(); ++i) nvec[i] /= norm;
            vector<FloatType> svec(classifier.parameters.dim); // shift vector
            // dot product between normal and shift is given by the bias
            FloatType sndot = -classifier.parameters.weights[classifier.parameters.dim] / norm;
            for (int i=0; i<nvec.size(); ++i) svec[i] = sndot * nvec[i];
            
            Classifier ortho_classifier;
            ortho_classifier.prepareTraining(ntotal, usersidx.size()*2);
            
            // fill the training data
            vector<FloatType> proj_mscdata(usersidx.size()*2);
            for (int pt=0; pt<ni; ++pt) {
                for (int i=0; i<usersidx.size(); ++i) {
                    mscdata[i*2] = data[(ibeg+pt)*fdim + usersidx[i]*2];
                    mscdata[i*2+1] = data[(ibeg+pt)*fdim + usersidx[i]*2+1];
                }
                // projection on the first classifier hyperplane
                FloatType dotprod = 0;
                for(int i=0; i<nvec.size(); ++i) dotprod += nvec[i] * (mscdata[i] - svec[i]);
                for(int i=0; i<nvec.size(); ++i) proj_mscdata[i] = mscdata[i] - dotprod * nvec[i];
                ortho_classifier.addTrainData(&proj_mscdata[0], -1);
            }
            for (int pt=0; pt<nj; ++pt) {
                for (int i=0; i<usersidx.size(); ++i) {
                    mscdata[i*2] = data[(jbeg+pt)*fdim + usersidx[i]*2];
                    mscdata[i*2+1] = data[(jbeg+pt)*fdim + usersidx[i]*2+1];
                }
                FloatType dotprod = 0;
                for(int i=0; i<nvec.size(); ++i) dotprod += nvec[i] * (mscdata[i] - svec[i]);
                for(int i=0; i<nvec.size(); ++i) proj_mscdata[i] = mscdata[i] - dotprod * nvec[i];
                ortho_classifier.addTrainData(&proj_mscdata[0], 1);
            }
            // no need to shuffle for training on the whole data set
            ortho_classifier.train();
            // store the weights and selected scale indices for later use
            params[idx].orthoweights = ortho_classifier.parameters.weights;

            // write data files
            ofstream dataout(str(boost::format("class_%d_%d.txt") % iclass % jclass).c_str());
            for (int pt=0; pt<ni; ++pt) {
                for (int i=0; i<usersidx.size(); ++i) {
                    mscdata[i*2] = data[(ibeg+pt)*fdim + usersidx[i]*2];
                    mscdata[i*2+1] = data[(ibeg+pt)*fdim + usersidx[i]*2+1];
                }
                // projection on the first classifier hyperplane
                FloatType dotprod = 0;
                for(int i=0; i<nvec.size(); ++i) dotprod += nvec[i] * (mscdata[i] - svec[i]);
                for(int i=0; i<nvec.size(); ++i) proj_mscdata[i] = mscdata[i] - dotprod * nvec[i];
                dataout << classifier.predict(&mscdata[0]);
                dataout << " " << ortho_classifier.predict(&proj_mscdata[0]);
                dataout << " " << -1 << endl;
            }
            for (int pt=0; pt<nj; ++pt) {
                for (int i=0; i<usersidx.size(); ++i) {
                    mscdata[i*2] = data[(jbeg+pt)*fdim + usersidx[i]*2];
                    mscdata[i*2+1] = data[(jbeg+pt)*fdim + usersidx[i]*2+1];
                }
                FloatType dotprod = 0;
                for(int i=0; i<nvec.size(); ++i) dotprod += nvec[i] * (mscdata[i] - svec[i]);
                for(int i=0; i<nvec.size(); ++i) proj_mscdata[i] = mscdata[i] - dotprod * nvec[i];
                dataout << classifier.predict(&mscdata[0]);
                dataout << " " << ortho_classifier.predict(&proj_mscdata[0]);
                dataout << " " << 1 << endl;
            }
            dataout.close();
#endif
        }
    }
#ifdef PROJ_USER_CLASSIF
    cout << "Projected data for each set of classes written in corresponding files" << endl;
    cout << "You need to run features_user_validate on the SVG files to produce the final classifier." << endl;
    return 0;
#endif

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
            vector<FloatType> mscdata(selectedScales[idx].size()*2);
            for (int ssi=0; ssi<selectedScales[idx].size(); ++ssi) {
                mscdata[ssi*2] = ptdata[selectedScales[idx][ssi]*2];
                mscdata[ssi*2+1] = ptdata[selectedScales[idx][ssi]*2+1];
            }
            Classifier classifier;
            classifier.parameters = params[idx];
            FloatType pred = classifier.predict(&mscdata[0]);
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
    // write the classifier type
    // write for each of the n(n-1)/2 classifiers 
    // - num of scales used by this classifier
    // - the scales used 
    // - the classifier parameters (ex: weights for linear classifiers)
    int nuniqueScales = uniqueScales.size();
    classifparamsfile.write((char*)&nuniqueScales, sizeof(nuniqueScales));
    cout << "Selected scales (for extraction of the whole scene):";
    for (set<int>::iterator it = uniqueScales.begin(); it!=uniqueScales.end(); ++it) {
        cout << " " << scales[*it];
        classifparamsfile.write((char*)&scales[*it], sizeof(FloatType));
    }
    cout << endl;
    classifparamsfile.write((char*)&nclasses, sizeof(nclasses));
    // write the classifier type
    int classifierType = Classifier::ClassifierID;
    classifparamsfile.write((char*)&classifierType, sizeof(int));
    // and the parameters for each pair of classes
    int nclassifiers = nclasses * (nclasses-1) / 2;
    for (int i=0; i<nclassifiers; ++i) {
        int numscales = selectedScales[i].size(); // TODO: implement multi-scale classifiers
        classifparamsfile.write((char*)&numscales, sizeof(numscales));
        for (int j=0; j<numscales; ++j) classifparamsfile.write((char*)&scales[selectedScales[i][j]], sizeof(FloatType));
        Classifier classifier;
        classifier.parameters = params[i];
        classifier.saveParameters(classifparamsfile);
    }
    
    return 0;
}


