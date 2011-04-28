#ifndef CANUPO_LINEAR_SVM_HPP
#define CANUPO_LINEAR_SVM_HPP

#include <iostream>
#include <vector>
#include <limits>

#include "points.hpp"

#ifdef _OPENMP
#include <omp.h>
#include "dlib/svm_threaded.h"
#include "dlib/threads/multithreaded_object_extension.cpp"
#include "dlib/threads/threaded_object_extension.cpp"
#include "dlib/threads/threads_kernel_1.cpp"
#include "dlib/threads/threads_kernel_2.cpp"
#include "dlib/threads/threads_kernel_shared.cpp"
#include "dlib/threads/thread_pool_extension.cpp"
#else
#include "dlib/svm.h"
#endif

struct LinearSVM {

    typedef dlib::matrix<FloatType, 0, 1> sample_type;
    typedef dlib::linear_kernel<sample_type> kernel_type;

    struct cross_validation_objective {
        cross_validation_objective (
            const std::vector<sample_type>& samples_,
            const std::vector<FloatType>& labels_,
            int _nfolds
        ) : samples(samples_), labels(labels_), nfolds(_nfolds) {}

        double operator() (FloatType lognu) const {
            using namespace dlib;
            // see below for changes from dlib examples
            const FloatType nu = exp(lognu);

            // Make an SVM trainer and tell it what the parameters are supposed to be.
#ifdef SVM_FAST_MODE
            svm_pegasos<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_lambda(nu);
#ifdef _OPENMP
            matrix<FloatType> result = cross_validate_trainer_threaded(batch_cached(trainer,0.1), samples, labels, nfolds, omp_get_num_threads());
#else
            matrix<FloatType> result = cross_validate_trainer(batch_cached(trainer,0.1), samples, labels, nfolds);
#endif
#else
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_nu(nu);
#ifdef _OPENMP
            matrix<FloatType> result = cross_validate_trainer_threaded(trainer, samples, labels, nfolds, omp_get_num_threads());
#else
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);
#endif
#endif

            return sum(result);
        }

        const std::vector<sample_type>& samples;
        const std::vector<FloatType>& labels;
        int nfolds;
    };


    FloatType crossValidate(int nfolds, const std::vector<sample_type>& samples, const std::vector<FloatType>& labels) {
        using namespace dlib;
        using namespace std;
#ifndef SVM_FAST_MODE
        // taken from dlib examples

        // largest allowed nu: strictly below what's returned by maximum_nu
        double max_nu = 0.999*maximum_nu(labels);
        double min_nu = 1e-7;

        matrix<double> gridsearchspace = logspace(log10(min_nu), log10(max_nu), 10); // nu parameter
        matrix<FloatType> best_result(2,1);
        best_result = -numeric_limits<FloatType>::max();
        FloatType best_nu = 1;

        cout << "Cross-validating best SVM parameters (pass 1)... 0" << flush;
        int nextpercentcomplete = 5;
        int niter = gridsearchspace.nc();
        int nrealised = 0;
        #pragma omp parallel for
        for (int col = 0; col < niter; ++col) {
            const FloatType nu = gridsearchspace(0, col);
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_nu(nu);
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);
            #pragma omp critical
            {
                if (sum(result) > sum(best_result)) {
                    best_result = result;
                    best_nu = nu;
                }
                ++nrealised;
                int percentcomplete = (nrealised * 100) / niter;
                while (percentcomplete>=nextpercentcomplete) {
                    if (nextpercentcomplete % 10 == 0) cout << nextpercentcomplete << flush;
                    else if (nextpercentcomplete % 5 == 0) cout << "." << flush;
                    nextpercentcomplete+=5;
                }
            }
        }
        cout << endl;
        if (best_nu>max_nu) best_nu = max_nu;
        if (best_nu<min_nu) best_nu = min_nu;
        double lnu = log(best_nu);
        
        cout << "Cross-validating best SVM parameters (pass 2)... 0" << flush;
        double grid_plus_minus_step = (log10(max_nu) - log10(min_nu)) / 20;
        gridsearchspace = logspace(lnu - grid_plus_minus_step, lnu + grid_plus_minus_step, 10);
        niter = gridsearchspace.nc();
        nrealised = 0;
        
        nextpercentcomplete = 5;
        #pragma omp parallel for
        for (int col = 0; col < gridsearchspace.nc(); ++col) {
            const FloatType nu = gridsearchspace(0, col);
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_nu(nu);
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);
            #pragma omp critical
            {
                if (sum(result) > sum(best_result)) {
                    best_result = result;
                    best_nu = nu;
                }
                ++nrealised;
                int percentcomplete = (nrealised * 100) / niter;
                while (percentcomplete>=nextpercentcomplete) {
                    if (nextpercentcomplete % 10 == 0) cout << nextpercentcomplete << flush;
                    else if (nextpercentcomplete % 5 == 0) cout << "." << flush;
                    nextpercentcomplete+=5;
                }
            }
        }
        cout << endl;
        if (best_nu>max_nu) best_nu = max_nu;
        if (best_nu<min_nu) best_nu = min_nu;
        
        lnu = log(best_nu);

        cout << "Optimising the nu parameter (SVM-nu formulation)" << endl;

        double best_score = find_max_single_variable(
            cross_validation_objective(samples, labels, nfolds), // Function to maximize
            lnu,              // starting point and result
            log(min_nu),          // lower bound
            log(max_nu),          // upper bound
            1e-2,
            50
        );
        return (FloatType)exp(lnu);

#else // SVM_FAST_MODE
        cout << "Optimising the lambda parameter (SVM-pegasos algorithm) " << endl;
        double llambda = -4;
        double best_score = find_max_single_variable(
            cross_validation_objective(samples, labels, nfolds), // Function to maximize
            llambda,              // starting point and result
            -6,          // lower bound, log(1e-6)
            -2,          // upper bound
            1e-2,
            50
        );
        return (FloatType)exp(llambda);
#endif
    }
    
    void train(int nfolds, FloatType nu, const std::vector<sample_type>& samples, const std::vector<FloatType>& labels) {
        using namespace dlib;
#ifdef SVM_FAST_MODE
        svm_pegasos<kernel_type> pegasos_trainer;
        pegasos_trainer.set_lambda(nu);
        pegasos_trainer.set_kernel(kernel_type());
        batch_trainer<svm_pegasos<kernel_type> > trainer = batch_cached(pegasos_trainer,0.1);
#else
        svm_nu_trainer<kernel_type> trainer;
        trainer.set_nu(nu);
        trainer.set_kernel(kernel_type());
#endif
        int dim = samples.back().size();
        // linear kernel: convert the decision function to an hyperplane rather than support vectors
        // This is equivalent but way more efficient for later scene classification
        //decision_function<kernel_type> decfun = trainer.train(samples, labels);
        probabilistic_decision_function<kernel_type> pdecfun = train_probabilistic_decision_function(trainer, samples, labels, nfolds);
        decision_function<kernel_type>& decfun = pdecfun.decision_funct;
        
        weights.clear();
        weights.resize(dim+1, 0);
        matrix<FloatType> w(dim,1);
        w = 0;
        for (int i=0; i<decfun.alpha.nr(); ++i) {
            w += decfun.alpha(i) * decfun.basis_vectors(i);
        }
        for (int i=0; i<dim; ++i) weights[i] = w(i);
        weights[dim] = -decfun.b;
        
        // p(x) = 1/(1+exp(alpha*d(x)+beta)) 
        // with d(x) the oriented dist the decision function
        // linear kernel: equivalently shift and scale the values
        // d(x) = w.x + w[dim]
        // So in the final classifier we have d(x)=0 matching the probabilistic decfun
        for (int i=0; i<=dim; ++i) weights[i] *= pdecfun.alpha;
        weights[dim] += pdecfun.beta;
/*
        // checking for a few samples
        for (int i=0; i<10; ++i) {
            cout << pdecfun(samples[i]) << " ";
            FloatType dx = weights[dim];
            for (int j=0; j<dim; ++j) dx += weights[j] * samples[i](j);
            dx = 1 / (1+exp(dx));
            cout << dx << endl;
        }
*/
        // revert the decision function so we compute proba to be in the -1 class, i.e.
        // the first class given to the classifier program
        for (int i=0; i<=dim; ++i) weights[i] = -weights[i];
        
        // note: we now have comparable proba for dx and dy
        //       as 1 / (1+exp(dx)) = 1 / (1+exp(dy)) means same dx and dy
        //       in this new space
        // => consistant orthogonal axis
    }
    
    FloatType predict(const sample_type& data) {
        int dim = weights.size()-1;
        FloatType ret = weights[dim];
        for (int d=0; d<dim; ++d) ret += weights[d] * data(d);
        return ret;
    }
    
    std::vector<FloatType> weights;
};


#endif
