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

#ifdef SVM_FAST_MODE
    FloatType trainer_batch_rate;
#endif

    struct cross_validation_objective {
        cross_validation_objective (
            const std::vector<sample_type>& samples_,
            const std::vector<FloatType>& labels_,
            int _nfolds,
            LinearSVM* _lsvm
        ) : samples(samples_), labels(labels_), nfolds(_nfolds), lsvm(_lsvm) {}

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
            matrix<FloatType> result = cross_validate_trainer_threaded(batch_cached(trainer,lsvm->trainer_batch_rate), samples, labels, nfolds, omp_get_num_threads());
#else
            matrix<FloatType> result = cross_validate_trainer(batch_cached(trainer,->trainer_batch_rate), samples, labels, nfolds);
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
        LinearSVM* lsvm;
    };


    FloatType crossValidate(int nfolds, const std::vector<sample_type>& samples, const std::vector<FloatType>& labels) {
        using namespace dlib;
        using namespace std;
        // taken from dlib examples
//#ifndef SVM_FAST_MODE

#ifdef SVM_FAST_MODE
        trainer_batch_rate = 0.1;//05 + tbri * 0.007;
        // nu is actually lambda
        double max_nu = 1e-2;
        double min_nu = 1e-6;
#else
        // largest allowed nu: strictly below what's returned by maximum_nu
        double max_nu = 0.999*maximum_nu(labels);
        double min_nu = max_nu * 1e-3;
#endif

        matrix<FloatType> best_result(2,1);
        best_result = -std::numeric_limits<FloatType>::max();
        int best_index = -1;
        int best_index_span = 0;

        int num_grid_nu = 25;
        double lmin = log(min_nu);
        double lmax = log(max_nu);
        for (int gidx = 0; gidx<num_grid_nu; ++gidx) {
            double nu = exp(lmin + (lmax - lmin) * gidx / (num_grid_nu - 1.0));
#ifdef SVM_FAST_MODE
            svm_pegasos<kernel_type> pegasos_trainer;
            pegasos_trainer.set_lambda(nu);
            pegasos_trainer.set_kernel(kernel_type());
            batch_trainer<svm_pegasos<kernel_type> > trainer = batch_cached(pegasos_trainer,trainer_batch_rate);
#else
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_nu(nu);
            trainer.set_kernel(kernel_type());
#endif
#ifdef _OPENMP
            matrix<FloatType> result = cross_validate_trainer_threaded(trainer, samples, labels, nfolds, omp_get_num_threads());
#else
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);
#endif
cout << "nu = " << nu << ", result=" << sum(result) << endl;
            if (sum(result) > sum(best_result)) {
                best_result = result;
                best_index = gidx;
                best_index_span = 0;
            }
            else if (sum(result) == sum(best_result)) {
                ++best_index_span;
            }
        }
        // take median of interval range with same best values
        double best_nu = exp(lmin + (lmax - lmin) * (best_index + best_index_span * 0.5) / (num_grid_nu - 1.0));
        
cout << "best_nu = " << best_nu << endl;
        // may exceed original min/max bounds
        // do it in linear scale
        double min2 = exp(lmin + (lmax - lmin) * (best_index - 1) / (num_grid_nu - 1.0));
        double max2 = exp(lmin + (lmax - lmin) * (best_index + best_index_span + 1) / (num_grid_nu - 1.0));
        best_index = -1;
        for (int gidx = 1; gidx<=num_grid_nu; ++gidx) {
            // take 2 more grid points corresponding to the previous steps already
            // computed, do not recompute them
            double nu = min2 + (max2 - min2) * gidx / (num_grid_nu + 1.0);
#ifdef SVM_FAST_MODE
            svm_pegasos<kernel_type> pegasos_trainer;
            pegasos_trainer.set_lambda(nu);
            pegasos_trainer.set_kernel(kernel_type());
            batch_trainer<svm_pegasos<kernel_type> > trainer = batch_cached(pegasos_trainer,trainer_batch_rate);
#else
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_nu(nu);
            trainer.set_kernel(kernel_type());
#endif
#ifdef _OPENMP
            matrix<FloatType> result = cross_validate_trainer_threaded(trainer, samples, labels, nfolds, omp_get_num_threads());
#else
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);
#endif
cout << "nu = " << nu << ", result=" << sum(result) << endl;
            if (sum(result) > sum(best_result)) {
                best_result = result;
                best_index = gidx;
                best_index_span = 0;
            }
            else if (sum(result) == sum(best_result)) {
                if (best_index==-1) best_index = gidx;
                ++best_index_span;
            }
        }
        // if no better index keep previous best_nu
        if (best_index>=0) best_nu = min2 + (max2 - min2) * (best_index + best_index_span * 0.5) / (num_grid_nu + 1.0);
cout << "best_nu = " << best_nu << endl;

        double lnu = log(best_nu);

        double best_score = dlib::find_max_single_variable(
            cross_validation_objective(samples, labels, nfolds,this), // Function to maximize
            lnu,              // starting point and result
            log(min_nu),          // lower bound
            log(max_nu),          // upper bound
            1e-3,
            50
        );
cout << "best_nu after find_max_single_variable = " << (FloatType)exp(lnu) << endl;
        return (FloatType)exp(lnu);

#if 0 //else // SVM_FAST_MODE
        double best_score = -1;
        double best_llambda = -4;
        cout << "Optimising the lambda parameter (SVM-pegasos algorithm) " << flush;
        //for (int tbri = 0; tbri <= 10; ++tbri) {
            cout << "." << flush;
            // batch rate from 0.05 to 0.12
            trainer_batch_rate = 0.1;//05 + tbri * 0.007;
            double llambda = -4;
            double score = find_max_single_variable(
                cross_validation_objective(samples, labels, nfolds, this), // Function to maximize
                llambda,              // starting point and result
                -6,          // lower bound, log(1e-6)
                -2,          // upper bound
                1e-4,
                100
            );
            if (score > best_score) {
                best_score = score;
                best_llambda = llambda;
            }
        //}
        cout << endl;
        return (FloatType)exp(best_llambda);

#endif
    }
    
    void train(int nfolds, FloatType nu, const std::vector<sample_type>& samples, const std::vector<FloatType>& labels) {
        using namespace dlib;
#ifdef SVM_FAST_MODE
        svm_pegasos<kernel_type> pegasos_trainer;
        pegasos_trainer.set_lambda(nu);
        pegasos_trainer.set_kernel(kernel_type());
        batch_trainer<svm_pegasos<kernel_type> > trainer = batch_cached(pegasos_trainer,trainer_batch_rate);
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
