//**********************************************************************
//* This file is a part of the CANUPO project, a set of programs for   *
//* classifying automatically 3D point clouds according to the local   *
//* multi-scale dimensionality at each point.                          *
//*                                                                    *
//* Author & Copyright: Nicolas Brodu <nicolas.brodu@numerimoire.net>  *
//*                                                                    *
//* This project is free software; you can redistribute it and/or      *
//* modify it under the terms of the GNU Lesser General Public         *
//* License as published by the Free Software Foundation; either       *
//* version 2.1 of the License, or (at your option) any later version. *
//*                                                                    *
//* This library is distributed in the hope that it will be useful,    *
//* but WITHOUT ANY WARRANTY; without even the implied warranty of     *
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU  *
//* Lesser General Public License for more details.                    *
//*                                                                    *
//* You should have received a copy of the GNU Lesser General Public   *
//* License along with this library; if not, write to the Free         *
//* Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,    *
//* MA  02110-1301  USA                                                *
//*                                                                    *
//**********************************************************************/
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

    typedef dlib::matrix<double, 0, 1> sample_type;
    typedef dlib::linear_kernel<sample_type> kernel_type;

    int grid_size;
    double trainer_batch_rate;
    
    LinearSVM(int _grid_size) : grid_size(_grid_size), trainer_batch_rate(0.1) {}

    struct cross_validation_objective {
        cross_validation_objective (
            const std::vector<sample_type>& samples_,
            const std::vector<double>& labels_,
            int _nfolds,
            LinearSVM* _lsvm
        ) : samples(samples_), labels(labels_), nfolds(_nfolds), lsvm(_lsvm) {}

        double operator() (double logarg) const {
            using namespace dlib;
            // see below for changes from dlib examples
            const double arg = exp(logarg);
            matrix<double> result;

            // Make an SVM trainer and tell it what the parameters are supposed to be.
#ifdef SVM_NU_TRAINER
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_nu(arg);

#if defined(_OPENMP)
            // multi-thread here only if not at the caller level
            if (lsvm->grid_size==1) {
                result = cross_validate_trainer_threaded(trainer, samples, labels, nfolds, omp_get_num_threads());
            } else
#endif
            result = cross_validate_trainer(trainer, samples, labels, nfolds);
            
#else
            svm_pegasos<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_lambda(arg);

#if defined(_OPENMP)
            // multi-thread here only if not at the caller level
            if (lsvm->grid_size==1) {
                result = cross_validate_trainer_threaded(batch_cached(trainer,lsvm->trainer_batch_rate), samples, labels, nfolds, omp_get_num_threads());
            } else
#endif
            result = cross_validate_trainer(batch_cached(trainer,lsvm->trainer_batch_rate), samples, labels, nfolds);
            
#endif
            return sum(result);
        }

        const std::vector<sample_type>& samples;
        const std::vector<double>& labels;
        int nfolds;
        LinearSVM* lsvm;
    };


    double crossValidate(int nfolds, const std::vector<sample_type>& samples, const std::vector<double>& labels) {
        using namespace dlib;
        using namespace std;

        double best_score = -1;
        double best_logarg = log(1e-4);
        cout << "Optimising the SVM" << flush;
#ifdef SVM_NU_TRAINER
        // largest allowed nu: strictly below what's returned by maximum_nu
        double max_arg = 0.999*maximum_nu(labels);
        double min_arg = max_arg * 1e-3;
        double lmax = log(max_arg);
        double lmin = log(min_arg);
#else
        // arg is actually lambda
        double lmin = log(1e-6);
        double lmax = log(1e-2);
#endif
        #pragma omp parallel for schedule(dynamic)
        for (int gidx = 1; gidx<=grid_size; ++gidx) {
            cout << "." << flush;
            double larg = lmin + (lmax - lmin) * gidx / (grid_size + 1.0);
            double score = find_max_single_variable(
                cross_validation_objective(samples, labels, nfolds, this), // Function to maximize
                larg,          // starting point and result
                lmin,          // lower bound, log(1e-6)
                lmax,          // upper bound
                // precision (here on the sum of both class accuracies)
                2e-4,
                100            // max number of iterations
            );
            #pragma omp critical
            if (score > best_score) {
                best_score = score;
                best_logarg = larg;
            }
        }
        cout << endl;
        cout << "cross-validated balanced accuracy = " << 0.5 * best_score << endl;
        return (double)exp(best_logarg);
    }
    
    void train(int nfolds, double nu, const std::vector<sample_type>& samples, const std::vector<double>& labels) {
        using namespace dlib;
#ifdef SVM_NU_TRAINER
        svm_nu_trainer<kernel_type> trainer;
        trainer.set_nu(nu);
        trainer.set_kernel(kernel_type());
#else
        svm_pegasos<kernel_type> pegasos_trainer;
        pegasos_trainer.set_lambda(nu);
        pegasos_trainer.set_kernel(kernel_type());
        batch_trainer<svm_pegasos<kernel_type> > trainer = batch_cached(pegasos_trainer,trainer_batch_rate);
#endif
        int dim = samples.back().size();
        // linear kernel: convert the decision function to an hyperplane rather than support vectors
        // This is equivalent but way more efficient for later scene classification
        //decision_function<kernel_type> decfun = trainer.train(samples, labels);
        probabilistic_decision_function<kernel_type> pdecfun = train_probabilistic_decision_function(trainer, samples, labels, nfolds);
        decision_function<kernel_type>& decfun = pdecfun.decision_funct;
        
        weights.clear();
        weights.resize(dim+1, 0);
        matrix<double> w(dim,1);
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

        // revert the decision function so we compute proba to be in the -1 class, i.e.
        // the first class given to the classifier program
        for (int i=0; i<=dim; ++i) weights[i] = -weights[i];
        
        // note: we now have comparable proba for dx and dy
        //       as 1 / (1+exp(dx)) = 1 / (1+exp(dy)) means same dx and dy
        //       in this new space
        // => consistant orthogonal axis
    }
    
    double predict(const sample_type& data) {
        int dim = weights.size()-1;
        double ret = weights[dim];
        for (int d=0; d<dim; ++d) ret += weights[d] * data(d);
        return ret;
    }
    
    std::vector<double> weights;
};


#endif
