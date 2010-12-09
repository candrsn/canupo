#ifndef CANUPO_LINEAR_SVM_HPP
#define CANUPO_LINEAR_SVM_HPP

#include <vector>
#include <limits>

#include "points.hpp"

#include "dlib/svm.h"

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
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_nu(nu);

            // Finally, perform 10-fold cross validation and then print and return the results.
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);

            return sum(result);
        }

        const std::vector<sample_type>& samples;
        const std::vector<FloatType>& labels;
        int nfolds;
    };


    FloatType crossValidate(int nfolds, const std::vector<sample_type>& samples, const std::vector<FloatType>& labels) {
        using namespace dlib;
        // taken from dlib examples
        
        // largest allowed nu: strictly below what's returned by maximum_nu
        double max_nu = 0.999*maximum_nu(labels);
        double min_nu = 1e-7;

        matrix<double> gridsearchspace = logspace(log10(min_nu), log10(max_nu), 50); // nu parameter
        matrix<FloatType> best_result(2,1);
        best_result = -std::numeric_limits<FloatType>::max();
        FloatType best_nu = 1;
        for (int col = 0; col < gridsearchspace.nc(); ++col) {
            const FloatType nu = gridsearchspace(0, col);
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_nu(nu);
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);
            if (sum(result) > sum(best_result)) {
                best_result = result;
                best_nu = nu;
            }
        }
        if (best_nu>max_nu) best_nu = max_nu;
        if (best_nu<min_nu) best_nu = min_nu;
        double lnu = log(best_nu);

        double best_score = dlib::find_max_single_variable(
            cross_validation_objective(samples, labels, nfolds), // Function to maximize
            lnu,              // starting point and result
            log(min_nu),          // lower bound
            log(max_nu),          // upper bound
            1e-2,
            50
        );
        return (FloatType)exp(lnu);
    }
    
    void train(int nfolds, FloatType nu, const std::vector<sample_type>& samples, const std::vector<FloatType>& labels) {
        using namespace dlib;
        svm_nu_trainer<kernel_type> trainer;
        trainer.set_nu(nu);
        trainer.set_kernel(kernel_type());
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
