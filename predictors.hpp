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
#ifndef CANUPO_PREDICTORS_HPP
#define CANUPO_PREDICTORS_HPP

#include <iostream>

#include <boost/make_shared.hpp>

#include <stdlib.h>

#include "points.hpp"


/* THIS SECTION SHALL BE CONSISTENT WITH features_XXX programs 
   TODO: abstract a bit more and share common headers
*/

struct Predictor {
    virtual ~Predictor() {}
    virtual void load(std::istream& is) = 0;
    virtual FloatType predict(const FloatType* data) = 0;
};

struct LinearPredictor : public Predictor {
    int dim;
    std::vector<FloatType> weights;
    
    void load(std::istream& is) {
        is.read((char*)&dim, sizeof(int));
        weights.resize(dim+1);
        for (int d=0; d<=dim; ++d) is.read((char*)&weights[d],sizeof(FloatType));
    }
    
    FloatType predict(const FloatType* data) {
        FloatType ret = weights[dim];
        for (int d=0; d<dim; ++d) ret += weights[d] * data[d];
        return ret;
    }
};

struct GaussianKernelPredictor : public Predictor {
    int dim;
    FloatType gamma;
    FloatType bias;
    std::vector<FloatType> coefs;
    std::vector<FloatType> support_vectors;
    
    void load(std::istream& is) {
        is.read((char*)&dim, sizeof(int));
        // dlib::deserialize(decfun, is);
        // Too bad the deserialize is buggy !
        is.read((char*)&gamma, sizeof(FloatType));
        is.read((char*)&bias, sizeof(FloatType));
        int nvec;
        is.read((char*)&nvec, sizeof(int));
        coefs.resize(nvec);
        support_vectors.resize(nvec*dim);
        for (int i=0; i<nvec; ++i) {
            is.read((char*)&coefs[i], sizeof(FloatType));
            for (int j=0; j<dim; ++j) {
                is.read((char*)&support_vectors[i*dim+j], sizeof(FloatType));
            }
        }        
    }
    
    FloatType predict(const FloatType* data) {
/*        dlib::matrix<FloatType> x(dim,1);
        for (int d=0; d<dim; ++d) x(d) = data[d];
        return decfun(x);
*/
        FloatType weighted_sum_kern_space = 0;
        for (int i=0; i<coefs.size(); ++i) {
            FloatType d2 = 0;
            for (int j=0; j<dim; ++j) {
                FloatType delta = support_vectors[i*dim+j] - data[j];
                d2 += delta*delta;
            }
            weighted_sum_kern_space += coefs[i] * exp(-gamma*d2);
        }
        return weighted_sum_kern_space - bias;
    }
};

boost::shared_ptr<Predictor> getPredictorFromClassifierID(int id) {
    switch(id) {
        // both least-squares and linear SVM return an hyperplane
        case 0:
        case 1: return boost::make_shared<LinearPredictor>();
        case 2: return boost::make_shared<GaussianKernelPredictor>();
    }
    std::cerr << "Invalid classifier ID" << std::endl;
    exit(1);
    return boost::shared_ptr<Predictor>();
}

#endif
