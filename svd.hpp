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
/*
This is just a wrapper around LAPACK's dgesvd and sgesvd function
*/
#ifndef CANUPO_SVD_H
#define CANUPO_SVD_H

#include <iostream>
#include <stdlib.h>

/* So, do not use the ublas wrapper.
   The original LAPACK interface is much more flexible and better optimised
   Ex: can reuse the matrix storage for the result, no need to duplicate data in memory
*/
extern "C" {
    void dgesvd_(char const* jobu, char const* jobvt, const int* M, const int* N, double* A, const int* lda, double* S, double* U, const int* ldu, double* Vt, const int* ldvt, double* work, const int* lwork, int *info);
    void sgesvd_(char const* jobu, char const* jobvt, const int* M, const int* N, float* A, const int* lda, float* S, float* U, const int* ldu, float* Vt, const int* ldvt, float* work, const int* lwork, int *info);
}

// wrapper to simplify somewhat the calls.
// Assumes A (M rows, N columns) is organised so that
// - the M rows of A are the N-dimensional observations
// - there are more observations than dimensions : M>=N
// - observations are centered around 0 so as to get the principal components on output
// - A is column-major in memory
// Then A will be overwritten on output:
// - if projectObservations is false, A is filled with garbage
// - otherwise with the projections of the M observations in the N-dimensional space of principle components
//   that is to say, the U * S result of the SVD
// S shall be an array of size N. It will be filled with the square roots of the eigenvalues, in decreasing order
// B shall be either null or a NxN column-major matrix
// - if B is not null, each row will be filled with one eigenvector of ( A.transpose() * A )
//
// In octave / matlab
//   Aori = (Ares .* S) * b'
// but b = inv(b') as it is unitary vects
//   Aori * b = Ares .* S
// thus
//   Ares = (Aori * b) ./ S
// 
// So it is now easy to project an unknown observation in the principal component space
// - divide each column of B by entries in S
// - multiply the new observation (as a row vector) by the B matrix

void svd(int nrows, int ncols, double* A, double* S, bool projectObservations = false, double* B = 0) {
#ifndef LAPACK_IS_THREAD_SAFE
#pragma omp critical
{
#endif
    int info = 0;
    double Dummy; int ld_Dummy = 1;
    int lwork = -1;
    double *tmpwork = new double[ncols]; // in case of failure, elements 1:ncols-1 are referenced
    dgesvd_(projectObservations?"O":"N", B==0?"N":"S", &nrows, &ncols, A, &nrows, S, &Dummy, &ld_Dummy, B==0 ? &Dummy : B, &ncols, tmpwork, &lwork, &info);
    if (info) {
        std::cerr << "Could not retreive the work array size for lapack" << std::endl;
        exit(1);
    }
    lwork = (int)tmpwork[0];
    double* work = new double[lwork];
    dgesvd_(projectObservations?"O":"N", B==0?"N":"S", &nrows, &ncols, A, &nrows, S, &Dummy, &ld_Dummy, B==0 ? &Dummy : B, &ncols, work, &lwork, &info);
    if (info) {
        std::cerr << "Error in dgesvd: " << info << std::endl;
        exit(1);
    }
    delete [] tmpwork;
    delete [] work;
#ifndef LAPACK_IS_THREAD_SAFE
}
#endif
}

// idem using floats
void svd(int nrows, int ncols, float* A, float* S, bool projectObservations = false, float* B = 0) {
#ifndef LAPACK_IS_THREAD_SAFE
#pragma omp critical
{
#endif
    int info = 0;
    float Dummy; int ld_Dummy = 1;
    int lwork = -1;
    float *tmpwork = new float[ncols]; // in case of failure, elements 1:ncols-1 are referenced
    sgesvd_(projectObservations?"O":"N", B==0?"N":"S", &nrows, &ncols, A, &nrows, S, &Dummy, &ld_Dummy, B==0 ? &Dummy : B, &ncols, tmpwork, &lwork, &info);
    if (info) {
        std::cerr << "Could not retreive the work array size for lapack" << std::endl;
        exit(1);
    }
    lwork = (int)tmpwork[0];
    float* work = new float[lwork];
    sgesvd_(projectObservations?"O":"N", B==0?"N":"S", &nrows, &ncols, A, &nrows, S, &Dummy, &ld_Dummy, B==0 ? &Dummy : B, &ncols, work, &lwork, &info);
    if (info) {
        std::cerr << "Error in sgesvd: " << info << std::endl;
        exit(1);
    }
    delete [] tmpwork;
    delete [] work;
#ifndef LAPACK_IS_THREAD_SAFE
}
#endif
}


#endif
