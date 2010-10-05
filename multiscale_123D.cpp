#include <iostream>
#include <fstream>

#include "points.hpp"
#include "svd.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost;

CNearTree<Point, FloatType, &point_dist> data;

int main(int argc, char** argv) {

    if (argc<4) {
        cout << "Argument required: bintree_data_file binary_feature_output_file radius1 [radius2...]" << endl;
        return 0;
    }

    vector<FloatType> radiusvec;
    for (int argi = 3; argi<argc; ++argi) {
        FloatType radius = atof(argv[3]);
        radiusvec.push_back(radius);
    }
    int nrad = radiusvec.size();

    ifstream bintreefile(argv[1], ifstream::binary);
    data.load(bintreefile);
    bintreefile.close();
    
    ofstream featuresfile(argv[2], ofstream::binary);
    
    int npts = 0;
    Sequence seq = data.sequence();
    while (seq.hasNext()) {++npts; seq.next();}

    // some informative file header
    featuresfile.write(reinterpret_cast<char*>(&npts), sizeof(int));
    int fdim = nrad * 2; // only a and b now, possibly more later
    featuresfile.write(reinterpret_cast<char*>(&nrad), sizeof(int));

    // compute PCA on the neighbors at this radius
    int ptidx = -1;

    seq = data.sequence();
    while (seq.hasNext()) {
        Point* p = seq.next();
        ++ptidx;

        // this is so we can recover the data set even with the tree ordering
        featuresfile.write(reinterpret_cast<char*>(&(*p)[0]), sizeof(FloatType));
        featuresfile.write(reinterpret_cast<char*>(&(*p)[1]), sizeof(FloatType));
        featuresfile.write(reinterpret_cast<char*>(&(*p)[2]), sizeof(FloatType));
        
        for (int radi = 0; radi<nrad; ++radi) {
            FloatType radius = radiusvec[radi];

            vector<FloatType> svalues(3);

            vector<Point> neighbors;
            data.FindInSphere(radius, neighbors, *p);
            if (neighbors.size()<3) {
                //cout << "Warning: not enough points for 3D PCA, try increasing radius size" << endl;
                svalues[0] = svalues[1] = svalues[2] = FloatType(1)/FloatType(3);
            } else {
                Point avg; avg[0] = 0; avg[1] = 0; avg[2] = 0;
                for (int i=0; i<neighbors.size(); ++i) for (int j=0; j<3; ++j) avg[j] += neighbors[i][j];
                for (int j=0; j<3; ++j) avg[j] /= neighbors.size();
                // compute PCA on the neighbors at this radius
                FloatType* A = new FloatType[neighbors.size() * 3];
                for (int j=0; j<3; ++j) for (int i=0; i<(int)neighbors.size(); ++i) {
                    A[j*neighbors.size()+i] = neighbors[i][j] - avg[j];
                }
                // SVD decomposition handled by LAPACK: we only need the singular values
                svd(neighbors.size(), 3, A, &svalues[0]);
                // convert to percent variance explained by each dim
                FloatType totalvar = 0;
                for (int i=0; i<3; ++i) {
                    svalues[i] = svalues[i] * svalues[i]; // / (neighbors.size() - 1);
                    totalvar += svalues[i];
                }
                for (int i=0; i<3; ++i) svalues[i] /= totalvar;
                //cout << neighbors.size() << ": " << svalues[0] << " " << svalues[1] << " " << svalues[2] << endl;
            }

            // transform svalue domain into equilateral triangle
            FloatType a = 0.5 * svalues[0] + 2.5 * svalues[1] - 0.5;
            // 3 * sqrt(3) / 2
            FloatType b = -2.59807621135332 * (svalues[0] + svalues[1] - 1);
            
            featuresfile.write(reinterpret_cast<char*>(&a), sizeof(FloatType));
            featuresfile.write(reinterpret_cast<char*>(&b), sizeof(FloatType));
            //featuresfile << " " << a << " " << b;
            //featuresfile << endl;
        }
        
    }

/* NO ! PCA on 10 million points not efficient with current processing capacities
    // First try linear PCA to sort out the features, if possible
    // center the features
    for (int i=0; i<featavg.size(); ++i) featavg[i] /= npts;
    for (int pt=0; pt<npts; ++pt) for (int i=0; i<featavg.size(); ++i) featmat(i,pt) -= featavg[i];
    // Now the PCA using SVD. we need both the singular values and right vector for the projections
    //ublas::matrix<FloatType, ublas::column_major> U(TODO,);
    vector<FloatType> svalues(nrad*2);
    lapack::gesvd(featmat, svalues);
    
    // projection of the data on the eigenspace
*/
    
    // TODO : another dimension reduction technique
    // possible sampling/discretising in the triangle space
/*    
    FloatType c = nsubdiv * (a + 0.577350269189626 * b); // sqrt(3)/3
    FloatType d = nsubdiv * (1.154700538379252 * b);     // sqrt(3)*2/3
    int cellx = (int)floor(c);
    int celly = (int)floor(d);
    int lower = (c - cellx) > (d - celly);
    if (cellx>=nsubdiv) {cellx=nsubdiv-1; lower = 1;}
    if (cellx<0) {cellx=0; lower = 1;}
    if (celly>=nsubdiv) {celly=nsubdiv-1; lower = 1;} // upper triangle cell = lower one
    if (celly<0) {celly=0; lower = 1;}
    if (celly>cellx) {celly=cellx; lower = 1;}
    int idx = ((cellx * (cellx+1) / 2) + celly) * 2 + lower;
    //++density[((cellx * (cellx+1) / 2) + celly) * 2 + lower];
*/  
    // TODO: distance measure between cells
    
    return 0;
}


