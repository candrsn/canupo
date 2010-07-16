#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include <boost/array.hpp>
//#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/lapack/gesvd.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/std_vector.hpp>

#include "TNear.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ublas = boost::numeric::ublas;
namespace lapack = boost::numeric::bindings::lapack;

using namespace std;
using namespace boost;

typedef float FloatType;

typedef array<FloatType, 3> Point;

inline FloatType point_dist(const Point& a, const Point& b) {
    return sqrt(
        (a[0]-b[0])*(a[0]-b[0])
      + (a[1]-b[1])*(a[1]-b[1])
      + (a[2]-b[2])*(a[2]-b[2])
    );
}

CNearTree<Point, FloatType, &point_dist> data;
typedef CNearTree<Point, FloatType, &point_dist>::Sequence Sequence;

int main(int argc, char** argv) {

    if (argc<2) {
        cout << "Argument required: data_file [radius]" << endl;
        return 0;
    }

    ifstream datafile(argv[1]);
    string line;
    int npts = 0;
    while (datafile && !datafile.eof()) {
        getline(datafile, line);
        if (line.empty()) continue;
        stringstream linereader(line);
        Point point;
        FloatType value;
        int i = 0;
        while (linereader >> value) {
            point[i] = value;
            if (++i==3) break;
        }
        if (i!=3) {
            cout << "invalid data file" << endl;
            return 1;
        }
        data.Insert(point);
        ++npts;
    }
    datafile.close();

    // in case of totally uniform distribution, plan for 10 points per cell on average
    // ncells = ndensitycells*ndensitycells and npts = 10 * ncells;
    int ndensitycells = sqrt(npts/10);
    if (ndensitycells<2) ndensitycells = 2; // at least 1 subdivision
    if (ndensitycells>100) ndensitycells = 100; // too much isn't visible, better build better stats with more points per cell
    vector<int> density(ndensitycells*(ndensitycells+1), 0);
    
    FloatType radius = 0;

    if (argc>2) radius = atof(argv[2]);

    if (radius==0) {
        cout << "Warning: inferring radius as max nearest neighbour distance in point cloud..." << endl;
        Sequence seq = data.sequence();
#if defined(_OPENMP)
        int nthreads = omp_get_max_threads();
        vector<Point*> points(nthreads);
        while (seq.hasNext()) {
        for (int t=0; t<nthreads; ++t) {
            if (seq.hasNext()) points[t] = seq.next();
            else points[t] = 0;
        }
#pragma omp parallel
        {
            int tnum = omp_get_thread_num();
            Point* p = points[tnum];
            if (p!=0) {
#else
        while (seq.hasNext()) {
            Point* p = seq.next();
#endif
            Point closest;
            if (data.NearestNeighbor(numeric_limits<FloatType>::max(), closest, *p)) {
                FloatType dist = point_dist(*p, closest);
                if (dist>radius) radius = dist;
            }
        }
#if defined(_OPENMP)
        }}
#endif        
        cout << "Using a radius of " << radius << endl;
    }

    ofstream abfile("ab.txt");
    ofstream annotatedfile("annotated.xyz");

    Sequence seq = data.sequence();
#if defined(_OPENMP)
    int nthreads = omp_get_max_threads();
    vector<Point*> points(nthreads);
    vector<vector<FloatType> > svaluesvec(nthreads);
    for (int t=0; t<nthreads; ++t) svaluesvec[t].resize(3);
    while (seq.hasNext()) {
    for (int t=0; t<nthreads; ++t) {
        if (seq.hasNext()) points[t] = seq.next();
        else points[t] = 0;
    }
#pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        Point* p = points[tnum];
        if (p!=0) {
        vector<FloatType>& svalues = svaluesvec[tnum];
#else
    while (seq.hasNext()) {
        Point* p = seq.next();
        vector<FloatType> svalues(3);
#endif
        vector<Point> neighbors;
        data.FindInSphere(radius, neighbors, *p);
        if (neighbors.size()<3) {
            cout << "Warning: not enough points for 3D PCA, try increasing radius size" << endl;
            svalues[0] = svalues[1] = svalues[2] = FloatType(1)/FloatType(3);
        } else {
            Point avg; avg[0] = 0; avg[1] = 0; avg[2] = 0;
            for (int i=0; i<neighbors.size(); ++i) for (int j=0; j<3; ++j) avg[j] += neighbors[i][j];
            for (int j=0; j<3; ++j) avg[j] /= neighbors.size();
            // compute PCA on the neighbors at this radius
            ublas::matrix<FloatType, ublas::column_major> neighmat(3,neighbors.size());
            for (int i=0; i<neighbors.size(); ++i) for (int j=0; j<3; ++j) {
                neighmat(j,i) = neighbors[i][j] - avg[j];
            }
            // SVD decomposition handled by LAPACK
            lapack::gesvd(neighmat, svalues);
            // convert to percent variance explained by each dim
            FloatType totalvar = 0;
            for (int i=0; i<3; ++i) {
                svalues[i] = svalues[i] * svalues[i]; // / (neighbors.size() - 1);
                totalvar += svalues[i];
            }
            for (int i=0; i<3; ++i) svalues[i] /= totalvar;
            //cout << neighbors.size() << ": " << svalues[0] << " " << svalues[1] << " " << svalues[2] << endl;
        }
#if defined(_OPENMP)
    }}
    for (int t=0; t<nthreads; ++t) if (points[t]!=0) {
        Point* p = points[t];
        vector<FloatType>& svalues = svaluesvec[t];
#endif

        // transform svalue domain into equilateral triangle
        FloatType a = 0.5 * svalues[0] + 2.5 * svalues[1] - 0.5;
        // 3 * sqrt(3) / 2
        FloatType b = -2.59807621135332 * (svalues[0] + svalues[1] - 1);

        //abfile << svalues[0] << " " << svalues[1] << endl;
        abfile << a << " " << b << endl;
        
        // Density plot of (a,b) points: discretize the triangle and count how many points are in each cell
        // Transform (a,b) such that equilateral triangle goes to triangle with right angle (0,0) (1,0) (0,1)
        FloatType c = ndensitycells * (a + 0.577350269189626 * b); // sqrt(3)/3
        FloatType d = ndensitycells * (1.154700538379252 * b);     // sqrt(3)*2/3
        int cellx = (int)floor(c);
        int celly = (int)floor(d);
        int lower = (c - cellx) > (d - celly);
        if (cellx>=ndensitycells) {cellx=ndensitycells-1; lower = 1;}
        if (cellx<0) {cellx=0; lower = 1;}
        if (celly>=ndensitycells) {celly=ndensitycells-1; lower = 1;} // upper triangle cell = lower one
        if (celly<0) {celly=0; lower = 1;}
        if (celly>cellx) {celly=cellx; lower = 1;}
        //int idx = ((cellx * (cellx+1) / 2) + celly) * 2 + lower;
        ++density[((cellx * (cellx+1) / 2) + celly) * 2 + lower];
        
        FloatType R = 255 - int(floor(sqrt( a*a + b*b)*255.9999));
        FloatType G = 255 - int(floor(sqrt( (a-1)*(a-1) + b*b)*255.9999));
        FloatType B = 255 - int(floor(sqrt( (a-0.5)*(a-0.5) + (b-0.866025403784439)*(b-0.866025403784439))*255.9999));
        annotatedfile << (*p)[0] << " " << (*p)[1] << " " << (*p)[2] << " " << R << " " << G << " " << B << endl;
#if defined(_OPENMP)
    }
#endif        
    }
    abfile.close();
    annotatedfile.close();    

    ofstream densityfile("dimdensity.txt");
    for (int x=0; x<ndensitycells; ++x) for (int y=0; y<=x; ++y) {
        // lower cell
        FloatType c = (x + 0.666666666666667)/FloatType(ndensitycells);
        FloatType d = (y + 0.333333333333333)/FloatType(ndensitycells);
        FloatType a = c - 0.5 * d;
        FloatType b = 0.866025403784439 * d;
        densityfile << a << " " << b << " " << density[(x*(x+1)/2+ y)*2] << endl;
        if (y<x) { // upper cell
            c = (x + 0.333333333333333)/FloatType(ndensitycells);
            d = (y + 0.666666666666667)/FloatType(ndensitycells);
            a = c - 0.5 * d;
            b = 0.866025403784439 * d;
            densityfile << a << " " << b << " " << density[(x*(x+1)/2+ y)*2 + 1] << endl;
        }
    }
    densityfile.close();
    
    return 0;
}
