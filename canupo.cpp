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
    }
    datafile.close();

    FloatType radius = 0;

    if (argc>2) radius = atof(argv[2]);

    if (radius==0) {
        cout << "Warning: inferring radius as max nearest neighbour distance in point cloud..." << endl;
        Sequence seq = data.sequence();
        while (seq.hasNext()) {
            Point* p = seq.next();
            Point closest;
            if (data.NearestNeighbor(numeric_limits<FloatType>::max(), closest, *p)) {
                FloatType dist = point_dist(*p, closest);
                if (dist>radius) radius = dist;
            }
        }
        cout << "Using a radius of " << radius << endl;
    }

    ofstream abfile("ab.txt");

    Sequence seq = data.sequence();
    int npts=0;
    while (seq.hasNext()) {
        ++npts;
        Point* p = seq.next();
        vector<Point> neighbors;
        data.FindInSphere(radius, neighbors, *p);
        if (neighbors.size()<3) {
            cout << "Warning: not enough points for 3D PCA, try increasing radius size" << endl;
            continue;
        }
        Point avg; avg[0] = 0; avg[1] = 0; avg[2] = 0;
        for (int i=0; i<neighbors.size(); ++i) for (int j=0; j<3; ++j) avg[j] += neighbors[i][j];
        for (int j=0; j<3; ++j) avg[j] /= neighbors.size();
        // compute PCA on the neighbors at this radius
        ublas::matrix<FloatType, ublas::column_major> neighmat(3,neighbors.size());
        for (int i=0; i<neighbors.size(); ++i) for (int j=0; j<3; ++j) {
            neighmat(j,i) = neighbors[i][j] - avg[j];
        }
        // SVD decomposition handled by LAPACK
        vector<FloatType> svalues(3);
        lapack::gesvd(neighmat, svalues);
        // convert to percent variance explained by each dim
        FloatType totalvar = 0;
        for (int i=0; i<3; ++i) {
            svalues[i] = svalues[i] * svalues[i]; // / (neighbors.size() - 1);
            totalvar += svalues[i];
        }
        for (int i=0; i<3; ++i) svalues[i] /= totalvar;
        cout << neighbors.size() << ": " << svalues[0] << " " << svalues[1] << " " << svalues[2] << endl;

        abfile << svalues[0] << " " << svalues[1] << endl;
    }
    cout << npts << " points processed." << endl;
    abfile.close();
    
    return 0;
}
