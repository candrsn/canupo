#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <set>
#include <map>
#include <functional>

#include "points.hpp"
#include "svd.hpp"

#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
canupo scales... - data.xyz data_core.xyz data_core.msc\n\
  inputs: scales         # list of scales at which to perform the analysis\n\
                         # The syntax minscale:increment:maxscale is also accepted\n\
                         # Use - to indicate the end of the list of scales\n\
  input: data.xyz        # whole raw point cloud to process\n\
  input: data_core.xyz   # points at which to do the computation. It is not necessary that these\n\
                         # points match entries in data.xyz: This means data_core.xyz need not be\n\
                         # (but can be) a subsampling of data.xyz, a regular grid is OK.\n\
                         # You can also take exactly the same file, or put more core points than\n\
                         # data points, the core points need only lie in the same region as data.\n\
                         # Tip: use core points at least at max_scale distance from the scene\n\
                         # boundaries in order to avoid spurious multi-scale relations\n\
  outputs: data_core.msc # corresponding multiscale parameters at each core point\n\
"<<endl;
    return 0;
}


int main(int argc, char** argv) {

    if (argc<3) return help();

    int separator = 0;
    for (int i=1; i<argc; ++i) if (!strcmp("-",argv[i])) {
        separator = i;
        break;
    }
    if (!separator) return help();

    // get all unique scales from large to small, for later processing of neighborhoods
    typedef set<FloatType, greater<FloatType> > ScaleSet;
    ScaleSet scales;
    for (int i=1; i<separator; ++i) {
        // perhaps it has the minscale:increment:maxscale syntax
        char* col1 = strchr(argv[i],':');
        char* col2 = strrchr(argv[i],':');
        if (col1==0 || col2==0 || col1==col2) {
            FloatType scale = atof(argv[i]);
            if (scale<=0) return help("Invalid scale");
            scales.insert(scale);
        } else {
            *col1++=0;
            FloatType minscale = atof(argv[i]);
            *col2++=0;
            FloatType increment = atof(col1);
            FloatType maxscale = atof(col2);
            if (minscale<=0 || maxscale<=0) return help("Invalid scale range");
            bool validRange = false;
            if ((minscale - maxscale) * increment > 0) return help("Invalid range specification");
            if (minscale<=maxscale) for (FloatType scale = minscale; scale < maxscale*(1-1e-6); scale += increment) {
                validRange = true;
                scales.insert(scale);
            } else for (FloatType scale = minscale; scale > maxscale*(1+1e-6); scale += increment) {
                validRange = true;
                scales.insert(scale);
            }
            // compensate roundoff errors for loop bounds
            scales.insert(minscale); scales.insert(maxscale);
            if (!validRange) return help("Invalid range specification");
        }
    }
    
    if (scales.empty()) return help();

    cout << "Selected scales:";
    for (ScaleSet::iterator it = scales.begin(); it!=scales.end(); ++it) {
        cout << " " << *it;
    }
    cout << endl;

    // whole data file, core points, msc file
    if (separator+3>=argc) return help();
    
    string datafilename = argv[separator+1];
    string corepointsfilename = argv[separator+2];
    string mscfilename = argv[separator+3];

    cout << "Loading data files" << endl;
    
    cloud.load_txt(datafilename);
    
    vector<Point> corepoints;
    ifstream corepointsfile(corepointsfilename.c_str());
    string line;
    while (corepointsfile && !corepointsfile.eof()) {
        getline(corepointsfile, line);
        if (line.empty()) continue;
        stringstream linereader(line);
        Point point;
        FloatType value;
        int i = 0;
        while (linereader >> value) {
            point[i] = value;
            if (++i==3) break;
        }
        if (i<3) {
            cerr << "Invalid data file: " << corepointsfilename << endl;
            return 1;
        }
        corepoints.push_back(point);
    }

    ofstream mscfile(mscfilename.c_str(), ofstream::binary);
    
    int npts = corepoints.size();
    mscfile.write((char*)&npts, sizeof(npts));
    int nscales = scales.size();
    mscfile.write((char*)&nscales, sizeof(nscales));
    for (ScaleSet::iterator scaleit = scales.begin(); scaleit != scales.end(); ++scaleit) {
        FloatType scale = *scaleit;
        mscfile.write((char*)&scale, sizeof(scale));
    }
    // file ready to write data for all points one by one

    // process each file separately, purely local features
    cout << "Processing \"" << datafilename << "\" using core points from \"" << corepointsfilename << "\"" << endl;
    cout << "Percent complete: 0" << flush;
    
    // for each core point
    int nextpercentcomplete = 5;
#pragma omp parallel for schedule(static)
    for (int ptidx = 0; ptidx < corepoints.size(); ++ptidx) {
#ifdef _OPENMP
if (omp_get_thread_num()==0) {
        int percentcomplete = ((ptidx+1) * 100 * omp_get_num_threads()) / corepoints.size();
#else
        int percentcomplete = ((ptidx+1) * 100) / corepoints.size();
#endif
        if (percentcomplete>=nextpercentcomplete) {
            if (percentcomplete>=nextpercentcomplete) {
                nextpercentcomplete+=5;
                if (percentcomplete % 10 == 0) cout << percentcomplete << flush;
                else if (percentcomplete % 5 == 0) cout << "." << flush;
            }
        }
#ifdef _OPENMP
}
#endif

        vector<DistPoint> neighbors;
        vector<Point> neighsums; // avoid recomputing cumulated sums at each scale
        
        vector<FloatType> abdata(nscales*2);
        int abdataidx = 0;
        
        // Scales shall be sorted from max to lowest 
        for (ScaleSet::iterator scaleit = scales.begin(); scaleit != scales.end(); ++scaleit) {
            // Neighborhood search only on max radius
            if (scaleit == scales.begin()) {
                // we have all neighbors, unsorted, but with distances computed already
                cloud.findNeighbors(back_inserter(neighbors), corepoints[ptidx], *scaleit);

                // Sort the neighbors from closest to farthest, so we can process all lower scales easily
                sort(neighbors.begin(), neighbors.end());
                
                // pre-compute cumulated sums. The total is needed anyway at the larger scale
                // so we might as well share the intermediates to lower levels
                neighsums.resize(neighbors.size());
                neighsums[0] = *neighbors[0].pt;
                for (int i=1; i<neighbors.size(); ++i) neighsums[i] = neighsums[i-1] + *neighbors[i].pt;
            }
            // lower scale : restrict previously found neighbors to the new distance
            else {
                FloatType radiussq = *scaleit * *scaleit;
                // dicho search might be faster than sequencially from the vector end if there are many points
                int dichofirst = 0;
                int dicholast = neighbors.size();
                int dichomed;
                while (true) {
                    dichomed = (dichofirst + dicholast) / 2;
                    if (dichomed==dichofirst) break;
                    if (radiussq==neighbors[dichomed].distsq) break;
                    if (radiussq<neighbors[dichomed].distsq) { dicholast = dichomed; continue;}
                    dichofirst = dichomed;
                }
                // dichomed is now the last index with distance below or equal to requested radius
                neighbors.resize(dichomed+1);
                neighsums.resize(dichomed+1);
            }
            
            // In any case we now have a vector of neighbors at the current scale
            FloatType svalues[3];
            if (neighbors.size()<3) {
                //cout << "Warning: not enough points for 3D PCA, try increasing radius size" << endl;
                // barycentric coords: P = a.S1 + b.S2 + c.S3
                // we want a=b=c=1/3
                // P = 1/3 (S1+S2+S3)
                // P = 1/3 ( 1 + 1/2 + 1/3 , 0 + 1/2 + 1/3 )
                // P = 1/3 ( 11/6 , 5/6 ) = (11/18, 5/18)
                svalues[0] = FloatType(11)/FloatType(18);
                svalues[1] = FloatType(5)/FloatType(18);
                svalues[2] = FloatType(2)/FloatType(18); // complement to 1
            } else {
                // use the pre-computed sums to get the average point
                Point avg = neighsums.back() / neighsums.size();
                // compute PCA on the neighbors at this radius
                // a copy is needed as LAPACK destroys the matrix, and the center changes anyway
                // => cannot keep the points from one scale to the lower, need to rebuild the matrix
                vector<FloatType> A(neighbors.size() * 3);
                for (int i=0; i<neighbors.size(); ++i) {
                    // A is column-major
                    A[i] = neighbors[i].pt->x - avg.x;
                    A[i+neighbors.size()] = neighbors[i].pt->y - avg.y;
                    A[i+neighbors.size()+neighbors.size()] = neighbors[i].pt->z - avg.z;
                }
                // SVD decomposition handled by LAPACK
                svd(neighbors.size(), 3, &A[0], &svalues[0]);
                // convert to percent variance explained by each dim
                FloatType totalvar = 0;
                for (int i=0; i<3; ++i) {
                    // singular values are squared roots of eigenvalues
                    svalues[i] = svalues[i] * svalues[i]; // / (neighbors.size() - 1);
                    totalvar += svalues[i];
                }
                for (int i=0; i<3; ++i) svalues[i] /= totalvar;
            }

            // Use barycentric coordinates : a for 1D, b for 2D and c for 3D
            // Formula on wikipedia page for barycentric coordinates
            // using directly the triangle in %variance space, they simplify a lot
            FloatType a = svalues[0] - svalues[1];
            FloatType b = 2 * svalues[0] + 4 * svalues[1] - 2;
            //FloatType c = 1 - a - b; // they sum to 1
            // negative values shall not happen, but there may be rounding errors and -1e25 is still <0
            if (a<0) a=0; if (b<0) b=0; //if (c<0) c=0;
            // similarly constrain the values to 0..1
            if (a>1) a=1; if (b>1) b=1; //if (c>1) c=1;
            
            abdata[abdataidx++] = a;
            abdata[abdataidx++] = b;
        }
        // need to write full blocks sequencially for each point
#pragma omp critical
        {
            mscfile.write((char*)&corepoints[ptidx].x,sizeof(FloatType));
            mscfile.write((char*)&corepoints[ptidx].y,sizeof(FloatType));
            mscfile.write((char*)&corepoints[ptidx].z,sizeof(FloatType));
            for (int i=0; i<abdata.size(); ++i) mscfile.write((char*)&abdata[i], sizeof(FloatType));
        }
    }
    cout << endl;
    
    mscfile.close();
    

    return 0;
}
