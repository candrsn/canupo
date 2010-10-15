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
canupo scales... - data1.xyz data2.xyz...\n\
  inputs: scales       # list of scales at which to perform the analysis\n\
                       # The syntax minscale:increment:maxscale is also accepted\n\
                       # Use - to indicate the end of the list of scales\n\
  inputs: dataX.xyz    # raw point clouds to process\n\
  outputs: dataX.msc   # corresponding multiscale parameters for each cloud\n\
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
            if (minscale<=maxscale) for (FloatType scale = minscale; scale <= maxscale*(1-1e-6); scale += increment) {
                validRange = true;
                scales.insert(scale);
            } else for (FloatType scale = minscale; scale >= maxscale*(1+1e-6); scale += increment) {
                validRange = true;
                scales.insert(scale);
            }
            // compensate roundoff errors for loop bounds
            scales.insert(minscale); scales.insert(maxscale);
            if (!validRange) return help("Invalid range specification");
        }
    }
    cout << "Selected scales:";
    for (ScaleSet::iterator it = scales.begin(); it!=scales.end(); ++it) {
        cout << " " << *it;
    }
    cout << endl;

    vector<string> datafiles;
    for (int i=separator+1; i<argc; ++i) datafiles.push_back(argv[i]);
    if (scales.empty() || datafiles.empty()) return help();

    // process each file separately, purely local features
    for(int dataidx=0; dataidx<datafiles.size(); ++dataidx) {

        cout << "Processing file: " << datafiles[dataidx] << endl;
        cout << "Percent complete: 0" << flush;
        
        cloud.load_txt(datafiles[dataidx]);

        datafiles[dataidx].replace(datafiles[dataidx].size()-3,3,"msc");
        ofstream mscfile(datafiles[dataidx].c_str(), ofstream::binary);
        
        int npts = cloud.data.size();
        mscfile.write((char*)&npts, sizeof(npts));
        int nscales = scales.size();
        mscfile.write((char*)&nscales, sizeof(nscales));
        for (ScaleSet::iterator scaleit = scales.begin(); scaleit != scales.end(); ++scaleit) {
            FloatType scale = *scaleit;
            mscfile.write((char*)&scale, sizeof(scale));
        }
        // file ready to write data for all points one by one
        
        // for each point in the cloud
        // TODO: subsampling and/or fixed points for the computation
        int nextpercentcomplete = 5;
        // parallel for needs to act on far away points so the risk of false sharing diminishes
        // as the points are nearby in the cloud
        // we also must keep the order of the points in the msc file => need to store the results
        // => use the points user data pointer
#pragma omp parallel for
        for (int ptidx = 0; ptidx < cloud.data.size(); ++ptidx) {
#ifndef _OPENMP
            int percentcomplete = ((ptidx+1) * 100) / cloud.data.size();
            if (percentcomplete>=nextpercentcomplete) {
                if (percentcomplete>=nextpercentcomplete) {
                    nextpercentcomplete+=5;
                    if (percentcomplete % 10 == 0) cout << percentcomplete << flush;
                    else if (percentcomplete % 5 == 0) cout << "." << flush;
                }
            }
#endif
#ifdef _OPENMP
            // store ab points for later if openmp is used
            FloatType* abarray = new FloatType[nscales*2];
            int abidx = 0;
            cloud.data[ptidx].user = abarray;
#endif

            vector<Point*> neighbors;
            vector<FloatType> sqdistances;
            vector<Point> neighsums; // avoid recomputing cumulated sums at each scale
            
            // Scales shall be sorted from max to lowest 
            for (ScaleSet::iterator scaleit = scales.begin(); scaleit != scales.end(); ++scaleit) {
                // Neighborhood search only on max radius
                if (scaleit == scales.begin()) {
                    cloud.findNeighbors(back_inserter(neighbors), cloud.data[ptidx], *scaleit);

                    // Sort the neighbors from closest to farthest, so we can process all lower scales easily
                    multimap<FloatType, Point*> sortingMap;
                    for (int i=0; i<neighbors.size(); ++i) sortingMap.insert(make_pair(dist2(cloud.data[ptidx], *neighbors[i]),neighbors[i]));
                    
                    // put data back in faster containers allowing dicho searches at lower scales
                    sqdistances.resize(neighbors.size());
                    int nidx=0;
                    for (multimap<FloatType, Point*>::iterator it = sortingMap.begin(); it != sortingMap.end(); ++it) {
                        sqdistances[nidx] = it->first;
                        neighbors[nidx] = it->second;
                        ++nidx;
                    }
                    // pre-compute cumulated sums. The total is needed anyway at the larger scale
                    // so we might as well share the intermediates to lower levels
                    neighsums.resize(neighbors.size());
                    neighsums[0] = *neighbors[0];
                    for (int i=1; i<neighbors.size(); ++i) neighsums[i] = neighsums[i-1] + *neighbors[i];
                }
                // lower scale : restrict previously found neighbors to the new distance
                else {
                    FloatType radiussq = *scaleit * *scaleit;
                    // dicho search might be faster than sequencially from the vector end if there are many points
                    int dichofirst = 0;
                    int dicholast = sqdistances.size();
                    int dichomed;
                    while (true) {
                        dichomed = (dichofirst + dicholast) / 2;
                        if (dichomed==dichofirst) break;
                        if (radiussq==sqdistances[dichomed]) break;
                        if (radiussq<sqdistances[dichomed]) { dicholast = dichomed; continue;}
                        dichofirst = dichomed;
                    }
                    // dichomed is now the last index with distance below or equal to requested radius
                    neighbors.resize(dichomed+1);
                    sqdistances.resize(dichomed+1);
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
                    for (int i=0; i<neighbors.size(); ++i) for (int j=0; j<3; ++j) {
                        // A is column-major
                        A[j * neighbors.size() + i] = (*neighbors[i])[j] - avg[j];
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

#ifdef _OPENMP
                abarray[abidx++] = a;
                abarray[abidx++] = b;
#else
                mscfile.write((char*)&a, sizeof(a));
                mscfile.write((char*)&b, sizeof(b));
#endif
            }
        }
#ifdef _OPENMP
        for (int ptidx = 0; ptidx < cloud.data.size(); ++ptidx) {
            FloatType* abarray = (FloatType*)cloud.data[ptidx].user;
            for (int i=0; i<nscales; ++i) {
                mscfile.write((char*)&abarray[i*2], sizeof(FloatType));
                mscfile.write((char*)&abarray[i*2+1], sizeof(FloatType));
            }
            delete [] abarray;
        }
#endif
        cout << endl;
        
        mscfile.close();
    }
    

    return 0;
}
