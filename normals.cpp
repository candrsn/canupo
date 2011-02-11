#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <set>
#include <map>
#include <functional>
#include <limits>

#include "points.hpp"
#include "svd.hpp"

#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
normals scales... : data.xyz data_core.xyz data_core_normals.xyz\n\
  inputs: scales         # list of scales at which to perform the analysis\n\
                         # The syntax minscale:increment:maxscale is also accepted\n\
                         # Use : to indicate the end of the list of scales\n\
                         # A scale correspond to a diameter for neighbor research.\n\
  input: data.xyz        # whole raw point cloud to process\n\
  input: data_core.xyz   # points at which to do the computation. It is not necessary that these\n\
                         # points match entries in data.xyz: This means data_core.xyz need not be\n\
                         # (but can be) a subsampling of data.xyz, a regular grid is OK.\n\
                         # You can also take exactly the same file, or put more core points than\n\
                         # data points, the core points need only lie in the same region as data.\n\
                         # Tip: use core points at least at max_scale distance from the scene\n\
                         # boundaries in order to avoid spurious multi-scale relations\n\
  outputs: data_core_normals.msc\n\
                         # The normals for each core points as the first xyz entries in the file\n\
                         # The 4rth column indicates the scale at which the normal was computed\n\
                         # That scale was selected as the one where the cloud is the most\n\
                         # 2D in the given range (in case of equality choose the largest scale)\n\
                         # Normals are oriented toward the vertical: normal.vertical > 0\n\
"<<endl;
    return 0;
}


int main(int argc, char** argv) {
    
    if (argc<3) return help();

    int separator = 0;
    for (int i=1; i<argc; ++i) if (!strcmp(":",argv[i])) {
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
    string normfilename = argv[separator+3];

    cout << "Loading data files" << endl;
    
    PointCloud<Point> cloud;
    cloud.load_txt(datafilename);
    
    ifstream corepointsfile(corepointsfilename.c_str());
    string line;
    bool use4 = false;
    int linenum = 0;
    while (corepointsfile && !corepointsfile.eof()) {
        ++linenum;
        getline(corepointsfile, line);
    }
    corepointsfile.close();
    Point core_center;
    // much better to reserve the memory than to let push_back double it!
    vector<Point> corepoints;
    corepoints.reserve(linenum);
    vector<FloatType> additionalInfo;
    additionalInfo.reserve(linenum);
    corepointsfile.open(corepointsfilename.c_str());
    linenum = 0;
    while (corepointsfile && !corepointsfile.eof()) {
        ++linenum;
        getline(corepointsfile, line);
        if (line.empty() || starts_with(line,"#") || starts_with(line,";") || starts_with(line,"!") || starts_with(line,"//")) continue;
        stringstream linereader(line);
        Point point;
        FloatType value;
        int i = 0;
        while (linereader >> value) {
            if (i<Point::dim) point[i] = value;
            if (++i==4) break;
        }
        if ((use4 && i<4) || (i<3)) {
            cout << "Warning: ignoring line " << linenum << " with only " << i << " value" << (i>1?"s":"") << " in file " << corepointsfilename << endl;
            continue;
        }
        if (i==4) {
            if (use4==false && !corepoints.empty()) {
                cout << "Warning: 4rth value met at line " << linenum << " but it was not present before, discarding all data up to that line." << endl;
                corepoints.clear();
            }
            use4 = true;
            additionalInfo.push_back(value);
        }
        corepoints.push_back(point);
        // HACK: maintain scene center to orient normals
        core_center += point;        
    }
    assert(additionalInfo.empty() || additionalInfo.size() == corepoints.size());
    
    // HACK: maintain scene center to orient normals
    core_center /= corepoints.size();

    ofstream normfile(normfilename.c_str());
    
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

        vector<DistPoint<Point> > neighbors;
        vector<Point> neighsums; // avoid recomputing cumulated sums at each scale
        
        // maintain the normal at the scale for which the cloud is the most "2D"
        Point best_scale_normal;
        FloatType best_scale;
        FloatType best_bidim_value = -1; // the parameter b
        
        // ab values implicitly reused from higher scale if there are not enough neighbors
        // TODO: nearest neighbors of ab at higher scales and get average of the neighbors ab at low scale
        FloatType a = 1.0/3.0, b = 1.0/3.0;
        
        // Scales shall be sorted from max to lowest 
        for (ScaleSet::iterator scaleit = scales.begin(); scaleit != scales.end(); ++scaleit) {
            // Neighborhood search only on max radius
            if (scaleit == scales.begin()) {
                // we have all neighbors, unsorted, but with distances computed already
                // use scales = diameters, not radius
                cloud.findNeighbors(back_inserter(neighbors), corepoints[ptidx], (*scaleit) * 0.5);

                // Sort the neighbors from closest to farthest, so we can process all lower scales easily
                sort(neighbors.begin(), neighbors.end());
                
                // pre-compute cumulated sums. The total is needed anyway at the larger scale
                // so we might as well share the intermediates to lower levels
                neighsums.resize(neighbors.size());
                if (!neighbors.empty()) neighsums[0] = *neighbors[0].pt;
                for (int i=1; i<neighbors.size(); ++i) neighsums[i] = neighsums[i-1] + *neighbors[i].pt;
            }
            // lower scale : restrict previously found neighbors to the new distance
            else {
                FloatType radiussq = *scaleit * *scaleit * 0.25;
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
            
            Point normal;
            // In any case we now have a vector of neighbors at the current scale
            if (neighbors.size()>=3) {
                FloatType svalues[3];
                // use the pre-computed sums to get the average point
                Point avg = neighsums.back() / neighsums.size();
                // compute PCA on the neighbors at this scale
                // a copy is needed as LAPACK destroys the matrix, and the center changes anyway
                // => cannot keep the points from one scale to the lower, need to rebuild the matrix
                vector<FloatType> A(neighbors.size() * 3);
                for (int i=0; i<neighbors.size(); ++i) {
                    // A is column-major
                    A[i] = neighbors[i].pt->x - avg.x;
                    A[i+neighbors.size()] = neighbors[i].pt->y - avg.y;
                    A[i+neighbors.size()+neighbors.size()] = neighbors[i].pt->z - avg.z;
                }
                // column-major matrix of the 3 eigenvectors
                vector<FloatType> eigenvectors(9);
                // SVD decomposition handled by LAPACK
                svd(neighbors.size(), 3, &A[0], &svalues[0], false, &eigenvectors[0]);
                // convert to percent variance explained by each dim
                FloatType totalvar = 0;
                for (int i=0; i<3; ++i) {
                    // singular values are squared roots of eigenvalues
                    svalues[i] = svalues[i] * svalues[i]; // / (neighbors.size() - 1);
                    totalvar += svalues[i];
                }
                for (int i=0; i<3; ++i) svalues[i] /= totalvar;
                // Use barycentric coordinates : a for 1D, b for 2D and c for 3D
                // Formula on wikipedia page for barycentric coordinates
                // using directly the triangle in %variance space, they simplify a lot
                //FloatType c = 1 - a - b; // they sum to 1
                a = svalues[0] - svalues[1];
                b = 2 * svalues[0] + 4 * svalues[1] - 2;
                
                // normals are computed as cross-product of the 2 first eigenvectors
                // column-major, but vectors are stored in rows...
                // shall be the same as e3, except if e3 is very badly conditionned = perfectly 2D !
                // => we want the normal vector even in the perfect case!
                Point e1(eigenvectors[0],eigenvectors[3],eigenvectors[6]);
                Point e2(eigenvectors[1],eigenvectors[4],eigenvectors[7]);
                Point e3(eigenvectors[2],eigenvectors[5],eigenvectors[8]);
                Point n = e2.cross(e1);
//cout << "a=" << a << ", b=" << b << ", e1 × e2 = " << n << ", e3 = " << e3 << endl;
// Orient on the vertical is a bad idea
//                // orient n toward the vertical so n.(0,0,1)>0...
//                if (n.z<0) n = n * (-1.0);

// HACK: orient toward the scene center
                if (n.dot(core_center-corepoints[ptidx])<0) n = (-1.0) * n;

                normal = n;
            }

            // negative values shall not happen, but there may be rounding errors and -1e25 is still <0
            if (b<0) b=0; if (b>1) b=1;
            // new best scale found
            if (b>best_bidim_value) {
                best_bidim_value = b;
                best_scale = *scaleit;
                best_scale_normal = normal;
            }
        }

        // need to write full blocks sequencially for each point
#pragma omp critical
        {
        // select the most 2D scale
        best_scale_normal /= best_scale_normal.norm();
        normfile << corepoints[ptidx].x << " " << corepoints[ptidx].y << " " << corepoints[ptidx].z << " " << best_scale_normal.x << " " << best_scale_normal.y << " " << best_scale_normal.z << " " << best_scale << endl;
        }
    }
    cout << endl;
    
    
    
    normfile.close();
    

    return 0;
}
