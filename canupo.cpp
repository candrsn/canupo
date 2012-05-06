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
canupo scales... : data.xyz data_core.xyz data_core.msc [flag]\n\
  inputs: scales         # list of scales at which to perform the analysis\n\
                         # A scale correspond to a diameter for neighbor research.\n\
                         # The syntax minscale:increment:maxscale is accepted.\n\
                         # Any other non-numeric values is interpreted as a classifier\n\
                         # parameter file (.prm) from which scales are loaded.\n\
  input: data.xyz        # whole raw point cloud to process\n\
  input: data_core.xyz   # points at which to do the computation. It is not necessary that these\n\
                         # points match entries in data.xyz: This means data_core.xyz need not be\n\
                         # (but can be) a subsampling of data.xyz, a regular grid is OK.\n\
                         # You can also take exactly the same file, or put more core points than\n\
                         # data points, the core points need only lie in the same region as data.\n\
                         # Tip: use core points at least at max_scale distance from the scene\n\
                         # boundaries in order to avoid spurious multi-scale relations\n\
  output: data_core.msc  # corresponding multiscale parameters at each core point\n\
  input: flag            # (optional) if the flag is set to 1 then an additionnal field is added into the output msc file for each core point: the angle (0<=a<=90°) between the vertical and the normal of the best 2D plane fit at that core point, at the largest given scale. 0 thus means a perfectly horizontal plane, 90 means a perfectly vertical one\n\
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
        bool tryprm = false;
        // perhaps it has the minscale:increment:maxscale syntax
        char* col1 = strchr(argv[i],':');
        char* col2 = strrchr(argv[i],':');
        if (col1==0 || col2==0 || col1==col2) {
            FloatType scale = atof(argv[i]);
            if (scale<=0) tryprm = true;
            else scales.insert(scale);
        } else {
            *col1++=0;
            FloatType minscale = atof(argv[i]);
            *col2++=0;
            FloatType increment = atof(col1);
            FloatType maxscale = atof(col2);
            if (minscale<=0 || maxscale<=0) tryprm = true;
            else {
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
        
        if (tryprm) {
            ifstream classifparamsfile(argv[i], ifstream::binary);
            int nscales_prm;
            classifparamsfile.read((char*)&nscales_prm, sizeof(int));
            if (nscales_prm>10000) return help("Invalid scale and/or prm file");
            for (int s=0; s<nscales_prm; ++s) {
                FloatType thescale = -1;
                classifparamsfile.read((char*)&thescale, sizeof(FloatType));
                if (thescale<=0) return help("Invalid scale within prm file");
                scales.insert(thescale);
            }
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
    
    int flag = 0;
    if (argc>separator+4) flag = atoi(argv[separator+4]);

    bool add_vertical_info = bool( (flag & 1) != 0 );
    
    cout << "Loading data files" << endl;
    
    PointCloud<Point> cloud;
    cloud.load_txt(datafilename);
    
    FILE* corepointsfile = fopen(corepointsfilename.c_str(), "r");
    bool use4 = false;
    int linenum = 0;
    vector<Point> corepoints;
    vector<FloatType> additionalInfo;
    char* line = 0; size_t linelen = 0; int num_read = 0;
    while ((num_read = getline(&line, &linelen, corepointsfile)) != -1) {
        ++linenum;
        if (linelen==0 || line[0]=='#') continue;
        Point point;
        FloatType value = 0;
        int i = 0;
        for (char* x = line; *x!=0;) {
            value = fast_atof_next_token(x);
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
    }
    fclose(corepointsfile);
    assert(additionalInfo.empty() || additionalInfo.size() == corepoints.size());

    ofstream mscfile(mscfilename.c_str(), ofstream::binary);
    
    int npts = corepoints.size();
    mscfile.write((char*)&npts, sizeof(npts));
    int nscales = scales.size();
    mscfile.write((char*)&nscales, sizeof(nscales));
    for (ScaleSet::iterator scaleit = scales.begin(); scaleit != scales.end(); ++scaleit) {
        FloatType scale = *scaleit;
        mscfile.write((char*)&scale, sizeof(scale));
    }
    int ptnparams = 3 + !additionalInfo.empty();
    if (add_vertical_info) ++ptnparams;
    mscfile.write((char*)&ptnparams, sizeof(int));
    // file ready to write data for all points one by one

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
        
        vector<FloatType> abdata(nscales*2);
//        vector<FloatType> avgndist(nscales);
        vector<int> nneigh(nscales);
        int abdataidx = 0;
        
        // ab values implicitly reused from higher scale if there are not enough neighbors
        // TODO: nearest neighbors of ab at higher scales and get average of the neighbors ab at low scale
        FloatType a = 1.0/3.0, b = 1.0/3.0;
        
        // used only when computing the additionnal vertical info
        FloatType vertical_angle = -1;
        vector<FloatType> eigenvectors(9);
        
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
                // SVD decomposition handled by LAPACK
                // compute the vertical info only at the larger scale
                if (add_vertical_info && vertical_angle==-1) {
                    svd(neighbors.size(), 3, &A[0], &svalues[0], false, &eigenvectors[0]);
                    // column-major matrix, eigenvectors as rows
                    Point e1(eigenvectors[0], eigenvectors[3], eigenvectors[6]);
                    Point e2(eigenvectors[1], eigenvectors[4], eigenvectors[7]);
                    // e3 shall be orthogonal to e1 and e2
                    // use the cross-product since the two first components are
                    // better conditionned
                    // then project to (0,0,1), possibly reverting the orientation
                    vertical_angle = fabs(e1.cross(e2).z);
                    // ensure no idiotic out-of-range due to float-point precision...
                    if (vertical_angle<0) vertical_angle = 0;
                    if (vertical_angle>1) vertical_angle = 1;
                    vertical_angle = acos(vertical_angle) * 180 / M_PI;
                }
                else svd(neighbors.size(), 3, &A[0], &svalues[0]);
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
            }

            // negative values shall not happen, but there may be rounding errors and -1e25 is still <0
            if (a<0) a=0; if (b<0) b=0; //if (c<0) c=0;
            // similarly constrain the values to 0..1
            if (a>1) a=1; if (b>1) b=1; //if (c>1) c=1;
            
            abdata[abdataidx++] = a;
            abdata[abdataidx++] = b;
                        
            nneigh[abdataidx/2-1] = neighbors.size();
            
            // compute average distance between nearest neighbors
#if 0
            FloatType avgnd = 0;
            for (int i=0; i<neighbors.size(); ++i) {
                // use min sq dist threshold to eliminate the same point
                int nidx = cloud.findNearest(*neighbors[i].pt, 1e-12);
                avgnd += dist(*neighbors[i].pt, cloud.data[nidx]);
                /*FloatType dmin2 = numeric_limits<FloatType>::max();
                for (int j=0; j<neighbors.size(); ++j) {
                    if (j==i) continue;
                    FloatType d2 = dist2(*neighbors[i].pt, *neighbors[j].pt);
                    if (d2<dmin2) dmin2 = d2;
                }
                avgnd += sqrt(dmin2);*/
            }
            avgnd /= neighbors.size();
            avgndist[abdataidx/2] = avgnd;            
#endif
        }
        // need to write full blocks sequencially for each point
#pragma omp critical
        {
            mscfile.write((char*)&corepoints[ptidx].x,sizeof(FloatType));
            mscfile.write((char*)&corepoints[ptidx].y,sizeof(FloatType));
            mscfile.write((char*)&corepoints[ptidx].z,sizeof(FloatType));
            if (!additionalInfo.empty()) mscfile.write((char*)&additionalInfo[ptidx],sizeof(FloatType));
            if (add_vertical_info) mscfile.write((char*)&vertical_angle,sizeof(FloatType));
            for (int i=0; i<abdata.size(); ++i) mscfile.write((char*)&abdata[i], sizeof(FloatType));
            for (int i=0; i<nscales; ++i) mscfile.write((char*)&nneigh[i], sizeof(int));
//            for (int i=0; i<nscales; ++i) mscfile.write((char*)&avgndist[i], sizeof(FloatType));
        }
    }
    cout << endl;
    
    mscfile.close();
    

    return 0;
}
