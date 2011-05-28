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
//#include "leastSquares.hpp"

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
normals scales... : refx refy data.xyz data_core.xyz data_core_normals.xyz\n\
  inputs: scales         # list of scales at which to perform the analysis\n\
                         # The syntax minscale:increment:maxscale is also accepted\n\
                         # Use : to indicate the end of the list of scales\n\
                         # A scale correspond to a diameter for neighbor research.\n\
  input: refx refy       # A reference point for orienting horizontal normals toward it\n\
  input: data.xyz        # whole raw point cloud to process\n\
  input: data_core.xyz   # points at which to do the computation. It is not necessary that these\n\
                         # points match entries in data.xyz: This means data_core.xyz need not be\n\
                         # (but can be) a subsampling of data.xyz, a regular grid is OK.\n\
                         # You can also take exactly the same file, or put more core points than\n\
                         # data points, the core points need only lie in the same region as data.\n\
                         # Tip: use core points at least at max_scale distance from the scene\n\
                         # boundaries in order to avoid spurious multi-scale relations\n\
  outputs: data_core_normals.msc\n\
                         # The normals for each core points\n\
                         # The core points are shifted along the normal direction to be at the\n\
                         # center of the cloud in that direction\n\
                         # The shifted core points are given as the first xyz entries in each line\n\
                         # The normal are given as the next xyz entries on the line\n\
                         # The 7th column indicates the scale at which the normal was computed\n\
                         # That scale was selected as the one where the cloud is the most\n\
                         # 2D in the given range (in case of equality choose the largest scale)\n\
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

    Point normalrefpoint(25000, 18000, 0);

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
    if (separator+5>=argc) return help();
    
    normalrefpoint.x = atof(argv[separator+1]);
    normalrefpoint.y = atof(argv[separator+2]);
    
    string datafilename = argv[separator+3];
    string corepointsfilename = argv[separator+4];
    string normfilename = argv[separator+5];

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
    
    // HACK: maintain scene center to orient normals - does not work
    core_center /= corepoints.size();

    ofstream normfile(normfilename.c_str());
    
    cout << "Processing \"" << datafilename << "\" using core points from \"" << corepointsfilename << "\"" << endl;
    cout << "Percent complete: 0" << flush;

    vector<Point> normals(corepoints.size());
    vector<FloatType> bestscales(corepoints.size());
    vector<Point> avgcore(corepoints.size());
    Point avgnormal;
    int ncount = 0;
    
    // for each core point
    int nextpercentcomplete = 5;
    for (int ptidx = 0; ptidx < corepoints.size(); ++ptidx) {
        int percentcomplete = ((ptidx+1) * 100) / corepoints.size();
        if (percentcomplete>=nextpercentcomplete) {
            if (percentcomplete>=nextpercentcomplete) {
                nextpercentcomplete+=5;
                if (percentcomplete % 10 == 0) cout << percentcomplete << flush;
                else if (percentcomplete % 5 == 0) cout << "." << flush;
            }
        }

        vector<DistPoint<Point> > neighbors;
        vector<Point> neighsums; // avoid recomputing cumulated sums at each scale
        
        // maintain the normal at the scale for which the projected cloud on the horizontal plane
        // has a line fit with the least squared error
        Point best_scale_normal;
        Point best_scale_avg;
        FloatType best_scale;
        FloatType best_residue = numeric_limits<FloatType>::max();
        
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
            Point normal;
            FloatType fit = numeric_limits<FloatType>::max();
            Point avg = corepoints[ptidx];
            if (neighbors.size()>=3) {

                FloatType svalues[2];
                
                // Project the points on the horizontal plane and compute
                // the Total Least Square fit (i.e. line with least orthogonal residues)
                // Solution is given by SVD, see: http://arxiv.org/pdf/math.RA/9805076/
                // a copy is needed as LAPACK destroys the matrix, and the center changes anyway
                // => cannot keep the points from one scale to the lower, need to rebuild the matrix
                avg = neighsums.back() / neighsums.size();
                vector<FloatType> A(neighbors.size() * 2);
                for (int i=0; i<neighbors.size(); ++i) {
                    // A is column-major
                    A[i] = neighbors[i].pt->x - avg.x;
                    A[i+neighbors.size()] = neighbors[i].pt->y - avg.y;
                }

                // column-major matrix of the 2 eigenvectors returned as rows...
                vector<FloatType> eigenvectors(4);
                // SVD decomposition handled by LAPACK
                svd(neighbors.size(), 2, &A[0], &svalues[0], false, &eigenvectors[0]);
                
                // total least squares solution is given by the singular vector with
                // minimal singular value
                int mins = 0; if (svalues[1]<svalues[0]) mins = 1;
                // line given by
                // a.(x-xavg) + b.(y-yavg) = 0
                // normal to that line is given by (-b,a,0) or (b,-a,0) in 3D
                normal = Point(eigenvectors[mins], eigenvectors[2+mins], 0);

//cout << normal[0] << ", " << normal[1] << " -- " << svalues[mins] << ", " << svalues[1 - mins] << endl;

                // HACK: orient it so that dot toward scene center is >0
                // if (normal.dot(core_center-corepoints[ptidx]) < 0) normal *= -1;
                // Does not work
                // Orient it so that at +scale in that direction there is max distance to the nearest neighbors
                // normal /= normal.norm();  eigenvectors are already unit length
                Point pt1 = corepoints[ptidx] + normal * *scaleit *2;
                Point pt2 = corepoints[ptidx] - normal * *scaleit *2;
                FloatType avg_dir1 = numeric_limits<FloatType>::max();
                FloatType avg_dir2 = numeric_limits<FloatType>::max();
                for (int i=0; i<neighbors.size(); ++i) {
                    FloatType d1 = (pt1 - neighbors[i].pt).norm();
                    if (d1 < avg_dir1) d1 = avg_dir1;
                    FloatType d2 = (pt2 - neighbors[i].pt).norm();
                    if (d2 < avg_dir2) d2 = avg_dir2;
                }
                if (avg_dir2 > avg_dir1) normal *= -1;
                
/*                // orient it so that density along the line from cloud center point in that direction is highest
                // Does not work either
                FloatType densitypos = 0;
                FloatType densityneg = 0;
                for (int i=0; i<neighbors.size(); ++i) {
                    FloatType dotprod = normal.dot(neighbors[i].pt - avg);
                    if (dotprod<0) densityneg += -dotprod;
                    else densitypos += dotprod;
                }
                if (densityneg>densitypos) normal *= -1;
*/
                
                // compute the fit quality
                fit = 0;
                for (int i=0; i<neighbors.size(); ++i) {
                    FloatType x = neighbors[i].pt->x - avg.x;
                    FloatType y = neighbors[i].pt->y - avg.y;
                    FloatType r = normal[1] * x - normal[0] * y;
                    fit += r*r;
                }
            }
            // new best scale found
            if (fit<best_residue) {
                best_residue = fit;
                best_scale = *scaleit;
                best_scale_normal = normal;
                best_scale_avg = avg;
            }
            
        }
        bestscales[ptidx] = best_scale;
        normals[ptidx] = best_scale_normal;
        avgnormal += best_scale_normal;
        avgcore[ptidx] = best_scale_avg;
        if (best_scale_normal.norm2()>0) ++ncount;
    }
    cout << endl;

    avgnormal /= ncount;

    for (int ptidx = 0; ptidx < corepoints.size(); ++ptidx) {
         Point& normal = normals[ptidx];
         
        // orientation HACK: does not work :(
        // if (normal.dot(avgnormal)<0) normal *= -1;
        
        // Orientation HACK toward given fixed point : OK with correct point !
        if (normal.dot(normalrefpoint)<0) normal *= -1;
        
        // Using average core points do NOT work, too jaggy
        // normfile << avgcore[ptidx].x << " " << avgcore[ptidx].y << " " << avgcore[ptidx].z << " " << normal.x << " " << normal.y << " " << normal.z << " " << bestscales[ptidx] << endl;
        
        // try projecting the average onto the core+normal
        Point shiftedcore = normal.dot(avgcore[ptidx] - corepoints[ptidx]) * normal + corepoints[ptidx];
        normfile << shiftedcore.x << " " << shiftedcore.y << " " << shiftedcore.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << bestscales[ptidx] << endl;
    }
    
    normfile.close();
    

    return 0;
}
