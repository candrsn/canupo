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

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>

#include <boost/format.hpp>

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
cout << "\
normaldiff scales... : p1.xyz p2.xyz cores.xyz extpts.xyz result.xyz [opt_flags [extra_info]]\n\
  inputs: scales         # list of scales at which to perform the analysis\n\
                         # The syntax minscale:increment:maxscale is also accepted\n\
                         # Use : to indicate the end of the list of scales\n\
                         # A scale correspond to a diameter for neighbor research.\n\
  input: p1.xyz          # first whole raw point cloud to process\n\
  input: p2.xyz          # second whole raw point cloud to process, possibly the same as p1\n\
                         # if you only care for the normal computation and core point shifting.\n\
  input: cores.xyz       # points near which to do the computation. It is not necessary that these\n\
                         # points match entries in either cloud. A regular grid is OK for example.\n\
                         # You can also take exactly the same file, or put more core points than\n\
                         # data points, the core points need only lie in the same region as data.\n\
                         # Core points are shifted toward the surface along the local normal\n\
                         # before doing the difference. The goal is to monitor the surface\n\
                         # evolution between p1 and p2.\n\
  input: extpts.xyz      # set of points on the exterior of the objets in the scene, toward wich\n\
                         # to orient normals. The closest point in the set is used to orient the\n\
                         # normal a each core point. The exterior points shall be exterior to\n\
                         # both p1 and p2...\n\
  outputs: result.txt    # A file containing as many result lines as core points.\n\
                         # All these results are influenced by the bootstrapping process except\n\
                         # the scale values that are fixed before the bootstrapping.\n\
                         # Each line contains space separated entries. In order:\n\
                         # - c1.x c1.y c1.z: core point coordinates, shifted to surface of p1\n\
                         # - c2.x c2.y c2.z: core point coordinates, shifted to surface of p2\n\
                         # - n1.x n1.y n1.z: surface normal vector coordinates for p1\n\
                         # - n2.x n2.y n2.z: surface normal vector coordinates for p2\n\
                         # - sn1 sn2: scales at which the normals were computed in p1 and p2\n\
                         # - sc1 sc2: scales at which the core points were computed in p1 and p2\n\
                         # - dev1 dev2: standard deviation of the distance between: the points\n\
                         #              surrounding c1 and c2 at scales sc1 and sc2; and the\n\
                         #              \"surface\" planes defined by the normals at c1 and c2.\n\
                         #              This can be seen as a quality measure of the plane fitting.\n\
                         # - np1 np2: number of points in these surroundings\n\
                         # - diff: average value of the signed distance c2 - c1, estimated by\n\
                         #         the statistical bootstrapping technique. Hence this value is\n\
                         #         not null when p1=p2, in that case it can be interpreted as\n\
                         #         a measure of the error made in shifting c to c1/2.\n\
                         # - diff_dev: deviation of the diff value, estimated by bootstrapping.\n\
  input: opt_flags       # Optional flags. Some may be combined together. Ex: \"hnc\".\n\
                         # Available flags are:\n\
                         #  h: make the resulting normals purely Horizontal (null z component).\n\
                         #  v: make the resulting normals purely Vertical (so, all normals are either +z or -z depending on the reference points).\n\
                         #  n: compute the Normal only at the max given scale (default is the scale at which the cloud is most 2D)\n\
                         #  c: compute the shifted Core point only at the lowest given scale (default is the scale at which the cloud is most 2D)\n\
                         #  e: provide the standard deviation of the Error on the point positions\n\
                         #     in p1 and p2 as extra info. This is a parameter provided by the device\n\
                         #     used for measuring p1 and p2. It is incorporated in the bootstrapping.\n\
                         #  s: Systematic error on the point differences given as a 3D vector\n\
                         #     containing the mean error in each direction\n\
                         #  b: Number of bootstrap iterations (default is 100). 0 disables\n\
                         #     bootstrapping: you won't get diff_dev and diff is the sample mean.\n\
  input: extra_info      # Extra parameters for the \"e\", \"s\" and \"b\" flags, given in the\n\
                         # same order as these flags were specified.\n\
                         # Ex: normaldiff (all other opts) ehbs 1e-2 1000 2e-3 3e-4 1e-3\n\
                         # The flags are \"e\", \"h\", \"b\" and \"s\". \"h\" has no extra parameter.\n\
                         # So, in this order: e=1e-2, b=1000, and s=(2e-3,3e-4,1e-3)\n\
"<<endl;
    if (errmsg) cout << "Error: " << errmsg << endl;
    return 0;
}


int main(int argc, char** argv) {

    if (argc<8) return help();

    int separator = 0;
    for (int i=1; i<argc; ++i) if (!strcmp(":",argv[i])) {
        separator = i;
        break;
    }
    if (separator==0) return help();
    
    if (argc<separator+6) return help();

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

    vector<FloatType> scalesvec(scales.begin(), scales.end());
    int nscales = scalesvec.size();
    
    if (scales.empty()) return help();

    cout << "Selected scales:";
    for (ScaleSet::iterator it = scales.begin(); it!=scales.end(); ++it) {
        cout << " " << *it;
    }
    cout << endl;

    string p1fname = argv[separator+1];
    string p2fname = argv[separator+2];
    string corefname = argv[separator+3];
    string extptsfname = argv[separator+4];
    string resultfname = argv[separator+5];
    
    bool force_horizontal = false;
    bool force_vertical = false;
    bool normal_max_scale = false;
    bool core_min_scale = false;
    double pos_dev = 0;
    int num_bootstrap_iter = 100;
    Point systematic_error(0,0,0);
    
    if (argc>separator+6) {
        int extra_info_idx = separator+6;
        for (char* opt=argv[separator+6]; *opt!=0; ++opt) {
            switch(*opt) {
                case 'h': force_horizontal = true; break;
                case 'v': force_vertical = true; break;
                case 'n': normal_max_scale = true; break;
                case 'c': core_min_scale = true; break;
                case 'e': if (++extra_info_idx<argc) {
                    pos_dev = atof(argv[extra_info_idx]); break;
                } else return help("Missing value for the e flag");
                case 'b': if (++extra_info_idx<argc) {
                    num_bootstrap_iter = atoi(argv[extra_info_idx]); break;
                } else return help("Missing value for the b flag");
                case 's': if (extra_info_idx+3<argc) {
                    systematic_error.x = atof(argv[++extra_info_idx]);
                    systematic_error.y = atof(argv[++extra_info_idx]);
                    systematic_error.z = atof(argv[++extra_info_idx]);
                    break;
                } else return help("Missing value for the s flag"); break;
                default: return help((string("Unrecognised optional flag: ")+opt).c_str());
            }
        }
    }
    
    if (force_horizontal && force_vertical) return help("Cannot force normals to be both horizontal and vertical!");
    
    if (num_bootstrap_iter<=0) {
        if (pos_dev>0) cout << "Warning: e flag is ignored when there is no bootstrap" << endl;
        num_bootstrap_iter = 1;
        pos_dev = 0;
    }
    
    boost::mt19937 rng;
    boost::normal_distribution<FloatType> poserr_dist(0, pos_dev);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<FloatType> > poserr_rand(rng, poserr_dist);

    cout << "Loading data files" << endl;
    
    PointCloud<Point> p1;
    p1.load_txt(p1fname);
    PointCloud<Point> p2;
    p2.load_txt(p2fname);
    
    ifstream corepointsfile(corefname.c_str());
    string line;
    int linenum = 0;
    vector<Point> corepoints;
    while (corepointsfile && !corepointsfile.eof()) {
        ++linenum;
        getline(corepointsfile, line);
        if (line.empty() || starts_with(line,"#") || starts_with(line,";") || starts_with(line,"!") || starts_with(line,"//")) continue;
        stringstream linereader(line);
        Point point;
        FloatType value;
        int i = 0;
        while (linereader >> value) {
            if (i<Point::dim) point[i++] = value;
            else break;
        }
        if (i<3) return help(str(boost::format("Error in the core points file line %d") % (linenum+1)).c_str());
        corepoints.push_back(point);
    }
    corepointsfile.close();
    
    
    ifstream refpointsfile(corefname.c_str());
    linenum = 0;
    vector<Point> refpoints;
    while (refpointsfile && !refpointsfile.eof()) {
        ++linenum;
        getline(refpointsfile, line);
        if (line.empty() || starts_with(line,"#") || starts_with(line,";") || starts_with(line,"!") || starts_with(line,"//")) continue;
        stringstream linereader(line);
        Point point;
        FloatType value;
        int i = 0;
        while (linereader >> value) {
            if (i<Point::dim) point[i++] = value;
            else break;
        }
        if (i<3) return help(str(boost::format("Error in the reference points file line %d") % (linenum+1)).c_str());
        refpoints.push_back(point);
    }
    refpointsfile.close();
    
    if (refpoints.empty()) return help("Please provide at least one reference point");

    ofstream resultfile(resultfname.c_str());
    
    cout << "Processing, percent complete: 0" << flush;
    
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

        // first extract the neighbors on which to do the normal computations
        // these are fixed for the whole bootstrapping. Technically we should apply
        // the pos_dev and resampling before looking for neighbors in order to
        // ensure proper computation at the exact selected scale. In practice
        // the difference does not matter (we're at scale +- pos_dev instead of scale)
        // and the computations should be much faster!
        
        vector<DistPoint<Point> > neighbors_1, neighbors_2;
        vector<int> neigh_num_1(nscales,0), neigh_num_2(nscales,0);
        vector<Point> neighsums_1, neighsums_2;
        
        // Scales shall be sorted from max to lowest 
        int scaleidx = 0;
        for (ScaleSet::iterator scaleit = scales.begin(); scaleit != scales.end(); ++scaleit, ++scaleidx) {
            // Neighborhood search only on max radius
            if (scaleidx==0) {
                // we have all neighbors, unsorted, but with distances computed already
                // use scales = diameters, not radius
                p1.findNeighbors(back_inserter(neighbors_1), corepoints[ptidx], (*scaleit) * 0.5);
                p2.findNeighbors(back_inserter(neighbors_2), corepoints[ptidx], (*scaleit) * 0.5);
                // Sort the neighbors from closest to farthest, so we can process all lower scales easily
                sort(neighbors_1.begin(), neighbors_1.end());
                sort(neighbors_2.begin(), neighbors_2.end());
                neigh_num_1[scaleidx] = neighbors_1.size();
                neigh_num_2[scaleidx] = neighbors_2.size();
                // pre-compute cumulated sums. The total is needed anyway at the larger scale
                // so we might as well share the intermediates to lower levels
                neighsums_1.resize(neighbors_1.size());
                if (!neighbors_1.empty()) neighsums_1[0] = *neighbors_1[0].pt;
                for (int i=1; i<neighbors_1.size(); ++i) neighsums_1[i] = neighsums_1[i-1] + *neighbors_1[i].pt;
                neighsums_2.resize(neighbors_2.size());
                if (!neighbors_2.empty()) neighsums_2[0] = *neighbors_2[0].pt;
                for (int i=1; i<neighbors_2.size(); ++i) neighsums_2[i] = neighsums_2[i-1] + *neighbors_2[i].pt;
            }
            // lower scale : restrict previously found neighbors to the new distance
            else {
                FloatType radiussq = *scaleit * *scaleit * 0.25;
                // dicho search might be faster than sequencially from the vector end if there are many points
                int dichofirst = 0;
                int dicholast = neighbors_1.size();
                int dichomed;
                while (true) {
                    dichomed = (dichofirst + dicholast) / 2;
                    if (dichomed==dichofirst) break;
                    if (radiussq==neighbors_1[dichomed].distsq) break;
                    if (radiussq<neighbors_1[dichomed].distsq) { dicholast = dichomed; continue;}
                    dichofirst = dichomed;
                }
                // dichomed is now the last index with distance below or equal to requested radius
                neigh_num_1[scaleidx] = dichomed+1;
                // same for cloud 2
                dichofirst = 0;
                dicholast = neighbors_2.size();
                while (true) {
                    dichomed = (dichofirst + dicholast) / 2;
                    if (dichomed==dichofirst) break;
                    if (radiussq==neighbors_2[dichomed].distsq) break;
                    if (radiussq<neighbors_2[dichomed].distsq) { dicholast = dichomed; continue;}
                    dichofirst = dichomed;
                }
                neigh_num_2[scaleidx] = dichomed+1;
            }
        }


        // The most planar scale is only computed once if needed
        // bootstrapping is then done on that scale for the normal computations
        int core_scale_idx_1 = nscales-1, core_scale_idx_2 = nscales-1;
        int normal_scale_idx_1 = 0, normal_scale_idx_2 = 0;
        // if either flag is not set, we must compute the most 2D scale now
        if (!normal_max_scale || !core_min_scale) {
            
            FloatType svalues[3];
            // avoid code dup below
            // but some dup in bootstrapping as I'm lazy to get rid of it
            int* core_scale_idx_ref[2] = {&core_scale_idx_1, &core_scale_idx_2};
            int* normal_scale_idx_ref[2] = {&normal_scale_idx_1, &normal_scale_idx_2};
            vector<DistPoint<Point> >* neighbors_ref[2] = {&neighbors_1, &neighbors_2};
            vector<int>* neigh_num_ref[2] = {&neigh_num_1, &neigh_num_2};
            vector<Point>* neighsums_ref[2] = {&neighsums_1, &neighsums_2};
            // loop on both pt sets
            for (int ref12_idx = 0; ref12_idx < 2; ++ref12_idx) {
                vector<DistPoint<Point> >& neighbors = *neighbors_ref[ref12_idx];
                FloatType maxbarycoord = -numeric_limits<FloatType>::max();
                for (int sidx=0; sidx<nscales; ++sidx) {
                    int npts = (*neigh_num_ref[ref12_idx])[sidx];
                    if (npts>=3) {
                        // use the pre-computed sums to get the average point
                        Point avg = (*neighsums_ref[ref12_idx])[npts-1] / npts;
                        // compute PCA on the neighbors at this scale
                        // a copy is needed as LAPACK destroys the matrix, and the center changes anyway
                        // => cannot keep the points from one scale to the lower, need to rebuild the matrix
                        vector<FloatType> A(npts * 3);
                        for (int i=0; i<npts; ++i) {
                            // A is column-major
                            A[i] = neighbors[i].pt->x - avg.x;
                            A[i+npts] = neighbors[i].pt->y - avg.y;
                            A[i+npts*2] = neighbors[i].pt->z - avg.z;
                        }
                        svd(npts, 3, &A[0], &svalues[0]);
                        // The most 2D scale. For the criterion for how "2D" a scale is, see canupo
                        // Ideally first and second eigenvalue are equal
                        // convert to percent variance explained by each dim
                        FloatType totalvar = 0;
                        for (int i=0; i<3; ++i) {
                            // singular values are squared roots of eigenvalues
                            svalues[i] = svalues[i] * svalues[i]; // / (neighbors.size() - 1);
                            totalvar += svalues[i];
                        }
                        for (int i=0; i<3; ++i) svalues[i] /= totalvar;
                        // ideally, 2D means first and second entries are both 1/2 and third is 0
                        // convert to barycentric coordinates and take the coefficient of the 2D
                        // corner as a quality measure.
                        // Use barycentric coordinates : a for 1D, b for 2D and c for 3D
                        // Formula on wikipedia page for barycentric coordinates
                        // using directly the triangle in %variance space, they simplify a lot
                        //FloatType c = 1 - a - b; // they sum to 1
                        // a = svalues[0] - svalues[1];
                        FloatType b = 2 * svalues[0] + 4 * svalues[1] - 2;
                        if (b > maxbarycoord) {
                            maxbarycoord = b;
                            if (!core_min_scale) *core_scale_idx_ref[ref12_idx] = sidx;
                            if (!normal_max_scale) *normal_scale_idx_ref[ref12_idx] = sidx;
                        }
                    }                    
                }
            } //ref12 loop
        }

        // closest ref point is also shared for all bootstrap iterations for efficiency
        // non-empty set => valid index
        int nearestidx = -1;
        FloatType mindist = numeric_limits<FloatType>::max();
        for (int i=0; i<refpoints.size(); ++i) {
            FloatType d = dist2(refpoints[i],corepoints[ptidx]);
            if (d<mindist) {
                mindist = d;
                nearestidx = i;
            }
        }

        Point& refpt = refpoints[nearestidx];
        Point deltaref = refpt - corepoints[ptidx];
        if (force_horizontal) deltaref.z = 0;
        deltaref.normalize();
            
        Point normal_1, normal_2;
        Point core_shift_1, core_shift_2;
        FloatType plane_dev_1 = 0, plane_dev_2 = 0;
        FloatType plane_nump_1 = 0, plane_nump_2 = 0;
        FloatType diff = 0, diff_dev = 0;

        // We have all core point neighbors at all scales in each data set
        // and the correct scales for the computation
        // Now bootstrapping...
        for (int bootstrap_iter = 0; bootstrap_iter < num_bootstrap_iter; ++bootstrap_iter) {
            
            Point core_shift_bs_1, core_shift_bs_2;
            
            FloatType svalues[3];
            FloatType eigenvectors[9];
            
            vector<Point> resampled_neighbors_1, resampled_neighbors_2;
            // avoid code dup below
            int* core_scale_idx_ref[2] = {&core_scale_idx_1, &core_scale_idx_2};
            int* normal_scale_idx_ref[2] = {&normal_scale_idx_1, &normal_scale_idx_2};
            Point* normal_ref[2] = {&normal_1, &normal_2};
            Point* core_shift_ref[2] = {&core_shift_1, &core_shift_2};
            Point* core_shift_bs_ref[2] = {&core_shift_bs_1, &core_shift_bs_2};
            FloatType* plane_dev_ref[2] = {&plane_dev_1, &plane_dev_2};
            FloatType* plane_nump_ref[2] = {&plane_nump_1, &plane_nump_2};
            vector<DistPoint<Point> >* neighbors_ref[2] = {&neighbors_1, &neighbors_2};
            vector<int>* neigh_num_ref[2] = {&neigh_num_1, &neigh_num_2};
            vector<Point>* resampled_neighbors_ref[2] = {&resampled_neighbors_1, &resampled_neighbors_2};
            // loop on both pt sets
            for (int ref12_idx = 0; ref12_idx < 2; ++ref12_idx) {
                vector<DistPoint<Point> >& neighbors = *neighbors_ref[ref12_idx];
                vector<Point>& resampled_neighbors = *resampled_neighbors_ref[ref12_idx];
                int sidx = *normal_scale_idx_ref[ref12_idx];                
                int npts = (*neigh_num_ref[ref12_idx])[sidx]; // ensured >=3 at this point
                resampled_neighbors.resize(npts);
                if (npts==0) continue;
                // resampling with replacement.
                Point avg = 0;
                vector<FloatType> A(npts * 3);
                boost::uniform_int<int> int_dist(0, npts-1);
                boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > randint(rng, int_dist);
                for (int i=0; i<npts; ++i) {
                    //int_dist(rng); // boost 1.47 is so much better :(
                    int selected_idx = randint();
                    resampled_neighbors[i] = *neighbors[selected_idx].pt;
                    // add some gaussian noise with dev specified by the user on each coordinate
                    resampled_neighbors[i].x += poserr_rand(); //poserr_dist(rng);
                    resampled_neighbors[i].y += poserr_rand();
                    resampled_neighbors[i].z += poserr_rand();
                    avg += resampled_neighbors[i];
                    A[i] = resampled_neighbors[i].x;
                    A[i+npts] = resampled_neighbors[i].y;
                    A[i+npts*2] = resampled_neighbors[i].z;
                }
            
                Point normal;
                
                // vertical normals need none of the SVD business
                // and could also skip A, etc...
                if (force_vertical) normal.z = 1;
                else {
                    // now finish the averaging
                    avg /= npts;
                    for (int i=0; i<npts; ++i) {
                        A[i] -= avg.x;
                        A[i+npts] -= avg.y;
                        A[i+npts*2] -= avg.z;
                    }
                    if (force_horizontal) {
                        if (npts<2 && bootstrap_iter==0) {
                            cout << "Warning: Invalid core point / data file / scale combination: less than 2 points at max scale for core point " << (ptidx+1) << " in data set " << ref12_idx+1 << endl;
                        } else {
                            // reshape A to 2x2 column-wise and skip z
                            A[2] = A[3]; A[3] = A[4]; // A[5 and more is ignored].
                            // SVD decomposition handled by LAPACK
                            svd(npts, 2, &A[0], &svalues[0], false, &eigenvectors[0]);
                            // The total least squares solution in the horizontal plane
                            // is given by the singular vector with minimal singular value
                            int mins = 0; if (svalues[1]<svalues[0]) mins = 1;
                            normal = Point(eigenvectors[mins], eigenvectors[2+mins], 0);
                        }
                    } else {
                        if (npts<3 && bootstrap_iter==0) {
                            cout << "Warning: Invalid core point / data file / scale combination: less than 3 points at max scale for core point " << (ptidx+1) << " in data set " << ref12_idx+1 << endl;
                        } else {
                            svd(npts, 3, &A[0], &svalues[0], false, &eigenvectors[0]);
                            // column-major matrix, eigenvectors as rows
                            Point e1(eigenvectors[0], eigenvectors[3], eigenvectors[6]);
                            Point e2(eigenvectors[1], eigenvectors[4], eigenvectors[7]);
                            normal = e1.cross(e2);
                        }
                    }
                }
                
                // normal orientation... simple with external help
                if (normal.dot(deltaref)<0) normal *= -1;
                
                *normal_ref[ref12_idx] += normal;
                
                // projection of the core point along a cylinder in the normal direction
                // The cylinder base diameter is core_scale
                // The cylinder height is normal_scale
                // The cylinder is centered on the original core point
                // Then:
                // - get the density profile along the cylinder center line
                // - look for a peak in the density
                //   alternatively, look for the plane with best fit ?
                //   => does not work for bi-modal distributions...
                // Possible answer:
                // - simple grid search along the line for best histogram density
                // - refining the grid, possibly twice
                // We need to compute dev of dist from pts to plane with scale core_scale
                // - finalise by using the average of the projections at +- half core scale
                //   from the current bin center, and use that average instead of the bin center
                //   => the average is now on the plane, and dev can be given
                // loop on both pt sets
                const int num_bins = 100;
                vector<int> histogram(num_bins, 0);
                vector<vector<FloatType> > incylptdist(num_bins, vector<FloatType>(0));
                FloatType normal_radius = 0.5 * scalesvec[*normal_scale_idx_ref[ref12_idx]];
                FloatType core_radius = 0.5 * scalesvec[*core_scale_idx_ref[ref12_idx]];
                FloatType core_radius_sq = core_radius * core_radius;
                for (int i=0; i<npts; ++i) {
                    Point delta = resampled_neighbors[i] - corepoints[ptidx];
                    FloatType dist_along_axis = delta.dot(normal);
                    FloatType dist_to_axis_sq = (delta - dist_along_axis * normal).norm2();
                    // ignore points outside the cylinder
                    if (dist_to_axis_sq > core_radius_sq) continue;
                    int bin = (int)floor((normal_radius + dist_along_axis) * num_bins / (2.0 * normal_radius));
                    // shall not happen except perhaps for roundoff errors
                    if (bin<0) bin = 0; if (bin>=num_bins) bin = num_bins-1;
                    ++histogram[bin];
                    incylptdist[bin].push_back(dist_along_axis);
                }
                int selected_bin = 0;
                int max_density = 0;
                for (int i=0; i<num_bins; ++i) {
                    // favor bins further along the normal direction
                    // in case of equality (bins nearest the surface, the most "outside")
                    // TODO: allow double-peaked distributions with lower second peak
                    if (histogram[i]>=max_density) {
                        max_density = histogram[i];
                        selected_bin = i;
                    }
                }
                // process average at +- core_radius from the current bin
                int min_bin = selected_bin - core_radius * num_bins / (2.0 * normal_radius);
                int max_bin = selected_bin + core_radius * num_bins / (2.0 * normal_radius);
                if (min_bin<0) min_bin = 0;
                if (max_bin>=num_bins) max_bin = num_bins - 1;
                FloatType avgd = 0;
                FloatType devd = 0;
                int numd = 0;
                for (int i=min_bin; i<=max_bin; ++i) {
                    for (vector<FloatType>::iterator it = incylptdist[i].begin(); it != incylptdist[i].end(); ++it) {
                        if (fabs(*it)<core_radius) {
                            avgd += *it;
                            devd += *it * *it;
                            ++numd;
                        }
                    }
                }
                if (numd>0) avgd /= numd;
                if (numd>1) devd = sqrt( (devd - avgd*avgd*numd)/(numd-1.0) );
                
                *core_shift_bs_ref[ref12_idx] = normal * avgd;
                *core_shift_ref[ref12_idx] += normal * avgd;
                *plane_dev_ref[ref12_idx] += devd;
                *plane_nump_ref[ref12_idx] += numd;
            }
            
            // also bootstrap the diff
            // in each bootstrap: deltavec = (core+shift2) - (core+shift1) = shift2 - shift1;
            // after bootstrap: avg deltavec = avg shift 2 - avg shift 1
            // BUT avg deltavec.norm() != norm avg deltavec
            // => we want avg deltavec.norm() as the "diff" result
            // and the dev of that as well after bootstrap
            // AND we also want the average shifted core points
            // but these can be recovered from the avg shifts after bootstrap
            FloatType deltanorm = (core_shift_bs_2 - core_shift_bs_1).norm();
            diff += deltanorm;
            diff_dev += deltanorm * deltanorm;
        }
        
        core_shift_1 /= num_bootstrap_iter;
        core_shift_2 /= num_bootstrap_iter;
        
        normal_1.normalize();
        normal_2.normalize();
        
        plane_dev_1 /= num_bootstrap_iter;
        plane_dev_2 /= num_bootstrap_iter;
        
        plane_nump_1 /= num_bootstrap_iter;
        plane_nump_2 /= num_bootstrap_iter;
        
        diff /= num_bootstrap_iter;
        if (num_bootstrap_iter>1) diff_dev = sqrt( (diff_dev - diff*diff*num_bootstrap_iter)/(num_bootstrap_iter-1.0) );
        
        Point core1 = corepoints[ptidx] + core_shift_1;
        Point core2 = corepoints[ptidx] + core_shift_2;
        
        resultfile << core1.x << " " << core1.y << " " << core1.z << " ";
        resultfile << core2.x << " " << core2.y << " " << core2.z << " ";
        resultfile << normal_1.x << " " << normal_1.y << " " << normal_1.z << " ";
        resultfile << normal_2.x << " " << normal_2.y << " " << normal_2.z << " ";
        resultfile << normal_scale_idx_1 << " " << normal_scale_idx_2 << " ";
        resultfile << core_scale_idx_1 << " " << core_scale_idx_2 << " ";
        resultfile << plane_dev_1 << " " << plane_dev_2 << " ";
        resultfile << plane_nump_1 << " " << plane_nump_2 << " ";
        resultfile << diff << " " << diff_dev << endl;
    }
    
    cout << endl;
    
    resultfile.close();

    return 0;
}
