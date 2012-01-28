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
#include <boost/tokenizer.hpp>

#include <boost/format.hpp>

#define FLOAT_TYPE double
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
cout << "\
normaldiff normal_scale(s) : [cylinder_base : [cylinder_length : ]] p1.xyz[:p1reduced.xyz] p2.xyz[:p2reduced.xyz] cores.xyz extpts.xyz result.txt[:format] [opt_flags [extra_info]]\n\
  input: normal_scale(s) # The scale at which to compute the normal. If multiple scales\n\
                         # are given the one at which the cloud looks most 2D is used.\n\
                         # The syntax minscale:increment:maxscale is also accepted.\n\
                         # Use : to indicate the end of the list of scales\n\
  input: cylinder_base   # Optional. If given, the search cylinder for projecting the core\n\
                         # points in the normal direction is build with this base diameter\n\
                         # Default is to use the minimal scale given above.\n\
  input: cylinder_length # Optional. If given, the search cylinder for projecting the core\n\
                         # extends up to that distance in length.\n\
                         # Default is to use the maximal scale given above.\n\
  input: p1.xyz          # first whole raw point cloud to process\n\
  input: p2.xyz          # second whole raw point cloud to process, possibly the same as p1\n\
                         # if you only care for the normal computation and core point shifting.\n\
  input: p1reduced.xyz   # Optional: use this subsampled cloud for performing the normal\n\
                         # computations instead of p1.xyz. The projection is still performed\n\
                         # with the full-resolution cloud, but you may drastically accelerate\n\
                         # the computations by using a subsampled cloud if the scale at which\n\
                         # the normals are computed is very large.\n\
  input: p2.xyz          # second whole raw point cloud to process, possibly the same as p1\n\
                         # if you only care for the normal computation and core point shifting.\n\
  input: p2reduced.xyz   # Optional: See p1reduced.xyz.\n\
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
  outputs: result.txt[,e1,e2,e3:res2.txt,e4,...]\n\
                         # A file containing as many result lines as core points\n\
                         # All these results are influenced by the bootstrapping process except\n\
                         # the scale values that are fixed before the bootstrapping.\n\
                         # Each line contains space separated entries, that can be\n\
                         # specified in order by their names below (ex: c1,diff).\n\
                         # Several output files may be specified separated by ':',\n\
                         # each with their own optional comma separated entry\n\
                         # specifications (see the syntax above)\n\
                         # The default is to use all entries in this order:\n\
                         # - c1.x c1.y c1.z: core point coordinates, shifted to surface of p1.\n\
                         #   The surface is defined by a mean position (default) or by a median position (option \"q\").\n\
                         #   Use the format name \"c1\", all three entries are then given.\n\
                         # - c2.x c2.y c2.z: core point coordinates, shifted to surface of p2.\n\
                         # - n1.x n1.y n1.z: surface normal vector coordinates for p1\n\
                         #   Use the format name \"n1\", all three entries are then given.\n\
                         # - n2.x n2.y n2.z: surface normal vector coordinates for p2\n\
                         # - sn1 sn2: scales at which the normals were computed in p1 and p2\n\
                         # - dev1 dev2: standard deviation of the distance between: the points\n\
                         #              surrounding c1 and c2 at scales sc1 and sc2; and the\n\
                         #              \"surface\" planes defined by the normals at c1 and c2.\n\
                         #              This can be seen as a quality measure of the plane fitting. When option \"q\" is activated the interquartile ranges are used instead of standard deviations.\n\
                         # - np1 np2: number of points in these surroundings\n\
                         # - diff: average value of the signed distance c2 - c1, estimated by\n\
                         #         the statistical bootstrapping technique. Hence this value is\n\
                         #         not null when p1=p2, in that case it can be interpreted as\n\
                         #         a measure of the error made in shifting c to c1/2.\n\
                         # - diff_dev: deviation of the diff value, estimated by bootstrapping.\n\
  input: opt_flags       # Optional flags. Some may be combined together. Ex: \"hqb\".\n\
                         # Available flags are:\n\
                         #  q: Use the interquartile range and the median instead of standard deviation and mean, in order to define the new core point.\n\
                         #  h: make the resulting normals purely Horizontal (null z component).\n\
                         #  v: make the resulting normals purely Vertical (so, all normals are either +z or -z depending on the reference points).\n\
                         #  1: Shift the core point on both clouds using the normal computed on the first cloud\n\
                         #     The default is to shift the core point for each cloud using the normal computed on that cloud.\n\
                         #  2: Shift the core point on both clouds using the normal computed on the second cloud.\n\
                         #  m: Shift the core point on both clouds using the mean of both normals.\n\
                         #  e: provide the standard deviation of the Error on the point positions\n\
                         #     in p1 and p2 as extra info. This is a parameter provided by the device\n\
                         #     used for measuring p1 and p2. It is incorporated in the bootstrapping.\n\
                         #  b: Number of bootstrap iterations. 0 (the default) disables\n\
                         #     bootstrapping: you won't get diff_dev and diff is the sample mean.\n\
  input: extra_info      # Extra parameters for the \"e\", \"s\" and \"b\" flags, given in\n\
                         # the same order as these flags were specified.\n\
                         # Ex: normaldiff (all other opts) ehb 1e-2 1000\n\
                         # The flags are \"e\", \"h\" and \"b\". \"h\" has no extra parameter.\n\
                         # So, in this order: e=1e-2 and b=1000\n\
"<<endl;
    if (errmsg) cout << "Error: " << errmsg << endl;
    return 0;
}

// median: common definition using mid-point average in the even case
FloatType median(FloatType* values, int num) {
    if (num<1) return numeric_limits<FloatType>::quiet_NaN();
    int nd2 = num/2;
    FloatType med = values[nd2];
    if (num%2==0) { // even case
        med = (med + values[nd2-1]) * 0.5;
    }
    return med;
}

int main(int argc, char** argv) {

    if (argc<8) return help();

    int separator = 0;
    for (int i=1; i<argc; ++i) if (!strcmp(":",argv[i])) {
        separator = i;
        break;
    }
    if (separator==0) return help();

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
    
    FloatType cylinder_base = scalesvec.back(); // smallest
    FloatType cylinder_length = scalesvec[0];   // largest
    
    int separator_next = 0;
    for (int i=separator+1; i<argc; ++i) if (!strcmp(":",argv[i])) {
        separator_next = i;
        break;
    }
    if (separator_next!=0) {
        if (separator_next != separator+2) return help();
        cylinder_base = atof(argv[separator+1]);
        separator = separator_next;
        separator_next = 0;
        for (int i=separator+1; i<argc; ++i) if (!strcmp(":",argv[i])) {
            separator_next = i;
            break;
        }
        if (separator_next!=0) {
            if (separator_next != separator+2) return help();
            cylinder_length = atof(argv[separator+1]);
            separator = separator_next;
        }
    }
    
    // cylinder is split in segments using small balls, covering up the whole length
    // Analysis of the volume of the balls outside the cylinder wrt the number of balls
    // and their diameters required for covering the cylinder shows a minimum
    // Use twice the length as we also look for negative shifts
    int num_cyl_balls = floor(2.*cylinder_length*sqrt(2.)/cylinder_base);
    double cyl_section_length = cylinder_length/num_cyl_balls; 
    double cyl_ball_radius = 0.5*sqrt(cyl_section_length*cyl_section_length + cylinder_base*cylinder_base);
    double cylinder_base_radius_sq = cylinder_base * cylinder_base * 0.25;
    
    if (argc<separator+6) return help();

    cout << "Scale" << (scales.size()==1?"":"s") << " for normals computation:";
    for (ScaleSet::iterator it = scales.begin(); it!=scales.end(); ++it) {
        cout << " " << *it;
    }
    cout << endl;
    cout << "Base of projection/search cylinder: " << cylinder_base << endl;
    cout << "Length of projection/search cylinder: " << cylinder_length << endl;
    
    string p1fname = argv[separator+1];
    string p1reducedfname;
    int p1redpos = p1fname.find(':');
    if (p1redpos>=0) {
        p1reducedfname = p1fname.substr(p1redpos+1);
        p1fname = p1fname.substr(0,p1redpos);
    }
    string p2fname = argv[separator+2];
    string p2reducedfname;
    int p2redpos = p2fname.find(':');
    if (p2redpos>=0) {
        p2reducedfname = p2fname.substr(p2redpos+1);
        p2fname = p2fname.substr(0,p2redpos);
    }
    
    string corefname = argv[separator+3];
    string extptsfname = argv[separator+4];
    string resultarg = argv[separator+5];
    
    vector<string> result_filenames;
    vector<vector<string> > result_formats;
    
    const char* all_result_formats[] = {"c1", "c2", "n1", "n2", "sn1", "sn2", "dev1", "dev2", "np1", "np2", "diff", "diff_dev"};
    const int nresformats = 12;
    
    char_separator<char> colsep(":");
    char_separator<char> commasep(",");
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    tokenizer resfile_tokenizer(resultarg, colsep);
    for (tokenizer::iterator ftokit = resfile_tokenizer.begin(); ftokit != resfile_tokenizer.end(); ++ftokit) {
        // get each file specification
        tokenizer spec_tokenizer(*ftokit, commasep);
        tokenizer::iterator stokit = spec_tokenizer.begin();
        if (stokit == spec_tokenizer.end()) return help("Invalid result file format");
        // first token is the file name
        result_filenames.push_back(*stokit);
        vector<string> formats;
        // next are the format specifications
        for(++stokit; stokit!=spec_tokenizer.end(); ++stokit) {
            string format = *stokit;
            bool found = false;
            for (int i=0; i<nresformats; ++i) if (format==all_result_formats[i]) {
                found = true;
                break;
            }
            if (!found) return help(("Invalid result file format: "+format).c_str());
            formats.push_back(format);
        }
        // default is to output all fields
        if (formats.empty()) {
            for (int i=0; i<nresformats; ++i) formats.push_back(all_result_formats[i]);
        }
        result_formats.push_back(formats);
    }
    
    bool force_horizontal = false;
    bool force_vertical = false;
    bool use_median = false;
    bool shift_first = false;
    bool shift_second = false;
    bool shift_mean = false;
    double pos_dev = 0;
    double max_core_shift_distance = numeric_limits<double>::max();
    int num_bootstrap_iter = 1;
//    Point systematic_error(0,0,0);
    
    if (argc>separator+6) {
        int extra_info_idx = separator+6;
        for (char* opt=argv[separator+6]; *opt!=0; ++opt) {
            switch(*opt) {
                case '1': shift_first = true; break;
                case '2': shift_second = true; break;
                case 'm': shift_mean = true; break;
                case 'q': use_median = true; break;
                case 'h': force_horizontal = true; break;
                case 'v': force_vertical = true; break;
                case 'e': if (++extra_info_idx<argc) {
                    pos_dev = atof(argv[extra_info_idx]); break;
                } else return help("Missing value for the e flag");
                case 'b': if (++extra_info_idx<argc) {
                    num_bootstrap_iter = atoi(argv[extra_info_idx]); break;
                } else return help("Missing value for the b flag");
/*                case 's': if (extra_info_idx+3<argc) {
                    systematic_error.x = atof(argv[++extra_info_idx]);
                    systematic_error.y = atof(argv[++extra_info_idx]);
                    systematic_error.z = atof(argv[++extra_info_idx]);
                    break;
                } else return help("Missing value for the s flag"); break;
*/
                default: return help((string("Unrecognised optional flag: ")+opt).c_str());
            }
        }
    }
    
    if (shift_first && shift_second) return help("Conflicting 1 and 2 options for shifting core points.");
    if (shift_first && shift_mean) return help("Conflicting 1 and m options for shifting core points.");
    if (shift_mean && shift_second) return help("Conflicting 2 and m options for shifting core points.");
    
    if (force_horizontal && force_vertical) return help("Cannot force normals to be both horizontal and vertical!");
    
    if (num_bootstrap_iter<=0) {
        if (pos_dev>0) cout << "Warning: e flag is ignored when there is no bootstrap" << endl;
        num_bootstrap_iter = 1;
        pos_dev = 0;
    }
    
    boost::mt19937 rng;
    boost::normal_distribution<FloatType> poserr_dist(0, pos_dev);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<FloatType> > poserr_rand(rng, poserr_dist);

    cout << "Loading cloud 1: " << p1fname << endl;
    
    PointCloud<Point> p1, p1reduced;
    p1.load_txt(p1fname);
    if (!p1reducedfname.empty()) {
        cout << "Loading subsampled cloud 1: " << p1reducedfname << endl;
        p1reduced.load_txt(p1reducedfname);
    }
    
    cout << "Loading cloud 2: " << p2fname << endl;
    
    PointCloud<Point> p2, p2reduced;
    p2.load_txt(p2fname);
    if (!p2reducedfname.empty()) {
        cout << "Loading subsampled cloud 2: " << p2reducedfname << endl;
        p2reduced.load_txt(p2reducedfname);
    }
        
    cout << "Loading core points: " << corefname << endl;
    
    FILE* corepointsfile = fopen(corefname.c_str(), "r");
    if (!corepointsfile) {cerr << "Could not load file: " << corefname << endl; return 1;}
    char* line = 0; size_t linelen = 0; int num_read = 0;
    int linenum = 0;
    vector<Point> corepoints;
    while ((num_read = getline(&line, &linelen, corepointsfile)) != -1) {
        ++linenum;
        if (linelen==0 || line[0]=='#') continue;
        Point point;
        int i = 0;
        for (char* x = line; *x!=0;) {
            FloatType value = fast_atof_next_token(x);
            if (i<Point::dim) point[i++] = value;
            else break;
        }
        if (i<3) {cerr << "Error in the core points file" << corefname << " line " << (linenum+1) << endl; continue;}
        corepoints.push_back(point);
    }
    fclose(corepointsfile);

    cout << "Loading external reference points: " << extptsfname << endl;
    
    FILE* refpointsfile = fopen(extptsfname.c_str(), "r");
    if (!refpointsfile) {std::cerr << "Could not load file: " << extptsfname << std::endl; return 1;}
    linenum = 0;
    vector<Point> refpoints;
    while ((num_read = getline(&line, &linelen, refpointsfile)) != -1) {
        ++linenum;
        if (linelen==0 || line[0]=='#') continue;
        Point point;
        int i = 0;
        for (char* x = line; *x!=0;) {
            FloatType value = fast_atof_next_token(x);
            if (i<Point::dim) point[i++] = value;
            else break;
        }
        if (i<3) return help(str(boost::format("Error in the reference points file line %d") % (linenum+1)).c_str());
        refpoints.push_back(point);
    }
    fclose(refpointsfile);
    
    if (refpoints.empty()) return help("Please provide at least one reference point");
    
    for (int i = separator+6; i<argc; ++i) {
        if (i==separator+6) cout << "Options given:";
        cout << " " << argv[i];
        if (i==argc-1) cout << endl;
    }
    
    cout << "Computing result files: " << endl;
    for (int i=0; i<(int)result_filenames.size(); ++i) {
        cout << "  " << result_filenames[i] << ":";
        vector<string>& formats = result_formats[i];
        for (int j=0; j<(int)formats.size(); ++j) cout << " " << formats[j];
        cout << endl;
    }

    vector<ofstream*> resultfiles(result_filenames.size());
    for (int i=0; i<(int)result_filenames.size(); ++i) {
        resultfiles[i] = new ofstream(result_filenames[i].c_str());
    }
    
    
    // parameters and files loaded, now the real work
    
    cout << "Percent complete: 0" << flush;
    
    FloatType core_global_diff_mean = 0;
    FloatType core_global_diff_min = numeric_limits<FloatType>::max();
    FloatType core_global_diff_max = -numeric_limits<FloatType>::max();
    
    // for each core point
    int nextpercentcomplete = 5;
    for (int ptidx = 0; ptidx < (int)corepoints.size(); ++ptidx) {
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
        
        // Scales are sorted from max to lowest
        for (int scaleidx = 0; scaleidx<nscales; ++scaleidx) {
            // Neighborhood search only on max radius
            if (scaleidx==0) {
                // we have all neighbors, unsorted, but with distances computed already
                // use scales = diameters, not radius
                if (p1reducedfname.empty())
                    p1.findNeighbors(back_inserter(neighbors_1), corepoints[ptidx], scalesvec[scaleidx] * 0.5);
                else 
                    p1reduced.findNeighbors(back_inserter(neighbors_1), corepoints[ptidx], scalesvec[scaleidx] * 0.5);
                if (p2reducedfname.empty())
                    p2.findNeighbors(back_inserter(neighbors_2), corepoints[ptidx], scalesvec[scaleidx] * 0.5);
                else
                    p2reduced.findNeighbors(back_inserter(neighbors_2), corepoints[ptidx], scalesvec[scaleidx] * 0.5);
                // Sort the neighbors from closest to farthest, so we can process all lower scales easily
                sort(neighbors_1.begin(), neighbors_1.end());
                sort(neighbors_2.begin(), neighbors_2.end());
                neigh_num_1[scaleidx] = neighbors_1.size();
                neigh_num_2[scaleidx] = neighbors_2.size();
                // pre-compute cumulated sums
                // so we might as well share the intermediates to lower levels
                neighsums_1.resize(neighbors_1.size());
                if (!neighbors_1.empty()) neighsums_1[0] = *neighbors_1[0].pt;
                for (int i=1; i<(int)neighbors_1.size(); ++i) neighsums_1[i] = neighsums_1[i-1] + *neighbors_1[i].pt;
                neighsums_2.resize(neighbors_2.size());
                if (!neighbors_2.empty()) neighsums_2[0] = *neighbors_2[0].pt;
                for (int i=1; i<(int)neighbors_2.size(); ++i) neighsums_2[i] = neighsums_2[i-1] + *neighbors_2[i].pt;
            }
            // lower scale : restrict previously found neighbors to the new distance
            else {
                FloatType radiussq = scalesvec[scaleidx] * scalesvec[scaleidx] * 0.25;
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
        int normal_scale_idx_1 = 0, normal_scale_idx_2 = 0;
        FloatType svalues[3];
        // avoid code dup below
        // but some dup in bootstrapping as I'm lazy to get rid of it
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
                        *normal_scale_idx_ref[ref12_idx] = sidx;
                    }
                }                    
            }
        } //ref12 loop

        // closest ref point is also shared for all bootstrap iterations for efficiency
        // non-empty set => valid index
        int nearestidx = -1;
        FloatType mindist = numeric_limits<FloatType>::max();
        for (int i=0; i<(int)refpoints.size(); ++i) {
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
        double diff = 0, diff_dev = 0;

        // We have all core point neighbors at all scales in each data set
        // and the correct scales for the computation
        // Now bootstrapping...
        for (int bootstrap_iter = 0; bootstrap_iter < num_bootstrap_iter; ++bootstrap_iter) {
            
            Point core_shift_bs_1, core_shift_bs_2;
            Point normal_bs_1, normal_bs_2;
            int npts_scale0_bs_1, npts_scale0_bs_2;
            
            FloatType svalues[3];
            FloatType eigenvectors[9];
            
            vector<Point> resampled_neighbors_1, resampled_neighbors_2;
            // avoid code dup below
            int* npts_scale0_bs_ref[2] = {&npts_scale0_bs_1, &npts_scale0_bs_2};
            int* normal_scale_idx_ref[2] = {&normal_scale_idx_1, &normal_scale_idx_2};
            Point* normal_ref[2] = {&normal_1, &normal_2};
            Point* normal_bs_ref[2] = {&normal_bs_1, &normal_bs_2};
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
                int& npts_scale0 = *npts_scale0_bs_ref[ref12_idx];
                npts_scale0 = (*neigh_num_ref[ref12_idx])[0]; // ensured >=3 at this point
                resampled_neighbors.resize(npts_scale0);
                if (npts_scale0==0) continue;
                // resampling with replacement
                Point avg = 0;
                vector<FloatType> A(npts_scale0 * 3);
                boost::uniform_int<int> int_dist(0, npts_scale0-1);
                boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > randint(rng, int_dist);
                int normal_sidx = *normal_scale_idx_ref[ref12_idx];
                int npts_scaleN = 0;
                FloatType radiussq = scalesvec[normal_sidx] * scalesvec[normal_sidx] * 0.25;
                
                for (int i=0; i<npts_scale0; ++i) {
                    //int_dist(rng); // boost 1.47 is so much better :(
                    int selected_idx = randint();
                    // actually no boostrap => keep the original points
                    if (num_bootstrap_iter==1) selected_idx = i;
                    resampled_neighbors[i] = *neighbors[selected_idx].pt;
                    // add some gaussian noise with dev specified by the user on each coordinate
                    if (pos_dev>0) {
                        resampled_neighbors[i].x += poserr_rand(); //poserr_dist(rng);
                        resampled_neighbors[i].y += poserr_rand();
                        resampled_neighbors[i].z += poserr_rand();
                    }
                    if (force_vertical) continue;
                    // filter only the points within normal scale for the normal
                    // computation below
                    if ((corepoints[ptidx] - resampled_neighbors[i]).norm2()>=radiussq) continue;
                    ++npts_scaleN;
                    avg += resampled_neighbors[i];
                    A[i] = resampled_neighbors[i].x;
                    A[i+npts_scale0] = resampled_neighbors[i].y;
                    A[i+npts_scale0*2] = resampled_neighbors[i].z;
                }
            
                Point& normal = *normal_bs_ref[ref12_idx];
                
                // vertical normals need none of the SVD business
                if (force_vertical) normal.z = 1;
                else {
                    // need to reshape A, colomn-major for Fortran :(
                    if (npts_scaleN<npts_scale0) {
                        for (int i=0; i<npts_scaleN; ++i) A[i+npts_scaleN] = A[i+npts_scale0];
                        for (int i=0; i<npts_scaleN; ++i) A[i+npts_scaleN*2] = A[i+npts_scale0*2];
                    }
                    // now finish the averaging
                    avg /= npts_scaleN;
                    for (int i=0; i<npts_scaleN; ++i) {
                        A[i] -= avg.x;
                        A[i+npts_scaleN] -= avg.y;
                        A[i+npts_scaleN*2] -= avg.z;
                    }
                    if (force_horizontal) {
                        if (npts_scaleN<2 && bootstrap_iter==0) {
                            cout << "Warning: Invalid core point / data file / scale combination: less than 2 points at max scale for core point " << (ptidx+1) << " in data set " << ref12_idx+1 << endl;
                        } else {
                            // column-wise A: no need to reshape so as to skip z
                            // SVD decomposition handled by LAPACK
                            svd(npts_scaleN, 2, &A[0], &svalues[0], false, &eigenvectors[0]);
                            // The total least squares solution in the horizontal plane
                            // is given by the singular vector with minimal singular value
                            int mins = 0; if (svalues[1]<svalues[0]) mins = 1;
                            normal = Point(eigenvectors[mins], eigenvectors[2+mins], 0);
                        }
                    } else {
                        if (npts_scaleN<3 && bootstrap_iter==0) {
                            cout << "Warning: Invalid core point / data file / scale combination: less than 3 points at max scale for core point " << (ptidx+1) << " in data set " << ref12_idx+1 << endl;
                        } else {
                            svd(npts_scaleN, 3, &A[0], &svalues[0], false, &eigenvectors[0]);
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
            }

            // once the normal has been added to the bootstrap above,
            // we can freely replace the local iteration value
            // for the options "1", "2", and "m"
            if (shift_first) normal_bs_2 = normal_bs_1;
            if (shift_second) normal_bs_1 = normal_bs_2;
            if (shift_mean) {
                normal_bs_1 = (normal_bs_1 + normal_bs_2) * 0.5;
                normal_bs_2 = normal_bs_1;
            }
                
            for (int ref12_idx = 0; ref12_idx < 2; ++ref12_idx) {
                Point& normal = *normal_bs_ref[ref12_idx];

                // cylinder is split in segments using small balls, covering up the whole length
                // for median computations, if any
                vector<FloatType> all_dists_along_axis;
                // mean/dev
                FloatType mean_dist = 0;
                FloatType dev_dist = 0;
                int npts_in_cylinder = 0;
                
                // the number of segments includes negative shifts
                for (int cylsec=0; cylsec<num_cyl_balls; ++cylsec) {
                    
                    // first segment center starts at +0.5 from min neg shift
                    Point base_segment_center = corepoints[ptidx] + (cyl_section_length*0.5-cylinder_length) * normal;

                    FloatType min_dist_along_axis = cylsec * cyl_section_length - cylinder_length;
                    FloatType max_dist_along_axis = min_dist_along_axis + cyl_section_length;
                    
                    // find full-res points in the current cylinder section
                    p1.applyToNeighbors(
                        // long life to C++11 lambdas !
                        [&](FloatType d2, Point* p) {
                            Point delta = *p - corepoints[ptidx];
                            FloatType dist_along_axis = delta.dot(normal);
                            FloatType dist_to_axis_sq = (delta - dist_along_axis * normal).norm2();
                            // check the point is in this cylinder section
                            if (dist_to_axis_sq>cylinder_base_radius_sq) return;
                            if (dist_along_axis<min_dist_along_axis) return;
                            if (dist_along_axis>=max_dist_along_axis) return;
                            if (use_median) all_dists_along_axis.push_back(dist_along_axis);
                            else {
                                mean_dist += dist_along_axis;
                                dev_dist += dist_along_axis * dist_along_axis;
                                ++npts_in_cylinder;
                            }
                        },
                        base_segment_center + (cylsec * cyl_section_length) * normal,
                        cyl_ball_radius
                    );
                }
                
                // process average / median
                FloatType avgd = 0, devd = 0;
                if (use_median) {
                    sort(all_dists_along_axis.begin(), all_dists_along_axis.end());
                    // median instead of average
                    npts_in_cylinder = all_dists_along_axis.size();
                    avgd = median(&all_dists_along_axis[0], npts_in_cylinder);
                    // interquartile range
                    //   there are several ways to compute it, with no standard
                    //   commonly accepted definition. Use that of mathworld
                    int num_pts_each_half = (npts_in_cylinder+1)/2;
                    int offset_second_half = npts_in_cylinder/2;
                    FloatType q1 = median(&all_dists_along_axis[0], num_pts_each_half);
                    FloatType q3 = median(&all_dists_along_axis[offset_second_half], num_pts_each_half);
                    devd = q3 - q1;
                } else {
                    // standard mean / std dev
                    if (npts_in_cylinder>0) avgd = mean_dist / npts_in_cylinder;
                    if (npts_in_cylinder>1) devd = sqrt( (dev_dist - avgd*avgd*npts_in_cylinder)/(npts_in_cylinder-1.0) );
                }
                
                *core_shift_bs_ref[ref12_idx] = normal * avgd;
                *core_shift_ref[ref12_idx] += normal * avgd;
                *plane_dev_ref[ref12_idx] += devd;
                *plane_nump_ref[ref12_idx] += npts_in_cylinder;
            }
            
            // also bootstrap the diff
            // in each bootstrap: deltavec = (core+shift2) - (core+shift1) = shift2 - shift1;
            // after bootstrap: avg deltavec = avg shift 2 - avg shift 1
            // BUT avg deltavec.norm() != norm avg deltavec
            // => we want avg deltavec.norm() as the "diff" result
            // and the dev of that as well after bootstrap
            // AND we also want the average shifted core points
            // but these can be recovered from the avg shifts after bootstrap
            Point core_shift_diff = core_shift_bs_2 - core_shift_bs_1;
            FloatType deltanorm = core_shift_diff.norm();
            // signed distance according to the normal direction
            if (core_shift_diff.dot(normal_bs_1+normal_bs_2)<0) deltanorm *= -1;
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
        else diff_dev = 0;
        
        Point core1 = corepoints[ptidx] + core_shift_1;
        Point core2 = corepoints[ptidx] + core_shift_2;

        for (int i=0; i<(int)resultfiles.size(); ++i) {
            ofstream& resultfile = *resultfiles[i];
            vector<string>& formats = result_formats[i];
            for (int j=0; j<(int)formats.size(); ++j) {
                if (j>0) resultfile << " ";
                if (formats[j] == "c1") resultfile << core1.x << " " << core1.y << " " << core1.z;
                else if (formats[j] == "c2") resultfile << core2.x << " " << core2.y << " " << core2.z;
                else if (formats[j] == "n1") resultfile << normal_1.x << " " << normal_1.y << " " << normal_1.z;
                else if (formats[j] == "n2") resultfile << normal_2.x << " " << normal_2.y << " " << normal_2.z;
                else if (formats[j] == "sn1") resultfile << scalesvec[normal_scale_idx_1];
                else if (formats[j] == "sn2") resultfile << scalesvec[normal_scale_idx_2];
                else if (formats[j] == "dev1") resultfile << plane_dev_1;
                else if (formats[j] == "dev2") resultfile << plane_dev_2;
                else if (formats[j] == "np1") resultfile << plane_nump_1;
                else if (formats[j] == "np2") resultfile << plane_nump_2;
                else if (formats[j] == "diff") resultfile << diff;
                else if (formats[j] == "diff_dev") resultfile << diff_dev;
                else {
                    if (ptidx==0) cout << "Invalid result format \"" << formats[j] << "\" is ignored." << endl;
                }
            }        
            resultfile << endl;
        }
        
        core_global_diff_mean += diff;
        core_global_diff_min = min((FloatType)core_global_diff_min, (FloatType)diff);
        core_global_diff_max = max((FloatType)core_global_diff_min, (FloatType)diff);
    }
    cout << endl;
    
    core_global_diff_mean /= corepoints.size();
    cout << "Global diff min / mean / max on all core points: " << core_global_diff_min << " / " << core_global_diff_mean << " / " << core_global_diff_max << endl;

    for (int i=0; i<(int)resultfiles.size(); ++i) resultfiles[i]->close();
        
    return 0;
}
