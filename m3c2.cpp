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

#include <boost/math/special_functions/erf.hpp>
#include <boost/math/distributions/normal.hpp>

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

const char* all_result_formats[] = {"diff", "dev1", "dev2", "shift1", "shift2", "c1", "c2", "n1", "n2", "sn1", "sn2", "np1", "np2", "diff_bsdev", "shift1_bsdev", "shift2_bsdev", "ksi1", "ksi2", "n1angle_bs", "n2angle_bs", "diff_ci_low", "diff_ci_high", "diff_sig", "normal_dev1", "normal_dev2", "ns1", "ns2", "c0"};
const int nresformats = 28;
const char* default_result_formats[] = {"c1","n1","diff","diff_sig"};
const int num_default_result_formats = 4;

int help(const char* errmsg = 0) {
cout << "\
m3c2 normal_scale(s) : [cylinder_base : [cylinder_length : ]] p1.xyz[:p1reduced.xyz] p2.xyz[:p2reduced.xyz] cores.xyz extpts.xyz result.txt[,format[:result2.txt,format...]] [opt_flags [extra_info]]\n\
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
  outputs: result.txt[,v1,v2,v3:res2.txt,v4,...]\n\
                         # A file containing as many result lines as core points\n\
                         # Each line contains space separated output variables, that can be\n\
                         # specified in order by their names below (ex: c1,diff).\n\
                         # Several output files may be specified separated by ':',\n\
                         # each with their own optional comma separated variables\n\
                         # specification (see the syntax above)\n\
                         # The default is to use c1,n1,diff,diff_sig\n\
                         # Available output variables are:\n\
                         # - c1.x c1.y c1.z: core point coordinates, shifted to surface of p1.\n\
                         #   The surface is defined by a mean position (default) or by a median position (option \"q\").\n\
                         #   Use the format name \"c1\", all three entries are then given.\n\
                         # - c2.x c2.y c2.z: core point coordinates, shifted to surface of p2.\n\
                         # - c0.x c0.y c0.z: original core point coordinates.\n\
                         # - n1.x n1.y n1.z: surface normal vector coordinates for p1\n\
                         #   Use the format name \"n1\", all three entries are then given.\n\
                         # - n2.x n2.y n2.z: surface normal vector coordinates for p2\n\
                         # - sn1 sn2: scales at which the normals were computed in p1 and p2\n\
                         # - ns1 ns2: number of points in the neighborhood sphere at the sn1,sn2 scale.\n\
                         # - np1 np2: number of points in the cylinders\n\
                         # - diff: point cloud difference. \n\
                         #         value of the signed distance c2 - c1, possibly averaged\n\
                         #         by the statistical bootstrapping technique.\n\
                         # - diff_ci_low: lower bound of the confidence interval for the diff values, at 95% confidence level (default, see the c and g flags). The default is to estimate the confidence interval using the Bias-Corrected-accelerated (BCa) technique when bootstrapping is active, or to use a normality assumption otherwise.\n\
                         # - diff_ci_high: higher bound of the confidence interval for the diff values. See diff_ci_low.\n\
                         # - diff_sig: A boolean (0 or 1) indicating whether the diff value is outside the confidence interval AND whether there were enough points for estimating that interval itself. See the c, g, and p flags).\n\
                         # - shift1 shift2: signed distance the core points were shifted along the normals. This value is possibly averaged by bootstrap.\n\
                         # - dev1 dev2: standard deviation of the distance between: the points\n\
                         #              surrounding c1 and c2 in the cylinder; and the\n\
                         #              \"surface\" planes defined by the normals at c1 and c2.\n\
                         #         This value is possibly averaged by bootstrap.\n\
                         #              When option \"q\" is activated the interquartile ranges are used instead of standard deviations.\n\
                         # - diff_bsdev: deviation of the diff value, estimated by bootstrapping.\n\
                         # - shift1_bsdev shift2_bsdev: deviation of the shift values, estimated by bootstrapping.\n\
                         # - normal_dev1 normal_dev2: standard deviation of the points around the plane corresponding to the normal, including all points within neighborhood at the normal scale sn1/sn2\n\
                         # - ksi1 ksi2: a shortcut for sn1/dev1 and sn2/dev2\n\
                         # - n1angle_bs n2angle_bs: average angle (in degrees) between the normals and their mean value during normal bootstrapping. This can be seen as a kind of mean angular deviation around the normal, and is an indicator of how the normal is stable at that point. Requires the \"n\" flag.\n\
  input: opt_flags       # Optional flags. Some may be combined together. Ex: \"hqb\".\n\
                         # Available flags are:\n\
                         #  h: make the resulting normals purely Horizontal (null z component).\n\
                         #  v: make the resulting normals purely Vertical (so, all normals are either +z or -z depending on the reference points).\n\
                         #  1: Shift the core point on both clouds using the normal computed on the first cloud\n\
                         #     The default is to shift the core point for each cloud using the normal computed on that cloud.\n\
                         #  2: Shift the core point on both clouds using the normal computed on the second cloud.\n\
                         #  m: Shift the core point on both clouds using the mean of both normals.\n\
                         #  b: Number of bootstrap iterations for the standard deviation around the shifted position of the core points (default is 0, see the f flag). 0 disables this bootstrapping\n\
                         #  n: Number of bootstrap iterations for the estimation of the normals. 0 (the default) disables this bootstrapping. This option is useless when the normal is fixed to be vertical.\n\
                         #  c: Confidence interval (in %) for setting the significance bounds. Default is 95. This option is only effective when computing the diff_ci_xxx parameters.\n\
                         #  p: Minimal number of points that shall be in the cylinders (the np1 and np2 values), below which the the diff is considered not significant (the diff_sig value is set to 0). Default is 10.\n\
                         #  q: Use the interquartile range and the median instead of standard deviation and mean, in order to define the new core point. The confidence intervals, if requested, are computed using the quantiles set by the c flag, either on the bootstrapped distribution (if available) or on a distribution of combinations of diffs between the clouds (can actually be slower and less reliable than bootstrap, check it!)\n\
                         #  e: provide the standard deviation of the Error on the point positions\n\
                         #     in p1 and p2 as extra info. This is a parameter provided by the device\n\
                         #     used for measuring p1 and p2. It is incorporated in the bootstrapping.\n\
                         #  k: Value for the ksi parameter (default is 0) for selecting the scale at which the normal is computed. The smallest scale satisfying ksi > this_value will be used. See the paper for what ksi means. Use a value of 0 to disable this parameter. In that case, the scale at which the cloud looks most 2D is used instead. Note: The value of ksi computed at the normal selection stage may differ from the final value given by the ksi1/2 result specifiers in case of normal bootstrapping. Note 2: For contrieved geometries, it is a good idea to double-check the ksi values and the selected scales if something goes amiss (ksi may increase at small scales). Note3: Using this selection mode takes some extra cpu.\n\
                         #  f: (default, no need to specify) Fast-but-not-too-wrong estimator for the confidence intervals. This is Fast-and-exact only when the same normal is used, when each cloud is totally independant, and the points distances to their planes are distributed according to a Gaussian in each cylinder. These assumptions may fail, in which case use either the bootstrap technique (recommended) or maintain a Gaussian assumption and allow for normals to differ (g experimental flag, not recommended)\n\
                         #  g: EXPERIMENTAL. Assume a normal (Gaussian) distribution of the point distances around the mean shift(1/2) values for estimating the confidence interval of the diff values, but allow the normals to differ. The worst case relies on monte-carlo sampling of the joint distribution, which may be slower and less precise than boostrapping. This option dos not take into account the e flag.\n\
                         #  w: show extra warnings.\n\
  input: extra_info      # Extra parameters for the \"e\", \"s\", \"b\", \"n\", \"c\", \"k\" and \"p\" flags,\n\
                         # given in the same order as these flags were specified.\n\
                         # Ex: m3c2 (all other opts) ehb 1e-2 1000\n\
                         # The flags are \"e\", \"h\" and \"b\". \"h\" has no extra parameter.\n\
                         # So, in this order: e=1e-2 and b=1000\n\
"<<endl;
    if (errmsg) cout << "Error: " << errmsg << endl;
    return 0;
}

void mean_dev(double* values, int num, double& mean, double& dev) {
    if (num<1) {mean = dev = numeric_limits<double>::quiet_NaN(); return;}
    double sum = 0, ssq = 0;
    for (int i=0; i<num; ++i) {
        sum += values[i];
        ssq += values[i] * values[i];
    }
    mean = sum / num;
    if (num>1) dev = sqrt( (ssq - mean*mean*num)/(num-1.0) );
    else dev = 0;
}

double mean(double* values, int num) {
    if (num<1) {return numeric_limits<double>::quiet_NaN();}
    double sum = 0;
    for (int i=0; i<num; ++i) sum += values[i];
    return sum / num;
}

// median: common definition using mid-point average in the even case
double median(double* values, int num) {
    if (num<1) return numeric_limits<double>::quiet_NaN();
    int nd2 = num/2;
    double med = values[nd2];
    if (num%2==0) { // even case
        med = (med + values[nd2-1]) * 0.5;
    }
    return med;
}
// interquartile range
//   there are several ways to compute it, with no standard
//   commonly accepted definition. Use that of mathworld
double interquartile(double* values, int num) {
    int num_pts_each_half = (num+1)/2;
    int offset_second_half = num/2;
    double q1 = median(values, num_pts_each_half);
    double q3 = median(values + offset_second_half, num_pts_each_half);
    return q3 - q1;
}

vector<uint32_t> randtable;
int randtableidx = 0;
static const int rand_table_size = 0x8000; // 32768

void randinit() {
    randtable.resize(rand_table_size);
    boost::mt11213b rng;
    for (uint32_t& x : randtable) x = (uint32_t)rng();
}

int randint(const int nint) {
    randtableidx = (randtableidx+1) & 0x7FFF;
    return randtable[randtableidx] % nint;
}

void resample(const vector<double>& original, vector<double>& resampled) {
    const int nint = original.size();
    for (double& sample : resampled) sample = original[randint(nint)];
}

inline double inverse_normal_cdf(double p) {
    // use non-inf arithmetic, faster
    if (p<=0) return -numeric_limits<double>::max();
    if (p>=1) return numeric_limits<double>::max();
//    try {
    return 1.4142135623730950488016887242096981 * boost::math::erf_inv(2.*p-1.);
//     } catch(...) {
//         cout << "inverse_normal_cdf error: p=" << p << endl;
//         return 0;
//     }
}
inline double normal_cumulative(double x) {
    return 0.5 * (1. + boost::math::erf(x*0.7071067811865475244008443621048490));
}

/*
boost::mt11213b rng;

void resample(const vector<double>& original, vector<double>& resampled) {
    // resampling with replacement
    boost::uniform_int<int> int_dist(0, original.size()-1);
    boost::variate_generator<boost::mt11213b&, boost::uniform_int<int> > randint(rng, int_dist);
    for (double& sample : resampled) sample = original[randint()];
}
*/

inline double nan_is_0(double x) {
    return isfinite(x)?x:0;
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
    typedef set<double, greater<double> > ScaleSet;
    ScaleSet scales;
    for (int i=1; i<separator; ++i) {
        // perhaps it has the minscale:increment:maxscale syntax
        char* col1 = strchr(argv[i],':');
        char* col2 = strrchr(argv[i],':');
        if (col1==0 || col2==0 || col1==col2) {
            double scale = atof(argv[i]);
            if (scale<=0) return help("Invalid scale");
            scales.insert(scale);
        } else {
            *col1++=0;
            double minscale = atof(argv[i]);
            *col2++=0;
            double increment = atof(col1);
            double maxscale = atof(col2);
            if (minscale<=0 || maxscale<=0) return help("Invalid scale range");
            bool validRange = false;
            if ((minscale - maxscale) * increment > 0) return help("Invalid range specification");
            if (minscale<=maxscale) for (double scale = minscale; scale < maxscale*(1-1e-6); scale += increment) {
                validRange = true;
                scales.insert(scale);
            } else for (double scale = minscale; scale > maxscale*(1+1e-6); scale += increment) {
                validRange = true;
                scales.insert(scale);
            }
            // compensate roundoff errors for loop bounds
            scales.insert(minscale); scales.insert(maxscale);
            if (!validRange) return help("Invalid range specification");
        }
    }

    vector<double> scalesvec(scales.begin(), scales.end());
    int nscales = scalesvec.size();
    
    if (scales.empty()) return help();
    
    double cylinder_base = scalesvec.back(); // smallest
    double cylinder_length = scalesvec[0];   // largest
    
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
    double cyl_section_length = 2.*cylinder_length/num_cyl_balls; 
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
    
    bool compute_normal_angles = false, compute_shift_bsdev = false;
    double n1angle_bs = 0, n2angle_bs = 0;
    // set to true below by default
    // disabled below if another option is set
    bool fast_ci = true;
    bool compute_normal_plane_dev = false;
    
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
        if (formats.empty()) {
            for (int i=0; i<num_default_result_formats; ++i) formats.push_back(default_result_formats[i]);
        }
        for (auto format : formats) {
            if (format=="ksi1" || format=="ksi2" || format=="normal_dev1" || format=="normal_dev2") compute_normal_plane_dev = true;
            if (format=="n1angle_bs" || format=="n2angle_bs") compute_normal_angles = true;
            if (format=="shift1_bsdev" || format=="shift2_bsdev") compute_shift_bsdev = true;
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
    int num_bootstrap_iter = 1, num_normal_bootstrap_iter = 1;
    double confidence_interval_percent = 95.;
    bool normal_ci = false;
    bool use_BCa = false;
    int num_pt_sig = 10;
    double ksi_autoscale = 0;
    bool warnings = false;
    
    int np_prod_max = 10000;
    
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
                case 'w': warnings = true; break;
                case 'e': if (++extra_info_idx<argc) {
                    pos_dev = atof(argv[extra_info_idx]); break;
                } else return help("Missing value for the e flag");
                case 'b': if (++extra_info_idx<argc) {
                    use_BCa = true;
                    num_bootstrap_iter = atoi(argv[extra_info_idx]); break;
                } else return help("Missing value for the b flag");
                case 'n': if (++extra_info_idx<argc) {
                    num_normal_bootstrap_iter = atoi(argv[extra_info_idx]); break;
                } else return help("Missing value for the n flag");
                case 'c': if (++extra_info_idx<argc) {
                    confidence_interval_percent = atof(argv[extra_info_idx]); break;
                } else return help("Missing value for the c flag");
                case 'k': if (++extra_info_idx<argc) {
                    ksi_autoscale = atof(argv[extra_info_idx]); break;
                } else return help("Missing value for the c flag");
                case 'f': fast_ci = true; break;
                case 'g': normal_ci = true; break;
                case 'p': if (++extra_info_idx<argc) {
                    num_pt_sig = atoi(argv[extra_info_idx]); break;
                } else return help("Missing value for the p flag");
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
    
    if (num_bootstrap_iter<=0) num_bootstrap_iter = 1;
    if (num_bootstrap_iter==1) {
        if (compute_shift_bsdev) return help("Error: cannot compute the shift bootstrap deviation without bootstrapping, set the b flag.");
        if (use_BCa && !normal_ci) {
            // too experimental, do not set any confidence interval then
            // normal_ci = true;
            // use_BCa = false;
            return help("Error: confidence interval cannot be computed with bootstrap and only one iteration.");
        }
    }
    // honor the g flag
    if (normal_ci && !use_median) use_BCa = false;
    
    // disable the fast estimator if another technique is specified
    if (use_BCa || normal_ci) fast_ci = false;
    
    if (fast_ci && use_median) return help("Estimating confidence interval when using median/interquartile ranges needs bootstrapping. Please set the b flag");
    
    if (num_normal_bootstrap_iter<=0) num_normal_bootstrap_iter = 1;
    if (compute_normal_angles && (num_normal_bootstrap_iter==1) && warnings) cout << "Warning: cannot compute the normal angle deviations without bootstraping the normals, set the n flag." << endl;

    if (confidence_interval_percent<0 || confidence_interval_percent>100) {
        return help("Invalid confidence interval for the c flag, shall be between 0 and 100");
    }
    double z_low = 0., z_high = 0.;
    double cdf_low = 0., cdf_high = 0.;
    if (fast_ci || use_BCa || normal_ci) {
        cdf_low = 0.5 - confidence_interval_percent/200.;
        cdf_high = 0.5 + confidence_interval_percent/200.;
        z_low = inverse_normal_cdf(cdf_low);
        z_high = inverse_normal_cdf(cdf_high);
    }
    
    vector<double> *bs_dist = 0;
    if (use_BCa) bs_dist = new vector<double>(num_bootstrap_iter);
    
    boost::mt11213b rng;
    boost::normal_distribution<double> poserr_dist(0, pos_dev);
    boost::variate_generator<boost::mt11213b&, boost::normal_distribution<double> > poserr_rand(rng, poserr_dist);

    randinit();
    
    cout << "Loading cloud 1: " << p1fname << endl;
    
    PointCloud<Point> p1, p1reduced;
    p1.load_txt(p1fname);
    if (!p1reducedfname.empty()) {
        cout << "Loading subsampled cloud 1: " << p1reducedfname << endl;
        if (p1reduced.load_txt(p1reducedfname)==0) {
            cout << "Bad or empty subsampled cloud 1: " << p1reducedfname << endl;
            return -1;
        }
    }
    
    cout << "Loading cloud 2: " << p2fname << endl;
    
    PointCloud<Point> p2, p2reduced;
    p2.load_txt(p2fname);
    if (!p2reducedfname.empty()) {
        cout << "Loading subsampled cloud 2: " << p2reducedfname << endl;
        if (p2reduced.load_txt(p2reducedfname)==0) {
            cout << "Bad or empty subsampled cloud 2: " << p2reducedfname << endl;
            return -1;
        }
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
            double value = fast_atof_next_token(x);
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
            double value = fast_atof_next_token(x);
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

    
    map<string, string> formats_disp_map;
    for (int i=0; i<(int)result_filenames.size(); ++i) {
        for (auto f : result_formats[i]) formats_disp_map[f] = f;
    }
    formats_disp_map["c1"] = "c1.x c1.y c1.z";
    formats_disp_map["c2"] = "c2.x c2.y c2.z";
    formats_disp_map["n1"] = "n1.x n1.y n1.z";
    formats_disp_map["n2"] = "n2.x n2.y n2.z";
    formats_disp_map["c0"] = "c0.x c0.y c0.z";
    
    vector<ofstream*> resultfiles(result_filenames.size());
    for (int i=0; i<(int)result_filenames.size(); ++i) {
        resultfiles[i] = new ofstream(result_filenames[i].c_str());
        // add the variables as a comment for matlab/octave
        // but no space between # and the first variable for cloud compare
        *resultfiles[i] << "#";
        vector<string>& formats = result_formats[i];
        for (int j=0; j<(int)formats.size(); ++j) {
            if (j>0) *resultfiles[i] << " ";
            *resultfiles[i] << formats_disp_map[formats[j]];
        }
        *resultfiles[i] << endl;
        resultfiles[i]->precision(20);
    }
    
    // parameters and files loaded, now the real work
    
    cout << "Percent complete: 0" << flush;
    
    double core_global_diff_mean = 0;
    double core_global_diff_min = numeric_limits<double>::max();
    double core_global_diff_max = -numeric_limits<double>::max();
    int num_nan_diff = 0;
    int num_nan_c1 = 0;
    int num_nan_c2 = 0;
    
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
        
        if (!force_vertical)
        // Scales are sorted from max to lowest
        for (int scaleidx = 0; scaleidx<nscales; ++scaleidx) {
            // Neighborhood search only on max radius
            if (scaleidx==0) {
                // we have all neighbors, unsorted, but with distances computed already
                // use scales = diameters, not radius
                if (!shift_second) {
                    if (p1reducedfname.empty())
                        p1.findNeighbors(back_inserter(neighbors_1), corepoints[ptidx], scalesvec[scaleidx] * 0.5);
                    else 
                        p1reduced.findNeighbors(back_inserter(neighbors_1), corepoints[ptidx], scalesvec[scaleidx] * 0.5);
                    // Sort the neighbors from closest to farthest, so we can process all lower scales easily
                    if (nscales>1) sort(neighbors_1.begin(), neighbors_1.end());
                    neigh_num_1[scaleidx] = neighbors_1.size();
                    // pre-compute cumulated sums
                    // so we might as well share the intermediates to lower levels
                    neighsums_1.resize(neighbors_1.size());
                    if (!neighbors_1.empty()) neighsums_1[0] = *neighbors_1[0].pt;
                    for (int i=1; i<(int)neighbors_1.size(); ++i) neighsums_1[i] = neighsums_1[i-1] + *neighbors_1[i].pt;
                }
                if (!shift_first) {
                    if (p2reducedfname.empty())
                        p2.findNeighbors(back_inserter(neighbors_2), corepoints[ptidx], scalesvec[scaleidx] * 0.5);
                    else
                        p2reduced.findNeighbors(back_inserter(neighbors_2), corepoints[ptidx], scalesvec[scaleidx] * 0.5);
                    // Sort the neighbors from closest to farthest, so we can process all lower scales easily
                    if (nscales>1) sort(neighbors_2.begin(), neighbors_2.end());
                    neigh_num_2[scaleidx] = neighbors_2.size();
                    neighsums_2.resize(neighbors_2.size());
                    if (!neighbors_2.empty()) neighsums_2[0] = *neighbors_2[0].pt;
                    for (int i=1; i<(int)neighbors_2.size(); ++i) neighsums_2[i] = neighsums_2[i-1] + *neighbors_2[i].pt;
                }
            }
            // lower scale : restrict previously found neighbors to the new distance
            else {
                double radiussq = scalesvec[scaleidx] * scalesvec[scaleidx] * 0.25;
                // dicho search might be faster than sequencially from the vector end if there are many points
                int dichofirst = 0;
                int dicholast = neighbors_1.size();
                int dichomed;
                if (!shift_second) {
                    while (true) {
                        dichomed = (dichofirst + dicholast) / 2;
                        if (dichomed==dichofirst) break;
                        if (radiussq==neighbors_1[dichomed].distsq) break;
                        if (radiussq<neighbors_1[dichomed].distsq) { dicholast = dichomed; continue;}
                        dichofirst = dichomed;
                    }
                    // dichomed is now the last index with distance below or equal to requested radius
                    neigh_num_1[scaleidx] = dichomed+1;
                }
                // same for cloud 2
                if (!shift_first) {
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
        }

        // The most planar scale is only computed once if needed
        // bootstrapping is then done on that scale for the normal computations
        int normal_scale_idx_1 = 0, normal_scale_idx_2 = 0;
        
        int ref12_idx_begin = 0;
        int ref12_idx_end = 2;
        if (!compute_normal_plane_dev) {
            if (shift_first) ref12_idx_end = 1;
            if (shift_second) ref12_idx_begin = 1;
        }
        
        if (ksi_autoscale>0 || (nscales>1 && !force_vertical)) {
            double svalues[3]; double eigenvectors[9];
            // avoid code dup below
            // but some dup in bootstrapping as I'm lazy to get rid of it
            int* normal_scale_idx_ref[2] = {&normal_scale_idx_1, &normal_scale_idx_2};
            vector<DistPoint<Point> >* neighbors_ref[2] = {&neighbors_1, &neighbors_2};
            vector<int>* neigh_num_ref[2] = {&neigh_num_1, &neigh_num_2};
            vector<Point>* neighsums_ref[2] = {&neighsums_1, &neighsums_2};
            // loop on both pt sets, unless shift1/2 specified
            for (int ref12_idx = ref12_idx_begin; ref12_idx < ref12_idx_end; ++ref12_idx) {
                vector<DistPoint<Point> >& neighbors = *neighbors_ref[ref12_idx];
                double maxbarycoord = -numeric_limits<double>::max();
                // init to largest scale in case ksi condition is never verified below
                if (ksi_autoscale>0) *normal_scale_idx_ref[ref12_idx] = 0;
                for (int sidx=0; sidx<nscales; ++sidx) {
                    int npts = (*neigh_num_ref[ref12_idx])[sidx];
                    if (npts>=3) {
                        // use the pre-computed sums to get the average point
                        Point avg = (*neighsums_ref[ref12_idx])[npts-1] / npts;
                        // compute PCA on the neighbors at this scale
                        // a copy is needed as LAPACK destroys the matrix, and the center changes anyway
                        // => cannot keep the points from one scale to the lower, need to rebuild the matrix
                        vector<double> A(npts * 3);
                        for (int i=0; i<npts; ++i) {
                            // A is column-major
                            A[i] = neighbors[i].pt->x - avg.x;
                            A[i+npts] = neighbors[i].pt->y - avg.y;
                            A[i+npts*2] = neighbors[i].pt->z - avg.z;
                        }
                        
                        if (ksi_autoscale>0) {
                            vector<double> Acopy = A;
                            svd(npts, 2, &Acopy[0], &svalues[0], false, &eigenvectors[0]);
                            Point e1(eigenvectors[0], eigenvectors[3], eigenvectors[6]);
                            Point e2(eigenvectors[1], eigenvectors[4], eigenvectors[7]);
                            Point normal = e1.cross(e2);
                            double avg_dist_to_plane = 0.;
                            double ssq_dist_to_plane = 0.;
                            for (int i=0; i<npts; ++i) {
                                // A is now centered on 0, plane goes on 0
                                Point a(A[i], A[i+npts], A[i+npts*2]);
                                // so dist is easy to compute
                                double d = a.dot(normal);
                                avg_dist_to_plane += d;
                                ssq_dist_to_plane += d*d;
                            }
                            avg_dist_to_plane /= npts;
                            ssq_dist_to_plane = (ssq_dist_to_plane - npts * avg_dist_to_plane * avg_dist_to_plane) / (npts-1.);
                            ssq_dist_to_plane = max(0.,ssq_dist_to_plane);
                            
                            // when ssq_dist_to_plane==0, estimate is infinite
                            // which means all scales match, so end up with the lowest one
                            if (ssq_dist_to_plane==0) *normal_scale_idx_ref[ref12_idx] = sidx;
                            else {
                                double ksi = scalesvec[sidx] / sqrt(ssq_dist_to_plane);
                                if (ksi > ksi_autoscale) *normal_scale_idx_ref[ref12_idx] = sidx;
                            }                            
                        } else {
                            svd(npts, 3, &A[0], &svalues[0]);
                            
                            // The most 2D scale. For the criterion for how "2D" a scale is, see canupo
                            // Ideally first and second eigenvalue are equal
                            // convert to percent variance explained by each dim
                            double totalvar = 0;
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
                            //double c = 1 - a - b; // they sum to 1
                            // a = svalues[0] - svalues[1];
                            double b = 2 * svalues[0] + 4 * svalues[1] - 2;
                            if (b > maxbarycoord) {
                                maxbarycoord = b;
                                *normal_scale_idx_ref[ref12_idx] = sidx;
                            }
                        }
                    }                    
                }
            } //ref12 loop
        }

        // closest ref point is also shared for all bootstrap iterations for efficiency
        // non-empty set => valid index
        int nearestidx = -1;
        double mindist = numeric_limits<double>::max();
        for (int i=0; i<(int)refpoints.size(); ++i) {
            double d = dist2(refpoints[i],corepoints[ptidx]);
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
        
        vector<Point> *normal_bs_sample1 = 0;
        vector<Point> *normal_bs_sample2 = 0;
        if (compute_normal_angles) {
            normal_bs_sample1 = new vector<Point>(num_normal_bootstrap_iter);
            normal_bs_sample2 = new vector<Point>(num_normal_bootstrap_iter);
        }
        
        double normal_dev1 = 0, normal_dev2 = 0;
        
        // largest possible A, reused at each bootstrap step
        vector<double> A1(neigh_num_1[0] * 3);
        vector<double> A2(neigh_num_2[0] * 3);
        vector<double>* A_ref[2] = {&A1, &A2};
        // lapack destroys the matrix, we need it for compute_normal_plane_dev
        vector<double> A1copy(0), A2copy(0);
        if (compute_normal_plane_dev) {
            A1copy.resize(A1.size());
            A2copy.resize(A2.size());
        }
        vector<double>* Acopy_ref[2] = {&A1copy, &A2copy};

        // We have all core point neighbors at all scales in each data set
        // and the correct scales for the computation
        // Now bootstrapping...
        for (int n_bootstrap_iter = 0; n_bootstrap_iter < num_normal_bootstrap_iter; ++n_bootstrap_iter) {
    
            Point normal_bs_1, normal_bs_2;
            int npts_scale0_bs_1, npts_scale0_bs_2;
            
            double svalues[3];
            double eigenvectors[9];
            
            vector<Point> resampled_neighbors_1, resampled_neighbors_2;
            // avoid code dup below
            int* npts_scale0_bs_ref[2] = {&npts_scale0_bs_1, &npts_scale0_bs_2};
            int* normal_scale_idx_ref[2] = {&normal_scale_idx_1, &normal_scale_idx_2};
            Point* normal_ref[2] = {&normal_1, &normal_2};
            Point* normal_bs_ref[2] = {&normal_bs_1, &normal_bs_2};
            vector<DistPoint<Point> >* neighbors_ref[2] = {&neighbors_1, &neighbors_2};
            vector<int>* neigh_num_ref[2] = {&neigh_num_1, &neigh_num_2};
            vector<Point>* resampled_neighbors_ref[2] = {&resampled_neighbors_1, &resampled_neighbors_2};
            double* normal_dev_ref[2] = {&normal_dev1, &normal_dev2};
            int npts_scaleN_1 = 0, npts_scaleN_2 = 0;
            int* npts_scaleN_ref[2] = {&npts_scaleN_1, &npts_scaleN_2};
                        
            // loop on both pt sets
            for (int ref12_idx = ref12_idx_begin; ref12_idx < ref12_idx_end; ++ref12_idx) {
                Point& normal = *normal_bs_ref[ref12_idx];
                // vertical normals need none of the SVD business
                if (force_vertical) {
                    normal.z = (deltaref.z<0) ? -1 : 1;
                    // but they may require a deviation around the plane
                    if (!compute_normal_plane_dev) continue;
                }

                vector<DistPoint<Point> >& neighbors = *neighbors_ref[ref12_idx];
                int normal_sidx = *normal_scale_idx_ref[ref12_idx];
                int npts_scale_base = (*neigh_num_ref[ref12_idx])[normal_sidx];
                Point avg = 0;
                vector<double>& A = *A_ref[ref12_idx];
                vector<double>& Acopy = *Acopy_ref[ref12_idx];
                int& npts_scaleN = *npts_scaleN_ref[ref12_idx];
                npts_scaleN = 0;
                double radiussq = scalesvec[normal_sidx] * scalesvec[normal_sidx] * 0.25;
                Point bspt;
                for (int i=0; i<npts_scale_base; ++i) {
                    Point* pt = neighbors[i].pt;
                    if (num_normal_bootstrap_iter>1) {
                        //int selectedidx = randint();
                        int selectedidx = randint(npts_scale_base);
                        pt = neighbors[selectedidx].pt;
                        // add some gaussian noise with dev specified by the user on each coordinate
                        if (pos_dev>0 && !force_vertical) {
                            bspt = *pt;
                            bspt.x += poserr_rand(); //poserr_dist(rng);
                            bspt.y += poserr_rand();
                            bspt.z += poserr_rand();
                            pt = &bspt;
                        }
                    }
                    // filter only the points within normal scale for the normal
                    // computation below
                    if ((corepoints[ptidx] - *pt).norm2()>=radiussq) continue;
                    avg += *pt;
                    A[npts_scaleN] = pt->x;
                    A[npts_scaleN+npts_scale_base] = pt->y;
                    A[npts_scaleN+npts_scale_base*2] = pt->z;
                    ++npts_scaleN;
                }
                // need to reshape A, colomn-major for Fortran :(
                if (npts_scaleN<npts_scale_base) {
                    for (int i=0; i<npts_scaleN; ++i) A[i+npts_scaleN] = A[i+npts_scale_base];
                    for (int i=0; i<npts_scaleN; ++i) A[i+npts_scaleN*2] = A[i+npts_scale_base*2];
                }
                // now finish the averaging
                avg /= npts_scaleN;
                for (int i=0; i<npts_scaleN; ++i) {
                    A[i] -= avg.x;
                    A[i+npts_scaleN] -= avg.y;
                    A[i+npts_scaleN*2] -= avg.z;
                }
                if (compute_normal_plane_dev) for (int i=0; i<npts_scaleN*3; ++i) Acopy [i] = A[i];
                
                if (!force_vertical) {
                    if (force_horizontal) {
                        if (npts_scaleN<2 && n_bootstrap_iter==0) {
                             if (warnings) cout << "Warning: Invalid core point / data file / scale combination: less than 2 points at max scale for core point " << (ptidx+1) << " in data set " << ref12_idx+1 << endl;
                        } else {
                            // column-wise A: no need to reshape so as to skip z
                            // SVD decomposition handled by LAPACK
                            svd(npts_scaleN, 2, &A[0], &svalues[0], false, &eigenvectors[0]);
                            // The total least squares solution in the horizontal plane
                            // is given by the singular vector with minimal singular value
                            int mins = 0; if (svalues[1]<svalues[0]) mins = 1;
                            normal = Point(eigenvectors[mins], eigenvectors[2+mins], 0);
                            normal.normalize();
                        }
                    } else {
                        if (npts_scaleN<3 && n_bootstrap_iter==0) {
                            if (warnings) cout << "Warning: Invalid core point / data file / scale combination: less than 3 points at max scale for core point " << (ptidx+1) << " in data set " << ref12_idx+1 << endl;
                        } else {
                            svd(npts_scaleN, 3, &A[0], &svalues[0], false, &eigenvectors[0]);
//std::cout << svalues[0] << " " << svalues[1] << " " << svalues[2] << std::endl;
                            // column-major matrix, eigenvectors as rows
                            Point e1(eigenvectors[0], eigenvectors[3], eigenvectors[6]);
                            Point e2(eigenvectors[1], eigenvectors[4], eigenvectors[7]);
                            normal = e1.cross(e2);
                        }
                    }
                    // normal orientation... simple with external help
                    if (normal.dot(deltaref)<0) normal *= -1;
                }
            }
            
            // replace the local iteration value for the options "1", "2", and "m"
            if (shift_first) normal_bs_2 = normal_bs_1;
            if (shift_second) normal_bs_1 = normal_bs_2;
            if (shift_mean) {
                if (normal_bs_1.norm2()==0) {
                    if (warnings) cout << "Warning: null normal on Cloud 1 for core point " << (ptidx+1) << endl;
                    normal_bs_1 = normal_bs_2;
                }
                if (normal_bs_2.norm2()==0) {
                    if (warnings) cout << "Warning: null normal on Cloud 2 for core point " << (ptidx+1) << endl;
                    normal_bs_2 = normal_bs_1;
                }
                normal_bs_1 = (normal_bs_1 + normal_bs_2) * 0.5;
                normal_bs_2 = normal_bs_1;
            }
            else {
                if (normal_bs_1.norm2()==0) {
                    if (warnings) cout << "Warning: null normal on Cloud 1 for core point " << (ptidx+1) << ", setting it to the normal computed on Cloud 2" << endl;
                    normal_bs_1 = normal_bs_2;
                }
                if (normal_bs_2.norm2()==0) {
                    if (warnings) cout << "Warning: null normal on Cloud 2 for core point " << (ptidx+1) << ", setting it to the normal computed on Cloud 1" << endl;
                    normal_bs_2 = normal_bs_1;
                }
            }
            
            if (compute_normal_plane_dev) {
                for (int ref12_idx = ref12_idx_begin; ref12_idx < ref12_idx_end; ++ref12_idx) {
                    vector<double>& A = *Acopy_ref[ref12_idx];
                    int& npts_scaleN = *npts_scaleN_ref[ref12_idx];
                    Point& normal = *normal_bs_ref[ref12_idx];
                    double avg_dist_to_plane = 0.;
                    double ssq_dist_to_plane = 0.;
                    for (int i=0; i<npts_scaleN; ++i) {
                        // A is now centered on 0, plane goes on 0
                        Point a(A[i], A[i+npts_scaleN], A[i+npts_scaleN*2]);
                        // so dist is easy to compute
                        double d = a.dot(normal);
                        avg_dist_to_plane += d;
                        ssq_dist_to_plane += d*d;
                    }
                    if (npts_scaleN>1) {
                        avg_dist_to_plane /= npts_scaleN;
                        ssq_dist_to_plane = (ssq_dist_to_plane - npts_scaleN * avg_dist_to_plane * avg_dist_to_plane) / (npts_scaleN-1.);
                        ssq_dist_to_plane = max(0.,ssq_dist_to_plane);
                    }
                    *normal_dev_ref[ref12_idx] += sqrt(ssq_dist_to_plane);
                }
                if (shift_first) normal_dev2 = normal_dev1;
                if (shift_second) normal_dev1 = normal_dev2;
            }
                
            // bootstrap mean normal
            normal_1 += normal_bs_1;
            normal_2 += normal_bs_2;
            
            if (compute_normal_angles) {
                (*normal_bs_sample1)[n_bootstrap_iter] = normal_bs_1;
                (*normal_bs_sample2)[n_bootstrap_iter] = normal_bs_2;
            }
        }
        
        normal_1.normalize();
        normal_2.normalize();
        
        if (compute_normal_plane_dev) {
            normal_dev1 /= num_normal_bootstrap_iter;
            normal_dev2 /= num_normal_bootstrap_iter;
        }

        // angles between normals and bs mean = a kind of directional deviation...
        // ... and a way to detect bad normals
        if (compute_normal_angles) {
            double dprod1 = 0, dprod2 = 0;
            for (int n_bootstrap_iter = 0; n_bootstrap_iter < num_normal_bootstrap_iter; ++n_bootstrap_iter) {
                dprod1 += (*normal_bs_sample1)[n_bootstrap_iter].dot(normal_1);
                dprod2 += (*normal_bs_sample2)[n_bootstrap_iter].dot(normal_2);
            }
            n1angle_bs = acos(dprod1 / num_normal_bootstrap_iter) * 180 / M_PI;
            n2angle_bs = acos(dprod2 / num_normal_bootstrap_iter) * 180 / M_PI;
            delete normal_bs_sample1;
            delete normal_bs_sample2;
        }
        
        /// estimate the diff separately from the normals
        /// First get all points in the cylinder
        /// then bootstrap to estimate the variance around the core shift distance
        vector<double> distances_along_axis_1;
        vector<double> distances_along_axis_2;
        Point* normal_ref[2] = {&normal_1, &normal_2};
        vector<double>* distances_along_axis_ref[2] = {&distances_along_axis_1, &distances_along_axis_2};
        
        for (int ref12_idx = 0; ref12_idx < 2; ++ref12_idx) {
            Point& normal = *normal_ref[ref12_idx];
            vector<double>& distances_along_axis = *distances_along_axis_ref[ref12_idx];

            // the number of segments includes negative shifts
            for (int cylsec=0; cylsec<num_cyl_balls; ++cylsec) {
                
                // first segment center starts at +0.5 from min neg shift
                Point base_segment_center = corepoints[ptidx] + (cyl_section_length*0.5-cylinder_length) * normal;

                double min_dist_along_axis = cylsec * cyl_section_length - cylinder_length;
                double max_dist_along_axis = min_dist_along_axis + cyl_section_length;
                
                // find full-res points in the current cylinder section
                ((ref12_idx==0)?p1:p2).applyToNeighbors(
                    // long life to C++11 lambdas !
                    [&](double d2, Point* p) {
                        Point delta = *p - corepoints[ptidx];
                        double dist_along_axis = delta.dot(normal);
                        double dist_to_axis_sq = (delta - dist_along_axis * normal).norm2();
                        // check the point is in this cylinder section
                        if (dist_to_axis_sq>cylinder_base_radius_sq) return;
                        if (dist_along_axis<min_dist_along_axis) return;
                        if (dist_along_axis>=max_dist_along_axis) return;
                        distances_along_axis.push_back(dist_along_axis);
                    },
                    base_segment_center + (cylsec * cyl_section_length) * normal,
                    cyl_ball_radius
                );
            }
        }
        
        int np1 = distances_along_axis_1.size();
        int np2 = distances_along_axis_2.size();
        
        // prepare bootstrap for processing average / median.
        
        vector<double>* daa1;
        vector<double>* daa2;
        if (num_bootstrap_iter==1) {
            daa1 = &distances_along_axis_1;
            daa2 = &distances_along_axis_2;
        } else {
            daa1 = new vector<double>(np1);
            daa2 = new vector<double>(np2);
        }

        // allow for some small numerical roundoff errors
        // The unique normal case simplifies a lot the confidence interval algorithms
        bool same_normal = (normal_1.dot(normal_2)>1.-1e-6);
        
        // Notes on using the normality assumption
        
        // d = dist(c+s1.n1,c+s2.n2)
        // d^2 = s1^2 + s2^2 - 2.s1.s2.cos(n1,n2)
        // E[d] is a bit problematic
        // E[d] = ∬ d(s1,s2) p(s1,s2) δs1 δs2
        // using p(s1,s2) = G(s1)G(s2) independent gaussians separation
        // E[d] = m ∬ sqrt(s1^2 + s2^2 - 2.s1.s2.cos(n1,n2)) G(s1)G(s2) δs1 δs2
        // Using signed distances such that m = sign( (s2n2-s1n1).extpt )
        // ⇒ Can be integrated numerically
        // var(d) = E[d^2] - E[d]^2
        // E[d^2] = E[s1^2 + s2^2 - 2.s1.s2.cos(n1,n2)]
        // Using independence of variations in each cloud, E[s1s2]=E[s1]E[s2]
        // E[d^2] = E[s1^2] + E[s2^2] - 2.E[s1].E[s2].cos(n1,n2)
        // Easy E[d^2], difficult E[d], but manageable
        
        // Check that this is consistent with n1=n2 ⇒ cos(n1,n2)=1
        // E'[d] = m ∬ sqrt(s1^2 + s2^2 - 2.s1.s2) G(s1)G(s2) δs1 δs2
        // E'[d] = m ∬ |s1-s2| G(s1)G(s2) δs1 δs2
        // E'[d] = ∬ (s2-s1) G(s1)G(s2) δs1 δs2
        // ± E'[d] = ∬ s2 G(s1)G(s2) δs1 δs2 - ∬ s1 G(s1)G(s2) δs1 δs2
        // ± E'[d] = ∫ G(s1) (∫s2G(s2)δs2) δs1 - ∫ G(s2) (∫s1G(s1)δs1) δs2
        // ± E'[d] = ∫ G(s1) E[s2] δs1 - ∫ G(s2) E[s1] δs2
        // ± E'[d] = E[s2] ∫ G(s1) δs1 - E[s1] ∫ G(s2) δs2
        // ± E'[d] = E[s2] - E[s1] = E[s2-s1]    OK, this is the 1D case !!!

        double avgd1, avgd2, devd1, devd2;
        double sample_diff = 0, BC_acceleration_factor = 0.;
        
        //double sample_dev = 0;
        double ci_low = 0., ci_high = 0.;
        double c1shift = 0, c2shift = 0;
        double c1dev = 0, c2dev = 0;
        double diff = 0;
        // work on sample distribution
        if (fast_ci || normal_ci || use_BCa || (num_bootstrap_iter==1)) {
            // use median ⇒ no BCa, see below for using the bootstrap distribution instead
            // use the quartiles for the confidence_interval_percent then
            if (use_median && (normal_ci || (num_bootstrap_iter==1))) {
                // rely on random sampling when there are too many combinations
                vector<double> sample_deltanorm(min(np1*np2,np_prod_max),0.);
                if (np1*np2>np_prod_max) for (int i=0; i<np_prod_max; ++i) {
                    double d1 = distances_along_axis_1[randint(np1)];
                    double d2 = distances_along_axis_2[randint(np2)];
                    sample_deltanorm[i] = (d2 * normal_2 - d1 * normal_1).norm();
                } else for (int i=0; i<np1; ++i) for (int j=0; j<np2; ++j) {
                    double d1 = distances_along_axis_1[i];
                    double d2 = distances_along_axis_2[j];
                    sample_deltanorm[i*np2+j] = (d2 * normal_2 - d1 * normal_1).norm();
                }
                sort(sample_deltanorm.begin(), sample_deltanorm.end());
                if (normal_ci) {
                    int idxlow = max(0, min((int)floor(((1.-confidence_interval_percent*0.01) * 0.5) * sample_deltanorm.size()), (int)sample_deltanorm.size()-1));
                    int idxhigh = max(0, min((int)floor((1.-(1.-confidence_interval_percent*0.01) * 0.5) * sample_deltanorm.size()), (int)sample_deltanorm.size()-1));
                    ci_low = sample_deltanorm[idxlow];
                    ci_high = sample_deltanorm[idxhigh];
                }
                if (num_bootstrap_iter==1) {
                    c1shift = median(&distances_along_axis_1[0], np1);
                    c2shift = median(&distances_along_axis_2[0], np2);
                    c1dev = interquartile(&distances_along_axis_1[0], np1);
                    c2dev = interquartile(&distances_along_axis_2[0], np2);
                    diff = median(&sample_deltanorm[0], sample_deltanorm.size());
                }
            }
            if (!use_median) {
                int nsamples = np1*np2;
                vector<double> samples(0);
                if (fast_ci || (same_normal && !use_BCa)) {
                    if (fast_ci || num_bootstrap_iter==1) {
                        mean_dev(&distances_along_axis_1[0], np1, c1shift, c1dev);
                        mean_dev(&distances_along_axis_2[0], np2, c2shift, c2dev);
                        diff = sample_diff = c2shift - c1shift;
                    }
                    if (num_bootstrap_iter>1) sample_diff = mean(&distances_along_axis_2[0], np2) - mean(&distances_along_axis_1[0], np1);
                }
                else {
                    sample_diff = 0.;
                    nsamples = min(np1*np2,np_prod_max);
                    if (use_BCa) samples.resize(nsamples);
                    for (int sidx=0; sidx<nsamples; ++sidx) {
                        int i1, i2;
                        if (nsamples==np_prod_max) {
                            i1 = randint(np1);
                            i2 = randint(np2);
                        } else {
                            i1 = sidx / np2;
                            i2 = sidx - i1 * np2;
                        }
                        double s1 = distances_along_axis_1[i1];
                        double s2 = distances_along_axis_2[i2];
                        Point sn = (distances_along_axis_2[i2] * normal_2 - distances_along_axis_1[i1] * normal_1);
                        double d = sn.norm();
                        if (sn.dot(deltaref)<0) d *= -1;
                        sample_diff += d;
                        //sample_dev += d*d;
                        if (use_BCa) samples[sidx] = d;
                    }
                    sample_diff /= nsamples;
                    //sample_dev = (nsamples>1) ? sqrt(max(0.,(sample_dev - nsamples*sample_diff*sample_diff) / (nsamples - 1.))) : 0;
                    if (num_bootstrap_iter==1) {
                        mean_dev(&distances_along_axis_1[0], np1, c1shift, c1dev);
                        mean_dev(&distances_along_axis_2[0], np2, c2shift, c2dev);
                        diff = sample_diff;
                    }
                }
                
                // assumption of normality around each plane case
                if (normal_ci) {
                    // compute the quantiles directly from the theoretical distribution
                    if (num_bootstrap_iter==1) {
                        avgd1 = c1shift; devd1 = c1dev;
                        avgd2 = c2shift; devd2 = c2dev;
                    } else {
                        mean_dev(&distances_along_axis_1[0], np1, avgd1, devd1);
                        mean_dev(&distances_along_axis_2[0], np2, avgd2, devd2);
                    }
                    if (devd1==0) devd1 = avgd1 * 1e-48;
                    if (devd1==0) devd1 = 1e-48;
                    boost::math::normal G1(avgd1, devd1);
                    if (devd2==0) devd2 = avgd2 * 1e-48;
                    if (devd2==0) devd2 = 1e-48;
                    boost::math::normal G2(avgd2, devd2);
                    // go from ±4σ in each distribution, i.e. p ≈ 5.34e-5
                    // which is more than enough for quantile estimation
                    // hope to use a fine enough discretization...
                    // ...at every 0.02 quantile in each dist, shall be OK
                    vector<double> x1(50), x2(50), p1(50), p2(50);
                    struct DP {double d, p;};
                    vector<DP> dp(49*49, {0.,0.});
                    for (int i=1; i<=49; ++i) {
                        x1[i] = quantile(G1,i * 0.02);
                        p1[i] = pdf(G1,x1[i]);
                    }
                    for (int j=1; j<=49; ++j) {
                        x2[j] = quantile(G2,j * 0.02);
                        p2[j] = pdf(G2,x2[j]);
                    }
                    double cosn1n2 = normal_1.dot(normal_2);
                    double mind = numeric_limits<double>::max();
                    double maxd = -numeric_limits<double>::max();
                    double meand = 0, sump = 0;
                    for (int i=1; i<=49; ++i) for (int j=1; j<=49; ++j) {
                        //(i-1)*49+(j-1)
                        dp[i*49+j-50].d = sqrt(max(0.,x1[i]*x1[i]+x2[j]*x2[j]-2*x1[i]*x2[j]*cosn1n2));

                        Point sn = (x2[j] * normal_2 - x1[i] * normal_1);
                        if (sn.dot(deltaref)<0) dp[i*49+j-50].d *= -1;
                        
                        dp[i*49+j-50].p = p1[i] * p2[j];
                        
                        meand += dp[i*49+j-50].d * dp[i*49+j-50].p;
                        sump += dp[i*49+j-50].p;
                    }
                    meand /= sump;
                    // estimate the sample mean stats, not the distance stats
                    // convert to same distribution rescaled to have sample mean dev
                    double smeanfactor = 1.0 / sqrt((double)nsamples);
                    for (int i=0; i<49*49; ++i) {
                        dp[i].d = meand + (dp[i].d - meand) * smeanfactor;
                        if (dp[i].d<mind) mind = dp[i].d;
                        if (dp[i].d>maxd) maxd = dp[i].d;
                    }
                    
                    // now fill 200 bins within min/max range
                    double bins[200]; for (int i=0; i<200; ++i) bins[i] = 0;
                    double extent = maxd - mind;
                    mind = (mind + maxd - extent)*0.5;
                    double binsize = extent / 200.0;
                    double binsizeinv = 200.0 / extent;
                    for (int i=0; i<49*49; ++i) {
                        int idx = max(0,min(199,(int)floor((dp[i].d - mind) * binsizeinv)));
                        bins[idx] += dp[i].p;
                    }
                    // integrate to convert to CDF
                    for (int i=1; i<200; ++i) bins[i] += bins[i-1];
                    // renormalize to 1 to compensate discretization
                    // update CI bounds
                    ci_low = mind + 0.5 * binsize;
                    ci_high = mind + 0.5 * binsize;
                    double m = bins[0];
                    double r = bins[199] - bins[0];
                    for (int i=0; i<200; ++i) {
                        bins[i] = (bins[i] - m) / r;
                        double center = mind+(i+0.5)*binsize;
                        if (bins[i]<=cdf_low) ci_low = center;
                        ci_high = center;
                        if (bins[i]>cdf_high) break;
                    }
                // BCa case
                } else if (use_BCa) {
                    // Mean value statistic, formula (7.4) in Efron's 87 paper
                    double s3 = 0., s2 = 0.;
                    for (int sidx=0; sidx<nsamples; ++sidx) {
                        double Ui = samples[sidx] - sample_diff;
                        double Ui2 = Ui*Ui;
                        s2 += Ui2;
                        s3 += Ui2 * Ui;
                    }
                    BC_acceleration_factor = s3 / (6. * sqrt(s2*s2*s2));
                    // ci_low, ci_high done below
                }
            }
        }
                        
        // bootstrap, if needed. case num_bootstrap_iter==1 done above
        double diff_bsdev = 0;
        double z0_sum = 0;
        double c1shift_bsdev = 0, c2shift_bsdev = 0;
        if (num_bootstrap_iter>1) for (int bootstrap_iter = 0; bootstrap_iter < num_bootstrap_iter; ++bootstrap_iter) {
            // resample the distances vectors
            resample(distances_along_axis_1, *daa1);
            resample(distances_along_axis_2, *daa2);
            double bsdiff = 0;
            if (use_median) {
                sort(daa1->begin(), daa1->end());
                sort(daa2->begin(), daa2->end());
                avgd1 = median(&(*daa1)[0], daa1->size());
                avgd2 = median(&(*daa2)[0], daa2->size());
                devd1 = interquartile(&(*daa1)[0], daa1->size());
                devd2 = interquartile(&(*daa2)[0], daa2->size());
                int nsamples = min(np1*np2,np_prod_max);
                vector<double> samples(nsamples);
                for (int sidx=0; sidx<nsamples; ++sidx) {
                    int i1, i2;
                    if (nsamples==np_prod_max) {
                        i1 = randint(np1);
                        i2 = randint(np2);
                    } else {
                        i1 = sidx / np2;
                        i2 = sidx - i1 * np2;
                    }
                    double s1 = (*daa1)[i1];
                    double s2 = (*daa2)[i2];
                    Point sn = ((*daa2)[i2] * normal_2 - (*daa1)[i1] * normal_1);
                    double d = sn.norm();
                    if (sn.dot(deltaref)<0) d *= -1;
                    samples[sidx] = d;
                }
                sort(samples.begin(), samples.end());
                bsdiff = median(&samples[0], nsamples);
            } else {
                mean_dev(&(*daa1)[0], daa1->size(), avgd1, devd1);
                mean_dev(&(*daa2)[0], daa2->size(), avgd2, devd2);
                if (same_normal) bsdiff = avgd2 - avgd1;
                else {
                    int nsamples = min(np1*np2,np_prod_max);
                    bsdiff = 0.;
                    for (int sidx=0; sidx<nsamples; ++sidx) {
                        int i1, i2;
                        if (nsamples==np_prod_max) {
                            i1 = randint(np1);
                            i2 = randint(np2);
                        } else {
                            i1 = sidx / np2;
                            i2 = sidx - i1 * np2;
                        }
                        double s1 = (*daa1)[i1];
                        double s2 = (*daa2)[i2];
                        Point sn = ((*daa2)[i2] * normal_2 - (*daa1)[i1] * normal_1);
                        double d = sn.norm();
                        if (sn.dot(deltaref)<0) d *= -1;
                        bsdiff += d;
                    }
                    bsdiff /= nsamples;
                }
            }
            
            diff += bsdiff;
            diff_bsdev += bsdiff*bsdiff;
            c1shift += avgd1;
            c1shift_bsdev += avgd1 * avgd1; // for bootstrap distribution dev
            c1dev += devd1;
            c2shift += avgd2;
            c2shift_bsdev += avgd2 * avgd2;
            c2dev += devd2;
            
            if (bsdiff < sample_diff) ++z0_sum;
            
            if (bs_dist) (*bs_dist)[bootstrap_iter] = bsdiff;
        }
        
        // finish bootstrap stats
        if (num_bootstrap_iter>1) {
            c1shift /= num_bootstrap_iter;
            c1dev /= num_bootstrap_iter;
            c2shift /= num_bootstrap_iter;
            c2dev /= num_bootstrap_iter;
            diff /= num_bootstrap_iter;
            delete daa1; delete daa2;
            diff_bsdev = sqrt( (diff_bsdev - diff*diff*num_bootstrap_iter)/(num_bootstrap_iter-1.0) );
            c1shift_bsdev = sqrt( (c1shift_bsdev - c1shift*c1shift*num_bootstrap_iter)/(num_bootstrap_iter-1.0) );
            c2shift_bsdev = sqrt( (c2shift_bsdev - c2shift*c2shift*num_bootstrap_iter)/(num_bootstrap_iter-1.0) );
        }
        else {
            diff_bsdev = 0;
            c1shift_bsdev = 0;
            c2shift_bsdev = 0;
        }

        // finish the computation of confidence intervals if needed
        if (fast_ci) {
            ci_high = z_high * (sqrt(c1dev*c1dev/np1 + c2dev*c2dev/np2) + pos_dev);
            ci_low = -ci_high;
        }
        else if (use_BCa) {
            sort(bs_dist->begin(), bs_dist->end());
            if (use_median) {
                int idxlow = max(0, min((int)floor(((1.-confidence_interval_percent*0.01) * 0.5) * num_bootstrap_iter), num_bootstrap_iter-1));
                int idxhigh = max(0, min((int)floor((1.-(1.-confidence_interval_percent*0.01) * 0.5) * num_bootstrap_iter), num_bootstrap_iter-1));
                ci_low = (*bs_dist)[idxlow];
                ci_high = (*bs_dist)[idxhigh];
            } else {
                double z0 = inverse_normal_cdf(z0_sum / (double)num_bootstrap_iter);
                double alow = normal_cumulative(z0+(z0+z_low)/(1.-BC_acceleration_factor*(z0+z_low)));
                double ahigh = normal_cumulative(z0+(z0+z_high)/(1.-BC_acceleration_factor*(z0+z_high)));
                int idxlow = max(0, min((int)floor(alow * num_bootstrap_iter), num_bootstrap_iter-1));
                int idxhigh = max(0, min((int)floor(ahigh * num_bootstrap_iter), num_bootstrap_iter-1));
                ci_low = (*bs_dist)[idxlow];
                ci_high = (*bs_dist)[idxhigh];
            }
        }
        
        int diff_sig = (np1>=num_pt_sig) && (np2>=num_pt_sig) && ((diff<ci_low) || (diff>ci_high));
        
        Point core1 = isfinite(c1shift) ? corepoints[ptidx] + c1shift * normal_1 : corepoints[ptidx];
        Point core2 = isfinite(c2shift) ? corepoints[ptidx] + c2shift * normal_2 : corepoints[ptidx];
        
        if (!isfinite(c1shift)) ++num_nan_c1;
        if (!isfinite(c2shift)) ++num_nan_c2;

        for (int i=0; i<(int)resultfiles.size(); ++i) {
            ofstream& resultfile = *resultfiles[i];
            vector<string>& formats = result_formats[i];
            for (int j=0; j<(int)formats.size(); ++j) {
                if (j>0) resultfile << " ";
                if (formats[j] == "c1") resultfile << nan_is_0(core1.x) << " " << nan_is_0(core1.y) << " " << nan_is_0(core1.z);
                else if (formats[j] == "c2") resultfile << nan_is_0(core2.x) << " " << nan_is_0(core2.y) << " " << nan_is_0(core2.z);
                else if (formats[j] == "c0") resultfile << nan_is_0(corepoints[ptidx].x) << " " << nan_is_0(corepoints[ptidx].y) << " " << nan_is_0(corepoints[ptidx].z);
                else if (formats[j] == "n1") resultfile << nan_is_0(normal_1.x) << " " << nan_is_0(normal_1.y) << " " << nan_is_0(normal_1.z);
                else if (formats[j] == "n2") resultfile << nan_is_0(normal_2.x) << " " << nan_is_0(normal_2.y) << " " << nan_is_0(normal_2.z);
                else if (formats[j] == "sn1") resultfile << nan_is_0(scalesvec[normal_scale_idx_1]);
                else if (formats[j] == "sn2") resultfile << nan_is_0(scalesvec[normal_scale_idx_2]);
                else if (formats[j] == "ns1") resultfile << nan_is_0(neigh_num_1[normal_scale_idx_1]);
                else if (formats[j] == "ns2") resultfile << nan_is_0(neigh_num_2[normal_scale_idx_2]);
                else if (formats[j] == "np1") resultfile << nan_is_0(np1);
                else if (formats[j] == "np2") resultfile << nan_is_0(np2);
                else if (formats[j] == "shift1") resultfile << nan_is_0(c1shift);
                else if (formats[j] == "shift2") resultfile << nan_is_0(c2shift);
                else if (formats[j] == "dev1") resultfile << nan_is_0(c1dev);
                else if (formats[j] == "dev2") resultfile << nan_is_0(c2dev);
                else if (formats[j] == "diff") resultfile << nan_is_0(diff);
                else if (formats[j] == "diff_bsdev") resultfile << nan_is_0(diff_bsdev);
                else if (formats[j] == "shift1_bsdev") resultfile << nan_is_0(c1shift_bsdev);
                else if (formats[j] == "shift2_bsdev") resultfile << nan_is_0(c2shift_bsdev);
                else if (formats[j] == "normal_dev1") resultfile << nan_is_0(normal_dev1);
                else if (formats[j] == "normal_dev2") resultfile << nan_is_0(normal_dev2);
                else if (formats[j] == "ksi1") resultfile << nan_is_0((scalesvec[normal_scale_idx_1] / normal_dev1));
                else if (formats[j] == "ksi2") resultfile << nan_is_0((scalesvec[normal_scale_idx_2] / normal_dev2));
                else if (formats[j] == "n1angle_bs") resultfile << nan_is_0(n1angle_bs);
                else if (formats[j] == "n2angle_bs") resultfile << nan_is_0(n2angle_bs);
                else if (formats[j] == "diff_ci_low") resultfile << nan_is_0(ci_low);
                else if (formats[j] == "diff_ci_high") resultfile << nan_is_0(ci_high);
                else if (formats[j] == "diff_sig") resultfile << nan_is_0(diff_sig);
                else {
                    if (ptidx==0) cout << "Invalid result format \"" << formats[j] << "\" is ignored." << endl;
                }
            }        
            resultfile << endl;
        }
        
        if (isfinite(diff)) core_global_diff_mean += diff;
        else ++num_nan_diff;
        core_global_diff_min = min((double)core_global_diff_min, (double)diff);
        core_global_diff_max = max((double)core_global_diff_min, (double)diff);
    }
    cout << endl;
    
    cout << num_nan_c1 << " / " << corepoints.size() << " core points could not be projected on the first cloud" << endl;
    cout << num_nan_c2 << " / " << corepoints.size() << " core points could not be projected on the second cloud" << endl;
    core_global_diff_mean /= corepoints.size() - num_nan_diff;
    cout << "Global diff min / mean / max on all core points: " << core_global_diff_min << " / " << core_global_diff_mean << " / " << core_global_diff_max << endl;

    for (int i=0; i<(int)resultfiles.size(); ++i) resultfiles[i]->close();
        
    return 0;
}
