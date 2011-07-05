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

#include <boost/format.hpp>

using namespace std;
using namespace boost;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
Computes a slice of the point cloud at given locations (core points) and in given directions (normals)\n\
\n\
Usage: tranche_berge  data_core_normals.xyz data.xyz hmin hmax lmin lmax E dx output_prefix type [type...] \n\
  input: data_core_normals.xyz\n\
                   # As given by the \"normals\" program\n\
  input: data.xyz  # Full point cloud, with xyz then possibly extra scalars on each line\n\
  input: hmin      # max distance below the core point for the slice\n\
  input: hmax      # max distance above the core point for the slice\n\
  input: lmin      # max distance along the normal in the negative direction from the core point\n\
  input: lmax      # max distance along the normal in the positive direction from the core point\n\
  input: E         # width of the slice\n\
  input: dx        # profile granularity\n\
  output: output_prefix   # Results will be generated for each core point as output_prefixNNN_type.txt where NNN is the number of the core point in order of appearance in the original file.\n\
  input: type      #  type=\"raw\": all points in the slice, as: x y z scalar1 xi yi zi with xi yi zi the local coordinates, x y z and scalars the original data. The first line contains the core point with scalar1=99.\n\
                   # type=\"slice\": all points in the slice, as: xi yi zi x y z scalars... and in a separate file \"base\" the coordinate of the core point and then the local x, y, z vectors, on 4 lines.\n\
                   # type=\"profile\": the profile containing on each line some statistics on a slab of size (dx,E,hmin+hmax) along the local X axis: xp xavg yavg zavg zmin zmax zdev count xg yg zg scalar_avg... where: xp is the center position of the slab along the local X (so the slab ranges xp-dx/2 to xp+dx/2), (xavg,yavg,zavg) is the real average position of the points in the slab (local coordinates), zmin/zmax/zdev are statistics on z in the slab (local coordinates), count is the number of points in the slab, (xg,yg,zg) are the global coordinates of the point with local coordinates (xp,0,zavg). Averaged scalars if any are then given on the remaining of the line\n\
                   # type=\"all\" computes all types. \n\
more types can be given, all corresponding files will be generated (saves computation time)  \n\
" << endl;
    return 0;
}


int main(int argc, char** argv) {

    if (argc<11) return help();
    
    string core_normals_file_name = argv[1];
    string data_file_name = argv[2];
    double hmin = atof(argv[3]);
    double hmax = atof(argv[4]);
    double lmin = atof(argv[5]);
    double lmax = atof(argv[6]);
    double E = atof(argv[7]);
    double dx = atof(argv[8]);
    string output_prefix = argv[9];

    bool compute_raw = false;
    bool compute_slice = false;
    bool compute_prof = false;
    bool compute_sec = false;
    for (int argi=10; argi<argc; ++argi) {
        if (strcasecmp(argv[argi],"raw")==0 || strcasecmp(argv[argi],"all")==0) compute_raw = true;
        if (strcasecmp(argv[argi],"slice")==0 || strcasecmp(argv[argi],"all")==0) compute_slice = true;
        if (strcasecmp(argv[argi],"profile")==0 || strcasecmp(argv[argi],"all")==0) compute_prof = true;
        if (strcasecmp(argv[argi],"sec")==0 || strcasecmp(argv[argi],"all")==0) compute_sec = true;
    }
    
    cout << "Loading core points" << endl;
    ifstream corepointsfile(core_normals_file_name.c_str());
    string line;
    int linenum = 0;
    while (corepointsfile && !corepointsfile.eof()) {
        getline(corepointsfile, line);
        trim(line);
        if (line.empty() || starts_with(line,"#") || starts_with(line,";") || starts_with(line,"!") || starts_with(line,"//")) continue;
        ++linenum;
    }
    corepointsfile.close();
    
    vector<Point> corepoints(linenum);
    vector<Point> normals(linenum);
    
    corepointsfile.open(core_normals_file_name.c_str());
    linenum = 0;
    while (corepointsfile && !corepointsfile.eof()) {
        getline(corepointsfile, line);
        trim(line);
        if (line.empty() || starts_with(line,"#") || starts_with(line,";") || starts_with(line,"!") || starts_with(line,"//")) continue;
        stringstream linereader(line);
        FloatType value;
        int i = 0;
        while (linereader >> value) {
            if (i<3) corepoints[linenum][i] = value;
            else if (i<6) normals[linenum][i-3] = value;
            if (++i==6) break;
        }
        ++linenum;
    }
    
    cout << "Loading data points" << endl;
    vector<vector<FloatType> > scalar_fields;
    PointCloud<Point> cloud;
    cloud.load_txt(data_file_name, &scalar_fields);

    int nscalar = -1; bool inconsistent_scalar = false;
    for (int i=0; i<scalar_fields.size(); ++i) {
        if (nscalar==-1) {
            nscalar = scalar_fields[i].size();
            continue;
        }
        if (nscalar!=scalar_fields[i].size()) inconsistent_scalar = true;
        if (scalar_fields[i].size()<nscalar) nscalar = scalar_fields[i].size();
    }
    if (inconsistent_scalar) cout << "Warning: Inconsistent number of scalar fields. Using minimum number of scalars found on a single line = " << nscalar << endl;
    if (nscalar==-1) nscalar = 0;
    cout << "number of scalars : " << nscalar << endl;
    
    // Compute the sphere radius englobing the slice: max in all directions
    double half_E = 0.5 * E;
    double englobing_radius = lmin*lmin+hmin*hmin;
    double d2_diag = lmin*lmin+hmax*hmax;
    if (d2_diag > englobing_radius) englobing_radius = d2_diag;
    d2_diag = lmax*lmax+hmax*hmax;
    if (d2_diag > englobing_radius) englobing_radius = d2_diag;
    d2_diag = lmax*lmax+hmin*hmin;
    if (d2_diag > englobing_radius) englobing_radius = d2_diag;
    // distance to the farthest slice corner in 3D
    englobing_radius = sqrt(englobing_radius + half_E * half_E);
    
    cout << "Computing the slices"<< endl;
    cout << "Percent complete: 0" << flush;
    int nextpercentcomplete = 5;
    for (int corenum = 0; corenum < corepoints.size(); ++corenum) {
        int percentcomplete = ((corenum+1) * 100) / corepoints.size();
        if (percentcomplete>=nextpercentcomplete) {
            if (percentcomplete>=nextpercentcomplete) {
                nextpercentcomplete+=5;
                if (percentcomplete % 10 == 0) cout << percentcomplete << flush;
                else if (percentcomplete % 5 == 0) cout << "." << flush;
            }
        }

        // local basis: z remains z upward
        Point lzvec(0,0,1);
        // local x alongside the normal, horizontal
        Point lxvec(normals[corenum].x, normals[corenum].y, 0);
        lxvec.normalize();
        // local y as cross-product
        Point lyvec = lzvec.cross(lxvec);

        int nprof = 0; 
        if (dx>0) nprof = (int)floor((lmin+lmax) / dx);
        vector<double> profile_xavg(nprof, 0.0);
        vector<double> profile_yavg(nprof, 0.0);
        vector<double> profile_zmin(nprof, 0.0);
        vector<double> profile_zmax(nprof, 0.0);
        vector<double> profile_zavg(nprof, 0.0);
        vector<double> profile_zdev(nprof, 0.0);
        vector<double> profile_count(nprof, 0.0);
        vector<vector<double> > profile_scalar(nprof, vector<double>(nscalar+1, 0.0) );
        
        // prepare the output file
        ofstream raw_ouput_file, slice_ouput_file, base_output_file, prof_output_file;
        if (compute_raw) {
            raw_ouput_file.open(str(boost::format("%s%d_raw.txt") % output_prefix % (corenum+1)).c_str());
            // spec: first line contains the core point with scalar 99
            raw_ouput_file << corepoints[corenum].x << " " << corepoints[corenum].y << " " << corepoints[corenum].z << " 99 0 0 0" << endl;
        }
        if (compute_slice) {
            slice_ouput_file.open(str(boost::format("%s%d_slice.txt") % output_prefix % (corenum+1)).c_str());
            base_output_file.open(str(boost::format("%s%d_base.txt") % output_prefix % (corenum+1)).c_str());
            base_output_file << corepoints[corenum].x << " " << corepoints[corenum].y << " " << corepoints[corenum].z << "" << endl;
            base_output_file << lxvec.x << " " << lxvec.y << " " << lxvec.z << "" << endl;
            base_output_file << lyvec.x << " " << lyvec.y << " " << lyvec.z << "" << endl;
            base_output_file << lzvec.x << " " << lzvec.y << " " << lzvec.z << "" << endl;
            base_output_file.close();
        }
        
        // neighborhood sphere
        vector<DistPoint<Point> > neighbors;
        cloud.findNeighbors(back_inserter(neighbors), corepoints[corenum], englobing_radius);
        // select which are in the slice
        vector<Point> slicepoints;
        for (int ni = 0; ni <neighbors.size(); ++ni) {
            // conversion to the local basis
            Point shifted_neighbor = *neighbors[ni].pt - corepoints[corenum];
            Point local_neighbor(shifted_neighbor.dot(lxvec), shifted_neighbor.dot(lyvec), shifted_neighbor.dot(lzvec));
            
            // retain only these which fall in the slice
            if (local_neighbor.x < -lmin) continue;
            if (local_neighbor.x > lmax) continue;
            if (fabs(local_neighbor.y) > half_E) continue;
            if (local_neighbor.z < -hmin) continue;
            if (local_neighbor.z > hmax) continue;
            
            int neighbor_index = neighbors[ni].pt - &cloud.data[0];
            
            if (compute_raw) {
                int class_num = -1;
                if (!scalar_fields[neighbor_index].empty()) class_num = scalar_fields[neighbor_index][0];
                raw_ouput_file << neighbors[ni].pt->x << " " << neighbors[ni].pt->y << " " << neighbors[ni].pt->z << " " << class_num << " " << local_neighbor.x << " " << local_neighbor.y << " " << local_neighbor.z << endl;
            }
            if (compute_slice) {
                slice_ouput_file << local_neighbor.x << " " << local_neighbor.y << " " << local_neighbor.z << " " << neighbors[ni].pt->x << " " << neighbors[ni].pt->y << " " << neighbors[ni].pt->z;
                for (int si = 0; si < scalar_fields[neighbor_index].size(); ++si) {
                    slice_ouput_file << " " << scalar_fields[neighbor_index][si];
                }
                slice_ouput_file << endl;
            }
            if (compute_prof) {
                int pi = (int)floor((local_neighbor.x + lmin) * nprof / (lmin+lmax));
                if (pi>=0 && pi<nprof) { // shall always be the case
                    profile_xavg[pi] += local_neighbor.x;
                    profile_yavg[pi] += local_neighbor.y;
                    if (local_neighbor.z < profile_zmin[pi]) {profile_zmin[pi] = local_neighbor.z;}
                    if (local_neighbor.z > profile_zmax[pi]) {profile_zmax[pi] = local_neighbor.z;}
                    profile_zavg[pi] += local_neighbor.z;
                    profile_zdev[pi] += local_neighbor.z * local_neighbor.z;
                    ++profile_count[pi];
                    for (int si=0; si<nscalar; ++si) profile_scalar[pi][si] += scalar_fields[neighbor_index][si];
                }
            }
        }
        
        if (compute_prof) {
            ofstream prof_ouput_file(str(boost::format("%s%d_profile.txt") % output_prefix % (corenum+1)).c_str());
            for (int pi = 0; pi < nprof; ++pi) {
                double xp = -lmin + (pi+0.5) * (lmin+lmax) / nprof;
                double count = profile_count[pi];
                if (count<1) count = 1; // in which case the avg are 0, so division is OK
                profile_xavg[pi] /= count;
                profile_yavg[pi] /= count;
                profile_zavg[pi] /= count;
                profile_zdev[pi] = sqrt((profile_zdev[pi] - profile_zavg[pi] * profile_zavg[pi] * count) / (count - 1.0));
                if (count==1) profile_zdev[pi] = 0;
                Point tranche_pos = corepoints[corenum] + xp * lxvec + profile_zavg[pi] * lzvec;
                prof_ouput_file << xp << " " << profile_xavg[pi] << " " << profile_yavg[pi] << " " << profile_zavg[pi] << " " << profile_zmin[pi] << " " << profile_zmax[pi] << " " << profile_zdev[pi] << " " << profile_count[pi] << " " << tranche_pos.x << " " << tranche_pos.y << " " << tranche_pos.z;
                for (int si=0; si<nscalar; ++si) prof_ouput_file << " " << (profile_scalar[pi][si] / count);
                prof_ouput_file << endl;
            }
            prof_ouput_file.close();
        }
    }
    
    return 0;
}
