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

#define FLOAT_TYPE double
#include "points.hpp"

using namespace std;

int help(const char* errmsg = 0) {
cout << "\
set_to_core cloud.xyz core.txt result.txt\n\
  input: cloud.xyz      # Cloud point, containing x,y,z coordinates as the\n\
                        # first fields on each line. One point per line.\n\
  input: core.xyz       # Cloud of \"core\" points, containing x,y,z\n\
                        # coordinates as the first fields on each line,\n\
                        # and an arbitrary number of extra numeric fields.\n\
  input: result.txt     # Cloud point. Each of the original cloud.xyz line\n\
                        # is copied with the extra fields taken from that\n\
                        # line first, then from the nearest core point.\n\
"<<endl;
    if (errmsg) cout << "Error: " << errmsg << endl;
    return 0;
}

int main(int argc, char** argv) {

    if (argc<4) return help();

    cout << "Loading core points" << endl;
    vector<vector<FloatType> > extrafields;
    PointCloud<Point> coreCloud;
    coreCloud.load_txt(argv[2], &extrafields);
    
    FILE* input_file = fopen(argv[1], "r");
    ofstream output_file(argv[3]);
    cout << "Processing file" << endl;
    
    int linenum = 0;
    int nvalues = -1;
    char* line = 0; size_t linelen = 0; int num_read = 0;
    while ((num_read = getline(&line, &linelen, input_file)) != -1) {
        ++linenum;
        if (linelen==0) continue;
        if (line[0]=='#') {output_file << line; continue;}
        vector<double> values;
        for (char* x = line; *x!=0;) values.push_back(fast_atof_next_token(x));
        if (values.size()<3) {
            cout << "Line " << linenum << " does not have valid xyz coordinates" << endl;
            continue;
        }
        int neighidx = coreCloud.findNearest(Point(values[0], values[1], values[2]));
        if (neighidx==-1) {
            cout << "Line " << linenum << ": no nearest core point, ligne ignored." << endl;
            return 1;
        }
        output_file << values[0] << " " << values[1] << " " << values[2];
        for (int i=3; i<values.size(); ++i) output_file << " " << values[i];
        for (auto x : extrafields[neighidx]) output_file << " " << x;
        output_file << endl;
    }
    return 0;
}
