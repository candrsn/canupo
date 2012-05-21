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
#include <vector>

#include <string.h>
#include <stdlib.h>

#include "points.hpp"

using namespace std;

int help(const char* msg = 0, int ret = 0) {
    if (msg) cout << msg << endl;
    cout << "Arguments: input_file output_file constraint1 [constraint2 [...] ]" << endl;
    cout << "Files are text files with one variable per space-separated column" << endl;
    cout << "Constraints are written var_num:min:max" << endl;
    cout << "This says that the variable whose number is given (counting from one) must be in the given range."<< endl;
    cout << "Constraints are combined with AND, a line must respect all constraints to be valid" << endl;
    cout << "The output file is written with only the valid lines from the input file" << endl;
    return ret;
}

struct Constraint {
    int varnum;
    double minval, maxval;
};

int main(int argc, char** argv) {

    if (argc<4) return help();

    vector<Constraint> constraints;

    FILE* input_file = fopen(argv[1], "r");
    ofstream output_file(argv[2]);

    for (int argi=3; argi<argc; ++argi) {
        char* col1 = strchr(argv[argi],':');
        char* col2 = strrchr(argv[argi],':');
        if (col1==0 || col2==0 || col1==col2) {
            return help("Invalid constraint specification", 1);
        } else {
            Constraint c;
            *col1++=0;
            c.varnum = atoi(argv[argi]) - 1; // convert to index
            *col2++=0;
            c.minval = atof(col1);
            c.maxval = atof(col2);
            if (c.varnum<0) return help("Invalid variable number", 2);
            if (c.minval>c.maxval) return help("Invalid range", 3);
            constraints.push_back(c);
        }
    }
    
    int linenum = 0;
    int nvalues = -1;
    char* line = 0; size_t linelen = 0; int num_read = 0;
    while ((num_read = getline(&line, &linelen, input_file)) != -1) {
        ++linenum;
        if (linelen==0 || line[0]=='#') continue;
        vector<double> values;
        for (char* x = line; *x!=0;) values.push_back(fast_atof_next_token(x));
        bool ok = true;
        for (unsigned int i=0; i<constraints.size(); ++i) {
            if (constraints[i].varnum>=values.size()) {ok = false; break;}
            if (values[constraints[i].varnum] < constraints[i].minval) {ok = false; break;}
            if (values[constraints[i].varnum] > constraints[i].maxval) {ok = false; break;}
        }
        if (ok) output_file << line;
    }
    
    return 0;
}
