#include <iostream>
#include <fstream>
#include <sstream>

#include "points.hpp"

using namespace std;

CNearTree<Point, FloatType, &point_dist> data;

int main(int argc, char** argv) {

    if (argc<3) {
        cout << "Arguments required:   xzy_ascii_data_file   binary_tree_output_file" << endl;
        return 0;
    }

    ifstream datafile(argv[1]);
    string line;
    int npts = 0;
    while (datafile && !datafile.eof()) {
        getline(datafile, line);
        if (line.empty()) continue;
        stringstream linereader(line);
        Point point;
        FloatType value;
        int i = 0;
        while (linereader >> value) {
            point[i] = value;
            if (++i==3) break;
        }
        if (i!=3) {
            cout << "invalid data file" << endl;
            return 1;
        }
        data.Insert(point);
        ++npts;
    }
    datafile.close();

    ofstream bintree(argv[2], ofstream::binary);
    data.save(bintree);
    bintree.close();

    return 0;
}
