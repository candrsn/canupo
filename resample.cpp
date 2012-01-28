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
#include <algorithm>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>

#include <boost/format.hpp>

#include "points.hpp"
using namespace std;
using namespace boost;

int help(const char* errmsg = 0) {
cout << "\
resample cloud.xyz result.xyz flag parameter [opt_param]\n\
  input: cloud.xyz          # raw point cloud to process\n\
  output: result.xyz        # resulting resampled cloud\n\
  input: flag               # Either \"r\" or \"s\" for a random or a spatial\n\
                            # supsampling, or both for subsampling first at random\n\
                            # (faster load time) then spatially.\n\
  input: parameter          # subsampling factor (retain one every X points on average)\n\
                            # for the random mode and when both \"r\" and \"s\"\n\
                            # are specified, or minimum distance between\n\
                            # two points for the spatial mode.\n\
  input: opt_param          # minimum distance between two points when both\n\
                            # \"r\" and \"s\" are specified.\n\
"<<endl;
    if (errmsg) cout << "Error: " << errmsg << endl;
    return 0;
}

struct Index {
    int idx;
};
typedef PointTemplate<Index> PointIdx;

int main(int argc, char** argv) {

    if (argc<5) return help();

    bool random_mode = false, spatial_mode = false;
    if (strstr(argv[3],"r")!=0) random_mode = true;
    if (strstr(argv[3],"s")!=0) spatial_mode = true;
    if (!random_mode && !spatial_mode) return help("invalid flag");

    int subsampling_factor = 0;
    FloatType minimum_distance = 0;
    int spatial_argi = 4;
    if (random_mode) {
        subsampling_factor = atoi(argv[4]);
        if (subsampling_factor<=0) return help("invalid subsampling factor");
        spatial_argi = 5;
    }
    if (spatial_mode) {
        if (argc<=spatial_argi) return help();
        minimum_distance = atof(argv[spatial_argi]);
        if (minimum_distance<=0) return help("invalid minimum distance");
    }

    ofstream resultfile(argv[2]);

    boost::mt19937 rng;

    if (random_mode && !spatial_mode) {
        // random mode: retain lines at random when loading the cloud
        cout << "Sampling data cloud: " << argv[1] << " into " << argv[2] << endl;
        FILE* datafile = fopen(argv[1], "r");
        if (!datafile) {std::cerr << "Could not load file: " << argv[1] << std::endl; return 1;}
        char* line = 0; size_t linelen = 0; int num_read = 0;
        long numretained = 0;
        long nlines = 0;
        while ((num_read = getline(&line, &linelen, datafile)) != -1) {
            ++nlines;
            if (linelen==0 || line[0]=='#') continue;
            if (rng() % subsampling_factor >0) continue;
            ++numretained;
            resultfile << line;
        }
        fclose(datafile);
        cout << "Retained " << numretained << " lines out of " << nlines << " (effective sampling ratio 1/" << (double)nlines / (double)numretained << ")" << endl;
        // done!
        return 0;
    }

    // spatial subsampling mode - harder
    // reload with parsing and spatial indexation
    PointCloud<PointIdx> cloud;
    vector<size_t> *line_numbers = new vector<size_t>();
    cout << "Loading data cloud: " << argv[1] << endl;
    size_t original_number_of_lines = cloud.load_txt(argv[1],0,line_numbers,subsampling_factor);
    // make the link data set and line numbers
    for (int i=0; i<cloud.data.size(); ++i) cloud.data[i].idx = (*line_numbers)[i];
    // free redundant memory
    delete line_numbers;

    // algo: select a point at random.
    //       output the corresponding line
    //       remove all neighbors in the given radius (including itself)
    //       until no point remains
    cout << "Processing spatial resampling." << endl;
    cout << "Percent complete: 0" << flush;
    int nextpercentcomplete = 5;
    vector<size_t> retained_lines;
    size_t initial_data_size = cloud.data.size();
    while (!cloud.data.empty()) {
        int percentcomplete = (initial_data_size - cloud.data.size()) * 100 / initial_data_size;
        if (percentcomplete>=nextpercentcomplete) {
            if (percentcomplete>=nextpercentcomplete) {
                nextpercentcomplete+=5;
                if (percentcomplete % 10 == 0) cout << percentcomplete << flush;
                else if (percentcomplete % 5 == 0) cout << "." << flush;
            }
        }

        boost::uniform_int<int> int_dist(0, cloud.data.size()-1);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > randint(rng, int_dist);
        size_t selected_point = randint();
        // WARNING: selected_point is the index of the data in the cloud.data vector
        // the line number is given by the idx field of the data
        retained_lines.push_back(cloud.data[selected_point].idx);
        // now remove the neighbors. The indices of the data points in the data vector
        // may change. This does not matter for the indices in the lines as these
        // are saved within the points themselves. However removal shall be done
        // from highest index to lowest index as elements are swapped with the last
        // entry in the data vector during a removel (lower indices are unchanged)
        vector<DistPoint<PointIdx> > neighbors;
        cloud.findNeighbors(back_inserter(neighbors), cloud.data[selected_point], minimum_distance);
        vector<int> data_indices(neighbors.size());
        for (int i=0; i<neighbors.size(); ++i) data_indices[i] = neighbors[i].pt - &cloud.data[0];
        sort(data_indices.begin(), data_indices.end(), greater<int>());
        for (int i=0; i<neighbors.size(); ++i) cloud.remove(data_indices[i]);
    }
    if (nextpercentcomplete==100) cout << 100;
    cout << endl;

    cout << "Sorting indices to retain" << endl;
    sort(retained_lines.begin(), retained_lines.end());

    cout << "Writing output file" << endl;
    FILE* datafile = fopen(argv[1], "r");
    if (!datafile) {std::cerr << "Could not load file: " << argv[1] << std::endl; return 1;}
    char* line = 0; size_t linelen = 0; int num_read = 0;
    size_t linenum = 0, retained_idx = 0;
    while ((num_read = getline(&line, &linelen, datafile)) != -1) {
        ++linenum;
        if (linelen==0 || line[0]=='#') continue;
        if (linenum==retained_lines[retained_idx]) {
            resultfile << line;
            ++retained_idx;
        }
    }
    fclose(datafile);

    cout << "Retained " << retained_idx << " out of " << original_number_of_lines << " initial data points (subsampling ratio = 1/" << (double)original_number_of_lines/(double)retained_idx << ")" << endl;

    return 0;
}






