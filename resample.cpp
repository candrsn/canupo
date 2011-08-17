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
resample cloud.xyz result.xyz flag parameter]\n\
  input: cloud.xyz          # raw point cloud to process\n\
  output: result.xyz        # resulting resampled cloud\n\
  input: flag               # Either \"r\" or \"s\" for a random or a spatial\n\
                            # supsampling\n\
  input: parameter          # subsampling factor (retain one every X points)\n\
                            # for the random mode, or minimum distance between\n\
                            # two points for the spatial mode\n\
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

    bool random_mode = false;
    if (strcmp(argv[3],"r")==0) random_mode = true;
    else if (strcmp(argv[3],"s")!=0) return help("invalid flag");
    
    int subsampling_factor = 0;
    FloatType minimum_distance = 0;
    if (random_mode) {
        subsampling_factor = atoi(argv[4]);
        if (subsampling_factor<=0) return help("invalid subsampling factor");
    }
    else {
        minimum_distance = atof(argv[4]);
        if (minimum_distance<=0) return help("invalid minimum distance");
    }

    ofstream resultfile(argv[2]);
    
    boost::mt19937 rng;
        
    if (random_mode) {

        // just load all data lines in a big string array
        // for later writing exactly the same entries
        cout << "Loading data cloud: " << argv[1] << endl;
        ifstream datafile(argv[1]);
        vector<string> lines;
        string line;
        while (datafile && !datafile.eof()) {
            getline(datafile, line);
            if (line.empty() || starts_with(line,"#") || starts_with(line,";") || starts_with(line,"!") || starts_with(line,"//")) continue;
            lines.push_back(line);
        }
        datafile.close();

        // randomly output some of the lines now
        cout << "Percent complete: 0" << flush;
        int final_size = lines.size() / subsampling_factor;
        int nextpercentcomplete = 5;
        for (int i=0; i<final_size; ++i) {
            int percentcomplete = ((i+1) * 100) / final_size;
            if (percentcomplete>=nextpercentcomplete) {
                if (percentcomplete>=nextpercentcomplete) {
                    nextpercentcomplete+=5;
                    if (percentcomplete % 10 == 0) cout << percentcomplete << flush;
                    else if (percentcomplete % 5 == 0) cout << "." << flush;
                }
            }

            boost::uniform_int<int> int_dist(0, lines.size()-1);
            boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > randint(rng, int_dist);
            int selected_point = randint();
            resultfile << lines[selected_point] << endl;
            lines[selected_point] = lines.back();
            lines.pop_back();
        }
        cout << endl;
        // done!
        return 0;
    }

    // spatial subsampling mode - harder
    // reload with parsing and spatial indexation
    vector<string> lines;
    PointCloud<PointIdx> cloud;
    cout << "Loading data cloud: " << argv[1] << endl;
    cloud.load_txt(argv[1],0,&lines);
    // make the link between both data sets
    for (int i=0; i<cloud.data.size(); ++i) cloud.data[i].idx = i;
    
    // algo: select a point at random.
    //       output the corresponding line
    //       remove all neighbors in the given radius (including itself)
    //       until no point remains
    cout << "Processing spatial resampling." << endl;
    cout << "Percent complete: 0" << flush;
    int nextpercentcomplete = 5;
    int num_retained = 0;
    while (!cloud.data.empty()) {
        int percentcomplete = (lines.size() - cloud.data.size()) * 100 / lines.size();
        if (percentcomplete>=nextpercentcomplete) {
            if (percentcomplete>=nextpercentcomplete) {
                nextpercentcomplete+=5;
                if (percentcomplete % 10 == 0) cout << percentcomplete << flush;
                else if (percentcomplete % 5 == 0) cout << "." << flush;
            }
        }
        
        boost::uniform_int<int> int_dist(0, cloud.data.size()-1);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > randint(rng, int_dist);
        int selected_point = randint();
        // WARNING: selected_point is the index of the data in the cloud.data vector
        // the index in the lines vector is given by the idx field of the data
        resultfile << lines[cloud.data[selected_point].idx] << endl;
        vector<DistPoint<PointIdx> > neighbors;
        cloud.findNeighbors(back_inserter(neighbors), cloud.data[selected_point], minimum_distance);
        // now remove the neighbors. The indices of the data points in the data vector
        // may change. This does not matter for the indices in the lines as these
        // are saved within the points themselves. However removal shall be done
        // from highest index to lowest index as elements are swapped with the last
        // entry in the data vector during a removel (lower indices are unchanged)
        vector<int> data_indices(neighbors.size());
        for (int i=0; i<neighbors.size(); ++i) data_indices[i] = neighbors[i].pt - &cloud.data[0];
        sort(data_indices.begin(), data_indices.end(), greater<int>());
        for (int i=0; i<neighbors.size(); ++i) cloud.remove(data_indices[i]);
        ++num_retained;
    }
    if (nextpercentcomplete==100) cout << 100;
    cout << endl;
    cout << "Retained " << num_retained << " out of " << lines.size() << " initial data points (subsampling ratio = 1/" << (double)lines.size()/(double)num_retained << ")" << endl;

    return 0;
}






