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
#include <limits>
#include <fstream>
#include <map>

#include "classifier.hpp"

using namespace std;
using namespace boost;

int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
Displays the content of a parameters file. Usage:\n\
prm_info features.prm\n\
  input: features.prm         # Classifier parameters computed by validate_classifier\n\
"<<endl;
        return 0;
}

struct ClassifInfo {
    bool reliable;
    int classif;
    FloatType confidence;
    ClassifInfo() : reliable(false), classif(-1), confidence(0.5) {}
};
typedef PointTemplate<ClassifInfo> PointClassif;

int main(int argc, char** argv) {

    if (argc<2) return help();

    ifstream classifparamsfile(argv[1], ifstream::binary);
    int nscales;
    classifparamsfile.read((char*)&nscales, sizeof(int));
    if (nscales>10000) {
        cout << "File is probably corrupted (it claims to contain more than 10000 scales)" << endl;
        return 1;
    }
    int fdim = nscales*2;
    vector<FloatType> scales(nscales);
    cout << "Number of scales: " << nscales << endl;
    cout << "Scales:";
    for (int s=0; s<nscales; ++s) {
        classifparamsfile.read((char*)&scales[s], sizeof(FloatType));
        cout << " " << scales[s];
    }
    cout << endl;
    int nclassifiers; // number of 2-class classifiers
    classifparamsfile.read((char*)&nclassifiers, sizeof(int));
    cout << "Number of classifiers: " << nclassifiers << endl;
    vector<Classifier> classifiers(nclassifiers);
    for (int ci=0; ci<nclassifiers; ++ci) {
        classifparamsfile.read((char*)&classifiers[ci].class1, sizeof(int));
        classifparamsfile.read((char*)&classifiers[ci].class2, sizeof(int));
        classifiers[ci].weights_axis1.resize(fdim+1);
        classifiers[ci].weights_axis2.resize(fdim+1);
        for (int i=0; i<=fdim; ++i) classifparamsfile.read((char*)&classifiers[ci].weights_axis1[i],sizeof(FloatType));
        for (int i=0; i<=fdim; ++i) classifparamsfile.read((char*)&classifiers[ci].weights_axis2[i],sizeof(FloatType));
        int pathsize;
        classifparamsfile.read((char*)&pathsize,sizeof(int));
        classifiers[ci].path.resize(pathsize);
        cout << "Classifier " << (ci+1) << " handles classes " << classifiers[ci].class1 << " and " << classifiers[ci].class2 << endl;
        for (int i=0; i<pathsize; ++i) {
            classifparamsfile.read((char*)&classifiers[ci].path[i].x,sizeof(FloatType));
            classifparamsfile.read((char*)&classifiers[ci].path[i].y,sizeof(FloatType));
        }
        classifparamsfile.read((char*)&classifiers[ci].refpt_pos.x,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].refpt_pos.y,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].refpt_neg.x,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].refpt_neg.y,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].absmaxXY,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].axis_scale_ratio,sizeof(FloatType));
        classifiers[ci].prepare();
    }
    classifparamsfile.close();

    return 0;
}
