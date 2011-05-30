#include <iostream>
#include <limits>
#include <fstream>

#include "points.hpp"
#include "helpers.hpp"
#include "classifier.hpp"

using namespace std;

int help(const char* errmsg = 0) {
cout << "\
msc_tool   cmd file.msc ( [file.prm [out.svg [classifnum]]] | file_out.xyz )\n\
  input: cmd            # A command to execute on the msc file:\n\
                        # \"info\": display information of the given msc file and quit\n\
                        # \"project\": project the given msc file on the parameter space provided by the given prm file. Write the result in out.svg.\n\
                        # \"xyz\": convert the given msc file to text format in file_out.xyz\n\
  input: file.msc       # the multiscale file to consider\n\
  input: file.prm       # (info, project) if given, displays information about the parameter file as well\n\
  output: out.svg       # (project) the msc file in the classifier parameter space\n\
                        # and produce a density visualisation of the points contained\n\
                        # in the msc file\n\
  input: classifnum     # the classifier number to use for multi-classifier parameter\n\
                        # files. Optional in case a single classifier is present.\n\
  output: file_out.xyz  # (xyz) convert the msc file to a text format containing\n\
                        # the position of the core points and the associated multiscale\n\
                        # values as extra columns\n\
"<<endl;
    if (errmsg) cout << "Error: " << errmsg << endl;
        return 0;
}

int main(int argc, char** argv) {
    if (argc<3) return help();
        
    bool cmd_info = !strcmp(argv[1],"info");
    bool cmd_project = !strcmp(argv[1],"project");
    bool cmd_xyz = !strcmp(argv[1],"xyz");
    if (!cmd_info && !cmd_project && !cmd_xyz) return help();

    vector<FloatType> scales;
    
    MSCFile mscfile(argv[2]);
    int ptnparams;
    int npts = read_msc_header(mscfile, scales, ptnparams);
    int nscales = scales.size();
    int fdim = nscales * 2;
    
    cout << "Multiscale file contains:" << endl;
    cout << "  " << npts << " data points" << endl;
    cout << "  " << nscales << " scales:";
    for (int si=0; si<nscales; ++si) cout << " " << scales[si];
    cout << endl;
    cout << "  " << (ptnparams-3) << " additional fields from original core points" << endl;

    if (argc==3) {
        if (cmd_info) return 0;
        if (cmd_project) return help("Need a parameter file defining the projection.");
        if (cmd_xyz) return help("Need a xyz file name to write to.");
    }

    if (cmd_xyz) {
        // TODO
        return help("not implemented yet");
    }
    
    ifstream classifparamsfile(argv[3], ifstream::binary);
    int prm_nscales;
    classifparamsfile.read((char*)&prm_nscales, sizeof(int));
    int prm_fdim = prm_nscales*2;
    vector<FloatType> prm_scales(prm_nscales);
    for (int s=0; s<prm_nscales; ++s) classifparamsfile.read((char*)&prm_scales[s], sizeof(FloatType));
    int nclassifiers; // number of 2-class classifiers
    classifparamsfile.read((char*)&nclassifiers, sizeof(int));
    vector<Classifier> classifiers(nclassifiers);
    for (int ci=0; ci<nclassifiers; ++ci) {
        classifparamsfile.read((char*)&classifiers[ci].class1, sizeof(int));
        classifparamsfile.read((char*)&classifiers[ci].class2, sizeof(int));
        classifiers[ci].weights_axis1.resize(prm_fdim+1);
        classifiers[ci].weights_axis2.resize(prm_fdim+1);
        for (int i=0; i<=prm_fdim; ++i) classifparamsfile.read((char*)&classifiers[ci].weights_axis1[i],sizeof(FloatType));
        for (int i=0; i<=prm_fdim; ++i) classifparamsfile.read((char*)&classifiers[ci].weights_axis2[i],sizeof(FloatType));
        int pathsize;
        classifparamsfile.read((char*)&pathsize,sizeof(int));
        classifiers[ci].path.resize(pathsize);
        for (int i=0; i<pathsize; ++i) {
            classifparamsfile.read((char*)&classifiers[ci].path[i].x,sizeof(FloatType));
            classifparamsfile.read((char*)&classifiers[ci].path[i].y,sizeof(FloatType));
        }
        classifparamsfile.read((char*)&classifiers[ci].refpt_pos.x,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].refpt_pos.y,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].refpt_neg.x,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].refpt_neg.y,sizeof(FloatType));
        classifparamsfile.read((char*)&classifiers[ci].absmaxXY,sizeof(FloatType));
        classifiers[ci].prepare();
    }
    classifparamsfile.close();

    cout << "Classifier file contains:" << endl;
    cout << "  " << nclassifiers << " binary classifiers for classes:";
    for (int ci=0; ci<nclassifiers; ++ci) cout << " (" << classifiers[ci].class1 << "," << classifiers[ci].class2 << ")";
    cout << endl;
    cout << "  " << prm_nscales << " scales:";
    for (int si=0; si<prm_nscales; ++si) cout << " " << prm_scales[si];
    cout << endl;

    if (argc==4) {
        if (cmd_project) return help("Need a svg file name to write to.");
        return help("internal error, shall not happen");
    }
    if (cmd_info) return 0; // ignore next arguments

    // necessarily project command at this point.
    int classifier_to_use = 0;
    if (nclassifiers>1) {
        if (argc==5) return help("Need a classifier number to use in case of multi-classifier parameter files.");
        classifier_to_use = atoi(argv[5]);
        if (classifier_to_use<1 || classifier_to_use>nclassifiers) return help("Invalid classifier number");
        --classifier_to_use; // from number to index in array
        cout << "Using classifier " << (classifier_to_use+1) << " for classes (" << classifiers[classifier_to_use].class1 << "," << classifiers[classifier_to_use].class2 << ")";
    }
    Classifier& classifier = classifiers[classifier_to_use];

    cout << "Reading msc data..." << endl;
    vector<FloatType> data(npts * fdim);
    read_msc_data(mscfile,nscales,npts,&data[0],ptnparams);

    cout << "Projecting msc data in the classifier plane..." << endl;
    
    static const int svgSize = 800;
    
    // dispatch every msc data over the nearest pixels using a Gaussian
    // with dev=sqrt(2)/2 to be somehow compatible with suggest_classifier
    // => variance = 0.5
    // exp(-2 * || pixel_location - msc_data_proj || ^2)
    // clip at 3 pixels distance = exp(-18)... already quite small
    // note : no need to normalise the exp, we'll normalise the whole grid
    // at the end for the color scale anyway!
    vector<double> density_grid(svgSize * svgSize, 0.0);
    
    
    cout << "Writing the svg file..." << endl;
    ofstream svgfile(argv[4]);
    
    static const int halfSvgSize = svgSize / 2;
    
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, svgSize, svgSize);
    cairo_t *cr = cairo_create(surface);
    
    return 0;
}

