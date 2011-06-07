#include <iostream>
#include <limits>
#include <fstream>

#include <cairo/cairo.h>

#include "points.hpp"
#include "base64.hpp"
#include "helpers.hpp"
#include "classifier.hpp"

using namespace std;

int help(const char* errmsg = 0) {
cout << "\
msc_tool   cmd file1.msc [file2.msc ...] ( : file.prm [out.svg [kernel_dev [classifnum]]] | file_out.xyz [...] )\n\
  input: cmd            # A command to execute on the msc file:\n\
                        # \"info\": display information of the given msc files and quit\n\
                        # \"project\": project the given msc files on the parameter space provided by the given prm file. Write the result in out.svg.\n\
                        # \"xyz\": convert the given msc files to text format in file_out.xyz\n\
  input: file.msc       # the multiscale file to consider\n\
  input: file.prm       # (info, project) if given, displays information about the parameter file as well\n\
  output: out.svg       # (project) the msc file in the classifier parameter space\n\
                        # and produce a density visualisation of the points contained\n\
                        # in the msc file\n\
  input: kernel_dev     # (project) the standard deviation of the gaussian kernel used\n\
                        # for computing the density, in pixels. Default is 0 = do not use a kernel.\n\
  input: classifnum     # the classifier number to use for multi-classifier parameter\n\
                        # files. Optional in case a single classifier is present.\n\
  output: file_out.xyz  # (xyz) convert the msc file to a text format containing\n\
                        # the position of the core points and the associated multiscale\n\
                        # values as extra columns\n\
"<<endl;
    if (errmsg) cout << "Error: " << errmsg << endl;
        return 0;
}

void hueToRGB(FloatType hue, int& r, int& g, int& b) {
    hue = 6.0f * (hue - floorf(hue)); // 0 <= hue < 1
    if (hue < 1.0f) {
        r=255; b=0;
        g = (int)(255.999 * hue);
    }
    else if (hue < 2.0f) {
        g=255; b=0;
        r = (int)(255.999 * (2.0f-hue));
    }
    else if (hue < 3.0f) {
        g=255; r=0;
        b = (int)(255.999 * (hue-2.0f));
    }
    else if (hue < 4.0f) {
        b=255; r=0;
        g = (int)(255.999 * (4.0f-hue));
    }
    else if (hue < 5.0f) {
        b=255; g=0;
        r = (int)(255.999 * (hue-4.0f));
    }
    else {
        r=255; g=0;
        b = (int)(255.999 * (6.0f-hue));
    }
}

int ppmwrite(cairo_surface_t *surface, const char* filename) {
    int height = cairo_image_surface_get_height(surface);
    int width = cairo_image_surface_get_width(surface);
    int stride = cairo_image_surface_get_stride(surface);
    unsigned char* data = cairo_image_surface_get_data(surface);
    ofstream ppmfile(filename);
    ppmfile << "P3 " << width << " " << height << " " << 255 << endl;
    for (int row = 0; row<height; ++row) {
        for (int col = 0; col<width*4; col+=4) {
            ppmfile << (int)data[col+2] << " " << (int)data[col+1] << " " << (int)data[col+0] << " ";
        }
        data += stride;
    }
}

cairo_status_t png_copier(void *closure, const unsigned char *data, unsigned int length) {
    std::vector<char>* pngdata = (std::vector<char>*)closure;
    int cursize = pngdata->size();
    pngdata->resize(cursize + length); // use reserve() before, or this will be slow
    memcpy(&(*pngdata)[cursize], data, length);
    return CAIRO_STATUS_SUCCESS;
}

int main(int argc, char** argv) {
    if (argc<3) return help();
        
    bool cmd_info = !strcmp(argv[1],"info");
    bool cmd_project = !strcmp(argv[1],"project");
    bool cmd_xyz = !strcmp(argv[1],"xyz");
    if (!cmd_info && !cmd_project && !cmd_xyz) return help();
    
    int arg_separator = -1;
    for (int argi = 2; argi<argc; ++argi) if (!strcmp(argv[argi],":")) {
        arg_separator = argi;
        break;
    }
    if (arg_separator<2 && !cmd_info) return help();

    bool inconsistent = false;

    vector<FloatType> scales;
    int ptnparams;
    int npts = 0;
    for (int argi=2; argi<(cmd_info?(arg_separator>0?arg_separator:argc):arg_separator); ++argi) {
        vector<FloatType> scales_thisfile;
        MSCFile mscfile(argv[argi]);
        int npts_thisfile = read_msc_header(mscfile, scales_thisfile, ptnparams);
        npts += npts_thisfile;
        
        cout << "Multiscale file "<< argv[argi] << " contains:" << endl;
        cout << "  " << npts << " data points" << endl;
        cout << "  " << scales_thisfile.size() << " scales:";
        for (int si=0; si<scales_thisfile.size(); ++si) cout << " " << scales_thisfile[si];
        cout << endl;
        cout << "  " << (ptnparams-3) << " additional fields from original core points" << endl;

        if (scales_thisfile.empty()) inconsistent = true;
        if (scales.empty()) scales = scales_thisfile;
        else if (scales.size() != scales_thisfile.size()) inconsistent = true;
        else for (int si=0; si<scales.size(); ++si) if (!fpeq(scales[si],scales_thisfile[si])) inconsistent = true;
    }

    if (arg_separator==-1 || arg_separator==argc-1) {
        if (cmd_info) return 0; // done
        if (cmd_project) return help("Need a parameter file defining the projection.");
        if (cmd_xyz) return help("Need a xyz file name to write to.");
    }
    
    int nscales = scales.size();
    int fdim = nscales * 2;

    if (cmd_xyz) {
        int nmscfiles = arg_separator - 2;
        if (nmscfiles != argc-arg_separator-1) return help("Need the same number of output files as there are input msc files.");
        for (int msci = 0; msci < nmscfiles; ++msci) {
            ifstream mscfile(argv[2+msci], ifstream::binary);
            ofstream xyzfile(argv[arg_separator+1+msci]);
            // read the file header
            int ncorepoints;
            mscfile.read((char*)&ncorepoints,sizeof(ncorepoints));
            int nscales;
            mscfile.read((char*)&nscales, sizeof(int));
            for (int si=0; si<nscales; ++si) {
                FloatType scale_msc;
                mscfile.read((char*)&scale_msc, sizeof(FloatType));
            }
            int ptnparams;
            mscfile.read((char*)&ptnparams, sizeof(int));
            if (ptnparams<3) {
                cerr << "Multiscale file does not contain point coordinates!" << endl;
                return 1;
            }
            for (int pt=0; pt<ncorepoints; ++pt) {
                FloatType x,y,z;
                mscfile.read((char*)&x, sizeof(FloatType));
                mscfile.read((char*)&y, sizeof(FloatType));
                mscfile.read((char*)&z, sizeof(FloatType));
                xyzfile << x << " " << y << " " << z;
                for (int i=3; i<ptnparams; ++i) {
                    FloatType param;
                    mscfile.read((char*)&param, sizeof(FloatType));
                    xyzfile << " " << param;
                }
                for (int s=0; s<nscales; ++s) {
                    FloatType a,b;
                    mscfile.read((char*)(&a), sizeof(FloatType));
                    mscfile.read((char*)(&b), sizeof(FloatType));
                    FloatType c = 1 - a - b;
                    xyzfile << " " << a << " " << b << " " << c;
                }
                int nneigh;
                for (int i=0; i<nscales; ++i) {
                    mscfile.read((char*)&nneigh, sizeof(int));
                    xyzfile << " " << nneigh;
                }
                xyzfile << endl;
            }
            mscfile.close();
            xyzfile.close();
        }
        return 0;
    }
    
    ifstream classifparamsfile(argv[arg_separator+1], ifstream::binary);
    int prm_nscales;
    classifparamsfile.read((char*)&prm_nscales, sizeof(int));
    int prm_fdim = prm_nscales*2;
    vector<FloatType> prm_scales(prm_nscales);
    for (int s=0; s<prm_nscales; ++s) classifparamsfile.read((char*)&prm_scales[s], sizeof(FloatType));
    if (scales.empty()) scales = prm_scales;
    else if (scales.size() != prm_scales.size()) inconsistent = true;
    else for (int si=0; si<scales.size(); ++si) if (!fpeq(scales[si],prm_scales[si])) inconsistent = true;
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

    if (argc<=arg_separator+2) {
        if (cmd_project) return help("Need a svg file name to write to.");
        return help("internal error, shall not happen");
    }
    if (cmd_info) return 0; // ignore next arguments

    if (inconsistent) {
        cout << "Combining multiscale and parameter files with different scales is not supported for now." << endl;
        return 0;
    }
        
    // necessarily project command at this point.
    int classifier_to_use = 0;
    if (nclassifiers>1) {
        if (argc<arg_separator+5) return help("Need a classifier number to use in case of multi-classifier parameter files.");
        classifier_to_use = atoi(argv[arg_separator+4]);
        if (classifier_to_use<1 || classifier_to_use>nclassifiers) return help("Invalid classifier number");
        --classifier_to_use; // from number to index in array
        cout << "Using classifier " << (classifier_to_use+1) << " for classes (" << classifiers[classifier_to_use].class1 << "," << classifiers[classifier_to_use].class2 << ")";
    }
    Classifier& classifier = classifiers[classifier_to_use];

    cout << "Reading msc data..." << endl;
    vector<FloatType> data(npts * fdim);
    npts = 0;
    for (int argi=2; argi<arg_separator; ++argi) {
        vector<FloatType> scales_thisfile;
        MSCFile mscfile(argv[argi]);
        int npts_thisfile = read_msc_header(mscfile, scales_thisfile, ptnparams);
        read_msc_data(mscfile,nscales,npts_thisfile,&data[npts*fdim],ptnparams,true);
        npts += npts_thisfile;
    }

    cout << "Projecting msc data in the classifier plane..." << endl;
    static const int svgSize = 800;
    static const int halfSvgSize = svgSize / 2;
    
    double kernel_dev = 0;
    if (argc>arg_separator+3) kernel_dev = atof(argv[arg_separator+3]);
    double kernel_var = kernel_dev * kernel_dev;
    
    // dispatch every msc data over the nearest pixels using a Gaussian
    // with dev=sqrt(2)/2 by default to be somehow compatible with suggest_classifier
    // exp(||pixel_location - msc_data_proj||^2 / kernel_dev^2)
    // clip at 3*kernel_dev pixels distance already quite small
    // note : no need to normalise the exp, we'll normalise the whole grid
    // at the end for the color scale anyway!
    vector<double> density_grid(svgSize * svgSize, 0.0);
    FloatType max_distance = kernel_dev * 3;
    FloatType scaleFactor = halfSvgSize / classifier.absmaxXY;
    
    for (int pi=0; pi<npts; ++pi) {
        FloatType a, b;
        
        // TODO: match classifier scales with msc scales and allow files with
        // different scales, so long as all necessary scales are present
        classifier.project(&data[pi*fdim], a, b);
        FloatType x = a * scaleFactor + halfSvgSize;
        FloatType y = halfSvgSize - b * scaleFactor;
        if (kernel_dev>0) {
            for (int j = (int)floor(y-max_distance); j<=(int)floor(y+max_distance); ++j) {
                if (j<0 || j>=svgSize) continue;
                for (int i = (int)floor(x-max_distance); i<=(int)floor(x+max_distance); ++i) {
                    if (i<0 || i>=svgSize) continue;
                    double dx = i+0.5 - x;
                    double dy = j+0.5 - y;
                    density_grid[j * svgSize + i] += exp( - (dx*dx+dy*dy) / kernel_var );
                }
            }
        }
        else {
            int i = (int)floor(x); int j = (int)floor(y); 
            if (i<0 || i>=svgSize) continue;
            if (j<0 || j>=svgSize) continue;
            ++density_grid[j * svgSize + i];
        }
    }
    
    double min_density = npts; // all points in the same pixel
    double max_density = 0;
    for (int j=0; j<svgSize; ++j) for (int i=0; i<svgSize; ++i) {
        double density = density_grid[j*svgSize+i];
        if (density < min_density) min_density = density;
        if (density > max_density) max_density = density;
    }
    if (max_density <= min_density) max_density = min_density +1;
    
    cout << "Writing the svg file..." << endl;
    ofstream svgfile(argv[arg_separator+2]);
    
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, svgSize, svgSize);
    cairo_t *cr = cairo_create(surface);
    int surface_height = cairo_image_surface_get_height(surface);
    int surface_width = cairo_image_surface_get_width(surface);
    int surface_stride = cairo_image_surface_get_stride(surface);
    unsigned char* surface_data = cairo_image_surface_get_data(surface);
    for (int j=0; j<svgSize; ++j) {
        for (int i=0; i<svgSize; ++i) {
            //double density_01 = (density_grid[j*svgSize+i] - min_density) / (max_density - min_density);
            double density_01 = (log(density_grid[j*svgSize+i]+1) - log(min_density+1)) / (log(max_density+1) - log(min_density+1));
            int r,g,b;
            // low value = blue(hue=4/6), high = red(hue=0)
            hueToRGB(4./6. * (1.0 - density_01),r,g,b);
            surface_data[i*4+2] = r;
            surface_data[i*4+1] = g;
            surface_data[i*4+0] = b;
        }
        surface_data += surface_stride;
    }

    cairo_set_source_rgb(cr, 0.25,0.25,0.25);
    cairo_select_font_face (cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size (cr, 12);
    cairo_text_extents_t extents;
    FloatType dprob = -log(1.0/0.99 - 1.0) * scaleFactor;
    const char* text = "p(classif)>99%";
    cairo_text_extents(cr, text, &extents);
    cairo_move_to(cr, svgSize - dprob - 20 - extents.width - extents.x_bearing, svgSize - 15 - extents.height/2 - extents.y_bearing);
    cairo_show_text(cr, text);
    cairo_move_to(cr, svgSize - dprob - 10, svgSize - 15);
    cairo_line_to(cr, svgSize - 10, svgSize - 15);
    dprob = -log(1.0/0.95 - 1.0) * scaleFactor;
    text = "p(classif)>95%";
    cairo_text_extents(cr, text, &extents);
    cairo_move_to(cr, svgSize - dprob - 20 - extents.width - extents.x_bearing, svgSize - 35 - extents.height/2 - extents.y_bearing);
    cairo_show_text(cr, text);
    cairo_move_to(cr, svgSize - dprob - 10, svgSize - 35);
    cairo_line_to(cr, svgSize - 10, svgSize - 35);
    dprob = -log(1.0/0.9 - 1.0) * scaleFactor;
    text = "p(classif)>90%";
    cairo_text_extents(cr, text, &extents);
    cairo_move_to(cr, svgSize - dprob - 20 - extents.width - extents.x_bearing, svgSize - 55 - extents.height/2 - extents.y_bearing);
    cairo_show_text(cr, text);
    cairo_move_to(cr, svgSize - dprob - 10, svgSize - 55);
    cairo_line_to(cr, svgSize - 10, svgSize - 55);
    cairo_stroke(cr);

    // draw lines on top of points
    double dashes[2]; 
    dashes[0] = dashes[1] = svgSize*0.01;
    cairo_set_dash(cr, dashes, 2, svgSize*0.005);
    cairo_set_source_rgb(cr, 0.25,0.25,0.25);
    cairo_move_to(cr, 0,halfSvgSize);
    cairo_line_to(cr, svgSize,halfSvgSize);
    cairo_move_to(cr, halfSvgSize,0);
    cairo_line_to(cr, halfSvgSize,svgSize);
    cairo_stroke(cr);

    svgfile << "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\""<< svgSize << "\" height=\""<< svgSize <<"\" >" << endl;
    vector<char> binary_parameters(
        sizeof(int)
      + nscales*sizeof(FloatType)
      + (fdim+1)*sizeof(FloatType)
      + (fdim+1)*sizeof(FloatType)
      + sizeof(FloatType)
      + sizeof(FloatType)
      + sizeof(int)
    );
    int bpidx = 0;
    memcpy(&binary_parameters[bpidx],&prm_nscales,sizeof(int)); bpidx += sizeof(int);
    for (int i=0; i<prm_nscales; ++i) {
        memcpy(&binary_parameters[bpidx],&prm_scales[i],sizeof(FloatType));
        bpidx += sizeof(FloatType);
    }
    // Projections on the two 2D axis
    for (int i=0; i<=fdim; ++i) {
        memcpy(&binary_parameters[bpidx],&classifier.weights_axis1[i],sizeof(FloatType));
        bpidx += sizeof(FloatType);
    }
    for (int i=0; i<=fdim; ++i) {
        memcpy(&binary_parameters[bpidx],&classifier.weights_axis2[i],sizeof(FloatType));
        bpidx += sizeof(FloatType);
    }
    // boundaries
    memcpy(&binary_parameters[bpidx],&classifier.absmaxXY,sizeof(FloatType)); bpidx += sizeof(FloatType);
    // conversion from svg to 2D space
    memcpy(&binary_parameters[bpidx],&scaleFactor,sizeof(FloatType)); bpidx += sizeof(FloatType);
    memcpy(&binary_parameters[bpidx],&halfSvgSize,sizeof(int)); bpidx += sizeof(int);

    base64 codec;
    int nbytes;
    
    std::vector<char> base64commentdata(codec.get_max_encoded_size(binary_parameters.size()));
    nbytes = codec.encode(&binary_parameters[0], binary_parameters.size(), &base64commentdata[0]);
    nbytes += codec.encode_end(&base64commentdata[nbytes]);
    
    // comments work well and do not introduce any artifact in the resulting SVG
    // but sometimes they are not preserved... use a hidden text then as workaround
#ifdef CANUPO_NO_SVG_COMMENT
    svgfile << "<text style=\"font-size:1px;fill:#ffffff;fill-opacity:0;stroke:none\" x=\"20\" y=\"20\">params=" << &base64commentdata[0] << "</text>" << endl;
#else
    svgfile << "<!-- params " << &base64commentdata[0] << " -->" << endl;
#endif

#ifdef CANUPO_NO_PNG
    string filename = argv[arg_separator+2];
    filename.replace(filename.size()-3,3,"ppm");
    ppmwrite(surface,filename.c_str());
    svgfile << "<image xlink:href=\""<< filename << "\" width=\""<<svgSize<<"\" height=\""<<svgSize<<"\" x=\"0\" y=\"0\" style=\"z-index:0\" />" << endl;
#else
    //cairo_surface_write_to_png (surface, argv[arg_shift+1]);
    std::vector<char> pngdata;
    pngdata.reserve(800*800*3); // need only large enough init size
    cairo_surface_write_to_png_stream(surface, png_copier, &pngdata);

    // encode the png data into base64
    std::vector<char> base64pngdata(codec.get_max_encoded_size(pngdata.size()));
    codec.reset_encoder();
    nbytes = codec.encode(&pngdata[0], pngdata.size(), &base64pngdata[0]);
    nbytes += codec.encode_end(&base64pngdata[nbytes]);
    
    // include the image inline    
    svgfile << "<image xlink:href=\"data:image/png;base64,"<< &base64pngdata[0]
            << "\" width=\""<<svgSize<<"\" height=\""<<svgSize<<"\" x=\"0\" y=\"0\" style=\"z-index:0\" />" << endl;
#endif
    
    // include the reference points
    svgfile << "<circle cx=\""<< (classifier.refpt_pos.x*scaleFactor+halfSvgSize) <<"\" cy=\""<< (halfSvgSize-classifier.refpt_pos.y*scaleFactor) <<"\" r=\"2\" style=\"fill:none;stroke:#000000;stroke-width:1px;z-index:1;\" />" << endl;
    svgfile << "<circle cx=\""<< (classifier.refpt_neg.x*scaleFactor+halfSvgSize) <<"\" cy=\""<< (halfSvgSize-classifier.refpt_neg.y*scaleFactor) <<"\" r=\"2\" style=\"fill:none;stroke:#000000;stroke-width:1px;z-index:1;\" />" << endl;

    // plot decision boundary as a path
    svgfile << "<path style=\"fill:none;stroke:#000000;stroke-width:1px;z-index:1;\" d=\"";
    for(int i=0; i<classifier.path.size(); ++i) {
        // convert the path back to SVG space
        FloatType px = classifier.path[i].x * scaleFactor + halfSvgSize;
        FloatType py = halfSvgSize - classifier.path[i].y * scaleFactor;
        if (i==0) svgfile << "M "; else svgfile << " L ";
        svgfile << px << "," << py;
    }
    svgfile << "\" />" << endl;

    svgfile << "</svg>" << endl;
    svgfile.close();

    cairo_surface_destroy(surface);
    cairo_destroy(cr);
    
    return 0;
}

