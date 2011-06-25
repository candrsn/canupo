#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <limits>

// graphics lib
#include <cairo/cairo.h>

#include <strings.h>
#include <math.h>

#include "points.hpp"
#include "base64.hpp"
#include "linearSVM.hpp"

using namespace std;

typedef LinearSVM::sample_type sample_type;

int help(const char* errmsg = 0) {
cout << "\
suggest_classifier [=N] outfile.svg [ unlabeled.msc ...] : class1.msc ... - class2.msc ...\n\
    input: class1.msc ... - class2.msc ...  # the multiscale files for each class, separated by -\n\
    output: outfile.svg                     # a svg file in which to write a classifier definition. This file may be edited graphically (ex: with inkscape) so as to add more points in the path that separates both classes. So long as there is only one path consisting only of line segments it shall be recognised.\n\
    \n\
    input(optional): unlabeled.msc          # additionnal multiscale files for the scene that are not classified. These provide more points for semi-supervised learning. The corresponding points will be displayed in grey in the output file. If these are not given a Linear Discriminant Analysis is performed in the projected space to separate the classes. To perform semi-supervised learning with density estimation even when no additionnal unlabelled data is available simply repeat the class1 and class2 data as unlabeled.\n\
    input(optional): =N                     # Size of the search grid for cross-validating the SVM. By default N=1 is used, with a local search for the best parameters around a default value hopefully adequate most of the time. Use N>1 in order to increase the quality of the training at the expense of a larger computation time.\n\
"<<endl;
    if (errmsg) cout << "Error: " << errmsg << endl;
    return 0;
}

bool fpeq(FloatType a, FloatType b) {
    static const FloatType epsilon = 1e-6;
    if (b==0) return fabs(a)<epsilon;
    FloatType ratio = a/b;
    return ratio>1-epsilon && ratio<1+epsilon;
}

// if vector is empty, fill it
// otherwise check the vectors match
int read_msc_header(ifstream& mscfile, vector<FloatType>& scales, int& ptnparams) {
    int npts;
    mscfile.read((char*)&npts,sizeof(npts));
    if (npts<=0) {
        cerr << "invalid msc file (negative or null number of points)" << endl;
        exit(1);
    }
    
    int nscales_thisfile;
    mscfile.read((char*)&nscales_thisfile, sizeof(nscales_thisfile));
    if (nscales_thisfile<=0) {
        cerr << "invalid msc file (negative or null number of scales)" << endl;
        exit(1);
    }
#ifndef MAX_SCALES_IN_MSC_FILE
#define MAX_SCALES_IN_MSC_FILE 1000000
#endif
    if (nscales_thisfile>MAX_SCALES_IN_MSC_FILE) {
        cerr << "This msc file claims to contain more than " << MAX_SCALES_IN_MSC_FILE << " scales. Aborting, this is probably a mistake. If not, simply recompile with a different value for MAX_SCALES_IN_MSC_FILE." << endl;
        exit(1);
    }
    vector<FloatType> scales_thisfile(nscales_thisfile);
    for (int si=0; si<nscales_thisfile; ++si) mscfile.read((char*)&scales_thisfile[si], sizeof(FloatType));
    
    // all files must be consistant
    if (scales.size() == 0) {
        scales = scales_thisfile;
    } else {
        if (scales.size() != nscales_thisfile) {cerr<<"input file mismatch"<<endl; exit(1);}
        for (int si=0; si<scales.size(); ++si) if (!fpeq(scales[si],scales_thisfile[si])) {cerr<<"input file mismatch"<<endl; exit(1);}
    }
    
    // TODO: check consistency of ptnparams
    mscfile.read((char*)&ptnparams, sizeof(int));

    return npts;
}

void read_msc_data(ifstream& mscfile, int nscales, int npts, sample_type* data, int ptnparams) {
    for (int pt=0; pt<npts; ++pt) {
        // we do not care for the point coordinates and other parameters
        for (int i=0; i<ptnparams; ++i) {
            FloatType param;
            mscfile.read((char*)&param, sizeof(FloatType));
        }
        for (int s=0; s<nscales; ++s) {
            FloatType a,b;
            mscfile.read((char*)(&a), sizeof(FloatType));
            mscfile.read((char*)(&b), sizeof(FloatType));
            FloatType c = 1 - a - b;
            // project in the equilateral triangle a*(0,0) + b*(1,0) + c*(1/2,sqrt3/2)
            // equivalently to the triangle formed by the three components unit vector
            // (1,0,0), (0,1,0) and (0,0,1) when considering a,b,c in 3D
            // so each a,b,c = dimensionality of the data is given equal weight
            // is this necessary ? => not for linear classifiers, but plan ahead...
            FloatType x = b + c / 2;
            FloatType y = c * sqrt(3)/2;
            (*data)(s*2) = x;
            (*data)(s*2+1) = y;
        }
        // we do not care for number of neighbors and average dist between nearest neighbors
        // TODO: take this info into account to weight the samples and improve the classifier
        int fooi;
        for (int i=0; i<nscales; ++i) mscfile.read((char*)&fooi, sizeof(int));
/*        FloatType foof;
        for (int i=0; i<nscales; ++i) mscfile.read((char*)&foof, sizeof(FloatType));*/
        ++data;
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

void GramSchmidt(dlib::matrix<dlib::matrix<FloatType,0,1>,0,1>& basis, dlib::matrix<FloatType,0,1>& newX) {
    using namespace dlib;
    // goal: find a basis so that the given vector is the new X
    // principle: at least one basis vector is not orthogonal with newX (except if newX is null but we suppose this is not the case)
    // => use the max dot product vector, and replace it by newX. this forms a set of
    // linearly independent vectors.
    // then apply the Gram-Schmidt process
    int dim = basis.size();
    double maxabsdp = -1; int selectedCoord = 0;
    for (int i=0; i<dim; ++i) {
        double absdp = fabs(dot(basis(i),newX));
        if (absdp > maxabsdp) {
            absdp = maxabsdp;
            selectedCoord = i;
        }
    }
    // swap basis vectors to use the selected coord as the X vector, then replaced by newX
    basis(selectedCoord) = basis(0);
    basis(0) = newX;
    // Gram-Schmidt process to re-orthonormalise the basis.
    // Thanks Wikipedia for the stabilized version
    for (int j = 0; j < dim; ++j) {
        for (int i = 0; i < j; ++i) {
            basis(j) -= (dot(basis(j),basis(i)) / dot(basis(i),basis(i))) * basis(i);
        }
        basis(j) /= sqrt(dot(basis(j),basis(j)));
    }
}

int dichosearch(const vector<double>& series, double x) {
    int dichofirst = 0;
    int dicholast = series.size();
    int dichomed;
    while (true) {
        dichomed = (dichofirst + dicholast) / 2;
        if (dichomed==dichofirst) break;
        if (x==series[dichomed]) break;
        if (x<series[dichomed]) { dicholast = dichomed; continue;}
        dichofirst = dichomed;
    }
    return dichomed;
}

int main(int argc, char** argv) {
    
    if (argc<5) return help();
    
    int grid_size = 1;
    int arg_shift = 0;
    
    string first_arg = argv[1];
    if (first_arg[0]=='=') {
        grid_size = atoi(first_arg.substr(1).c_str());
        if (grid_size<1) return help();
        ++arg_shift;
    }
    
    ofstream svgfile(argv[arg_shift+1]);
    
    int arg_class1 = argc;
    for (int argi = arg_shift+2; argi<argc; ++argi) if (!strcmp(argv[argi],":")) {
        arg_class1 = argi+1;
        break;
    }
    if (arg_class1>=argc) return help();
    
    int arg_class2 = argc;
    for (int argi = arg_class1+1; argi<argc; ++argi) if (!strcmp(argv[argi],"-")) {
        arg_class2 = argi+1;
        break;
    }
    if (arg_class2>=argc) return help();
    
    sample_type undefsample;
    int ptnparams;

    cout << "Loading unlabeled files" << endl;
    
    // neutral files, if any
    int ndata_unlabeled = 0;
    vector<FloatType> scales;
    for (int argi = arg_shift+2; argi<arg_class1-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        // read the file header
        int npts = read_msc_header(mscfile, scales, ptnparams);
        mscfile.close();
        ndata_unlabeled += npts;
    }
    int nscales = scales.size();
    int fdim = nscales * 2;
    if (nscales) undefsample.set_size(fdim,1);
    // fill data
    vector<sample_type> data_unlabeled(ndata_unlabeled, undefsample);
    int base_pt = 0;
    for (int argi = arg_shift+2; argi<arg_class1-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        // read the file header (again)
        int npts = read_msc_header(mscfile, scales, ptnparams);
        // read data
        read_msc_data(mscfile,nscales,npts,&data_unlabeled[base_pt], ptnparams);
        mscfile.close();
        base_pt += npts;
    }
    
    cout << "Loading class files" << endl;
    
    // class1 files
    int ndata_class1 = 0;
    for (int argi = arg_class1; argi<arg_class2-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales, ptnparams);
        mscfile.close();
        ndata_class1 += npts;
    }
    // class2 files
    int ndata_class2 = 0;
    for (int argi = arg_class2; argi<argc; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales, ptnparams);
        mscfile.close();
        ndata_class2 += npts;
    }
    nscales = scales.size(); // in case there is no unlabeled data
    fdim = nscales * 2;
    undefsample.set_size(fdim,1);
    int nsamples = ndata_class1+ndata_class2;
    vector<sample_type> samples(nsamples, undefsample);
    vector<FloatType> labels(nsamples, 0);
    for (int i=0; i<ndata_class1; ++i) labels[i] = -1;
    for (int i=ndata_class1; i<nsamples; ++i) labels[i] = 1;
    
    base_pt = 0;
    for (int argi = arg_class1; argi<arg_class2-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales, ptnparams);
        read_msc_data(mscfile,nscales,npts,&samples[base_pt], ptnparams);
        mscfile.close();
        base_pt += npts;
    }
    for (int argi = arg_class2; argi<argc; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales, ptnparams);
        read_msc_data(mscfile,nscales,npts,&samples[base_pt], ptnparams);
        mscfile.close();
        base_pt += npts;
    }
    
    cout << "Computing the two best projection directions" << endl;

    LinearSVM classifier(grid_size);

    // shuffle before cross-validating to spread instances of each class
    dlib::randomize_samples(samples, labels);
    FloatType nu = classifier.crossValidate(10, samples, labels);
    cout << "Training" << endl;
    classifier.train(10, nu, samples, labels);

    // get the projections of each sample on the first classifier direction
    vector<FloatType> proj1(nsamples);
    for (int i=0; i<nsamples; ++i) proj1[i] = classifier.predict(samples[i]);
    
    // we now have the first hyperplane and corresponding decision boundary
    // projection onto the orthogonal subspace and repeat SVM to get a 2D plot
    // The procedure is a bit like PCA except we seek the successive directions of maximal
    // separability instead of maximal variance
    
    // perform a real projection with reduced dimension to help the SVM a bit
    dlib::matrix<dlib::matrix<FloatType,0,1>,0,1> basis;
    basis.set_size(fdim);
    for (int i=0; i<fdim; ++i) {
        basis(i).set_size(fdim);
        for (int j=0; j<fdim; ++j) basis(i)(j) = 0;
        basis(i)(i) = 1;
    }
    dlib::matrix<FloatType,0,1> w_vect;
    w_vect.set_size(fdim);
    for (int i=0; i<fdim; ++i) w_vect(i) = classifier.weights[i];
    GramSchmidt(basis,w_vect);
    
    vector<sample_type> samples_reduced(nsamples);
    for (int i=0; i<nsamples; ++i) samples_reduced[i].set_size(fdim-1);
    // project the data onto the hyperplane so as to get the second direction
    for (int si=0; si<nsamples; ++si) {
        for(int i=1; i<fdim; ++i) samples_reduced[si](i-1) = dlib::dot(samples[si], basis(i));
    }

    // already shuffled, and do not change order for the proj1 anyway
    LinearSVM ortho_classifier(grid_size);
    nu = ortho_classifier.crossValidate(10, samples_reduced, labels);
    cout << "Training" << endl;
    ortho_classifier.train(10, nu, samples_reduced, labels);
    
    // convert back the classifier weights into the original space
    ortho_classifier.weights.resize(fdim+1);
    ortho_classifier.weights[fdim] = ortho_classifier.weights[fdim-1];
    for(int i=0; i<fdim; ++i) w_vect(i) = 0;
    for(int i=1; i<fdim; ++i) w_vect += ortho_classifier.weights[i-1] * basis(i);
    for(int i=0; i<fdim; ++i) ortho_classifier.weights[i] = w_vect(i);
    
    vector<FloatType> proj2(nsamples);
    for (int i=0; i<nsamples; ++i) proj2[i] = ortho_classifier.predict(samples[i]);

    // compute the reference points for orienting the classifier boundaries
    // pathological cases are possible where an arbitrary point in the (>0,>0)
    // quadrant is not in the +1 class for example
    // here, just use the mean of the classes
    Point refpt_pos(0,0,0);
    Point refpt_neg(0,0,0);
    for (int i=0; i<nsamples; ++i) {
        if (labels[i]>0) refpt_pos += Point(proj1[i], proj2[i], 1);
        else refpt_neg += Point(proj1[i], proj2[i], 1);
    }
    refpt_pos /= refpt_pos.z;
    refpt_neg /= refpt_neg.z;
    
    FloatType xming = numeric_limits<FloatType>::max();
    FloatType xmaxg = -numeric_limits<FloatType>::max();
    FloatType yming = numeric_limits<FloatType>::max();
    FloatType ymaxg = -numeric_limits<FloatType>::max();
    for (int i=0; i<data_unlabeled.size(); ++i) {
        FloatType x = classifier.predict(data_unlabeled[i]);
        FloatType y = ortho_classifier.predict(data_unlabeled[i]);
        xming = min(xming, x);
        xmaxg = max(xmaxg, x);
        yming = min(yming, y);
        ymaxg = max(ymaxg, y);
    }
    FloatType xminc = numeric_limits<FloatType>::max();
    FloatType xmaxc = -numeric_limits<FloatType>::max();
    FloatType yminc = numeric_limits<FloatType>::max();
    FloatType ymaxc = -numeric_limits<FloatType>::max();
    for (int i=0; i<nsamples; ++i) {
        xminc = min(xminc, proj1[i]);
        xmaxc = max(xmaxc, proj1[i]);
        yminc = min(yminc, proj2[i]);
        ymaxc = max(ymaxc, proj2[i]);
    }
    xming = min(xming, xminc);
    xmaxg = max(xmaxg, xmaxc);
    yming = min(yming, yminc);
    ymaxg = max(ymaxg, ymaxc);

    static const int svgSize = 800;
    static const int halfSvgSize = svgSize / 2;
    FloatType minX = numeric_limits<FloatType>::max();
    FloatType maxX = -minX;
    FloatType minY = minX;
    FloatType maxY = -minX;
    for (int i=0; i<nsamples; ++i) {
        minX = min(minX, proj1[i]);
        maxX = max(maxX, proj1[i]);
        minY = min(minY, proj2[i]);
        maxY = max(maxY, proj2[i]);
    }
    FloatType absmaxXY = fabs(max(max(max(-minX,maxX),-minY),maxY));
    FloatType scaleFactor = halfSvgSize / absmaxXY;

    PointCloud<Point2D> cloud2D;
    cloud2D.prepare(xming,xmaxg,yming,ymaxg,nsamples+data_unlabeled.size());
    for (int i=0; i<data_unlabeled.size(); ++i) cloud2D.insert(Point2D(
        classifier.predict(data_unlabeled[i]),
        ortho_classifier.predict(data_unlabeled[i])
    ));
    for (int i=0; i<nsamples; ++i) {
        cloud2D.insert(Point2D(proj1[i],proj2[i]));
    }
    
    FloatType absxymax = fabs(max(max(max(-xming,xmaxg),-yming),ymaxg));
    int nsearchpointm1 = 25;
    // radius from probabilistic SVM, diameter = 90% chance of correct classif
    FloatType radius = -log(1.0/0.9 - 1.0) / 2;
    
    FloatType wx = 0, wy = 0, wc = 0, minspcx = 0, minspcy = 0;

    if (ndata_unlabeled) {
        int minsumd = numeric_limits<int>::max();
        FloatType minvx = 0, minvy = 0;

        cout << "Finding the line with least density" << flush;
        
        for (int spci = 0; spci <= nsearchpointm1; ++spci) {
            cout << "." << flush;
            
            FloatType spcx = refpt_neg.x + spci * (refpt_pos.x - refpt_neg.x) / nsearchpointm1;
            FloatType spcy = refpt_neg.y + spci * (refpt_pos.y - refpt_neg.y) / nsearchpointm1;
        
            // now we swipe a decision boundary in each direction around the point
            // and look for the lowest overall density along the boundary
            int nsearchdir = 90; // each 2 degree, as we swipe from 0 to 180 (unoriented lines)
            FloatType incr = max(xmaxg-xming, ymaxg-yming) / nsearchpointm1;
            vector<FloatType> sumds(nsearchdir);
    #pragma omp parallel for
            for(int sd = 0; sd < nsearchdir; ++sd) {
                // use the parametric P = P0 + alpha*V formulation of a line
                // unit vector in the direction of the line
                FloatType vx = cos(M_PI * sd / nsearchdir);
                FloatType vy = sin(M_PI * sd / nsearchdir);
                sumds[sd] = 0;
                for(int sp = -nsearchpointm1/2; sp < nsearchpointm1/2; ++sp) {
                    int s = sp * incr;
                    FloatType x = vx * s + spcx;
                    FloatType y = vy * s + spcy;
                    vector<DistPoint<Point2D> > neighbors;
                    cloud2D.findNeighbors(back_inserter(neighbors), Point2D(x,y), radius);
                    sumds[sd] += neighbors.size();
                }
            }
            for(int sd = 0; sd < nsearchdir; ++sd) {
                if (sumds[sd]<minsumd) {
                    minsumd = sumds[sd];
                    minvx = cos(M_PI * sd / nsearchdir);
                    minvy = sin(M_PI * sd / nsearchdir);
                    minspcx = spcx;
                    minspcy = spcy;
                }
            }
        }
        cout << endl;
    
        // so we finally have the decision boundary in this 2D space
        // P = P0 + alpha * V : px-p0x = alpha * vx  and  py-p0y = alpha * vy,
        // alpha = (px-p0x) / vx; // if vx is null see below
        // py-p0y = (px-p0x) * vy / vx
        // py = px * vy/vx + p0y - p0x * vy / vx
        // px * vy/vx - py + p0y - p0x * vy / vx = 0
        // equa: wx * px + wy * py + c = 0
        // with: wx = vy/vx; wy = -1; c = p0y - p0x * vy / vx
        // null vx just reverse roles as vy is then !=0 (v is unit vec)
        if (minvx!=0) { wx = minvy / minvx; wy = -1; wc = minspcy - minspcx * wx;}
        else {wx = -1; wy = minvx / minvy; wc = minspcx - minspcy * wy;}
    } else {
        Point2D c1(0,0), c2(0,0);
        for (int i=0; i<nsamples; ++i) {
            if (labels[i]<0) c1 += Point2D(proj1[i],proj2[i]);
            else c2 += Point2D(proj1[i],proj2[i]);
        }
        c1 /= ndata_class1;
        c2 /= ndata_class2;
        
        Point2D w_vect = c2 - c1;
        w_vect /= w_vect.norm();
        Point2D w_orth(-w_vect.y,w_vect.x);

        double cump_diff_min = 2.0;

        for(int sd = 1; sd < 180; ++sd) {
            FloatType vx = cos(sd * M_PI / 180.0);
            FloatType vy = sin(sd * M_PI / 180.0);
            
            dlib::matrix<double,2,2> basis;
            Point2D base_vec1 = w_vect;
            Point2D base_vec2 = vx * w_vect + vy * w_orth;
            basis(0,0) = base_vec1.x; basis(0,1) = base_vec2.x;
            basis(1,0) = base_vec1.y; basis(1,1) = base_vec2.y;
            basis = inv(basis);
            dlib::matrix<double,2,1> P;
            
            double m1 = 0, m2 = 0;
            vector<double> p1, p2;
            for (int i=0; i<nsamples; ++i) {
                P(0) = proj1[i];
                P(1) = proj2[i];
                P = basis * P;
                double d = P(0); // projection on w_vect along the slanted direction
                if (labels[i]<0) {p1.push_back(d); m1+=d;}
                else {p2.push_back(d); m2+=d;}
            }
            m1 /= ndata_class1;
            m2 /= ndata_class2;
            
            // search for optimal separation
            bool reversed = false;
            double n1 = ndata_class1, n2 = ndata_class2;
            if (m1 > m2) {
                reversed = true;
                swap(m1,m2);
                p1.swap(p2);
            }
            sort(p1.begin(), p1.end());
            sort(p2.begin(), p2.end());
            for (int i=0; i<=100; ++i) {
                double pos = m1 + i * (m2 - m1) / 100.0;
                int idx1 = dichosearch(p1, pos);
                int idx2 = dichosearch(p2, pos);
                double pr1 = idx1 / (double)ndata_class1;
                double pr2 = 1.0 - idx2 / (double)ndata_class2;
                double cump_diff = fabs(pr1 - pr2);
                if (cump_diff < cump_diff_min) {cump_diff_min = cump_diff;
                    double r = (pos - m1) / (m2 - m1);
                    if (reversed) r = 1.0 - r;
                    Point2D center = c1 + r * (c2 - c1);
                    minspcx = center.x;
                    minspcy = center.y;
                    wx = base_vec2.x;
                    wy = base_vec2.y;
                    // wx * cx + wy * cy + wc = 0
                    wc = -wx * center.x - wy * center.y;
                }
            }
            
        }
    }

    cout << "Drawing image" << endl;

    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, svgSize, svgSize);
    cairo_t *cr = cairo_create(surface);

    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_set_line_width(cr, 0);
    cairo_rectangle(cr, 0, 0, svgSize, svgSize);
    cairo_fill(cr);
    cairo_stroke(cr);
    
    cairo_set_line_width(cr, 1);
    // cumulating transluscent points to easily get a density estimate
    cairo_set_source_rgba(cr, 0.4, 0.4, 0.4, 0.1);
    // Plot points
    // first the unlabeled data, if any
    for (int i=0; i<ndata_unlabeled; ++i) {
        // we have to project this data as this was not done above
        FloatType x = classifier.predict(data_unlabeled[i]) * scaleFactor + halfSvgSize;
        FloatType y = halfSvgSize - ortho_classifier.predict(data_unlabeled[i]) * scaleFactor;
        cairo_arc(cr, x, y, 0.714, 0, 2*M_PI);
        cairo_stroke(cr);
    }
    // now plot the reference data. It is very well that it was randomised so we do not have one class on top of the other
    for (int i=0; i<nsamples; ++i) {
        FloatType x = proj1[i]*scaleFactor + halfSvgSize;
        FloatType y = halfSvgSize - proj2[i]*scaleFactor;
        if (labels[i]==1) cairo_set_source_rgba(cr, 1, 0, 0, 0.75);
        else cairo_set_source_rgba(cr, 0, 0, 1, 0.75);
        cairo_arc(cr, x, y, 0.714, 0, 2*M_PI);
        cairo_stroke(cr);
    }


/*  // probabilistic circles every 5 % proba of correct classification
    // too much, can't see anything in the middle
    // 1 / (1+exp(-d)) = pval = 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    for (int i=5; i<=45; i+=5) {
        FloatType pval = (50.0 + i) / 100.0;
        // 1+exp(-d) = 1/pval
        // exp(-d) = 1/pval - 1
        FloatType d = -log(1.0/pval - 1.0);  // OK as pval<1
        cairo_arc(cr, halfSvgSize, halfSvgSize, d * scaleFactor, 0, 2*M_PI);
        cairo_stroke(cr);
    }

    // plot the circle at 95% proba of being correct (5% of being wrong)
    FloatType d95 = -log(1.0/0.95 - 1.0);
    cairo_arc(cr, halfSvgSize, halfSvgSize, d95 * scaleFactor, 0, 2*M_PI);
    cairo_stroke(cr);
*/
    // circles are prone to misinterpretation (radius = dist to hyperplane,
    // error not only in the center zone)
    // specify scales at the bottom-right of the image, in a less-used quadrant
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

    cout << "Writing the svg file" << endl;

    // output the svg file
    svgfile << "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\""<< svgSize << "\" height=\""<< svgSize <<"\" >" << endl;
    
    // Save the classifier parameters as an SVG comment so we can find them back later on
    // Use base64 encoded binary to preserve full precision
    
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
    memcpy(&binary_parameters[bpidx],&nscales,sizeof(int)); bpidx += sizeof(int);
    for (int i=0; i<nscales; ++i) {
        memcpy(&binary_parameters[bpidx],&scales[i],sizeof(FloatType));
        bpidx += sizeof(FloatType);
    }
    // Projections on the two 2D axis
    for (int i=0; i<=fdim; ++i) {
        memcpy(&binary_parameters[bpidx],&classifier.weights[i],sizeof(FloatType));
        bpidx += sizeof(FloatType);
    }
    for (int i=0; i<=fdim; ++i) {
        memcpy(&binary_parameters[bpidx],&ortho_classifier.weights[i],sizeof(FloatType));
        bpidx += sizeof(FloatType);
    }
    // boundaries
    memcpy(&binary_parameters[bpidx],&absmaxXY,sizeof(FloatType)); bpidx += sizeof(FloatType);
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
    string filename = argv[arg_shift+1];
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
    svgfile << "<circle cx=\""<< (refpt_pos.x*scaleFactor+halfSvgSize) <<"\" cy=\""<< (halfSvgSize-refpt_pos.y*scaleFactor) <<"\" r=\"2\" style=\"fill:none;stroke:#000000;stroke-width:1px;z-index:1;\" />" << endl;
    svgfile << "<circle cx=\""<< (refpt_neg.x*scaleFactor+halfSvgSize) <<"\" cy=\""<< (halfSvgSize-refpt_neg.y*scaleFactor) <<"\" r=\"2\" style=\"fill:none;stroke:#000000;stroke-width:1px;z-index:1;\" />" << endl;

    // plot decision boundary as a path
    // xy space in plane => scale and then reverse
    // first find homogeneous equa in the 2D space
    // convert the decision boundary to SVG space
    // ori: wx * x + wy * y + wc = 0
    // xsvg = x * scaleFactor + halfSvgSize; => x = (xsvg - halfSvgSize)  / scaleFactor
    // ysvg = halfSvgSize - y * scaleFactor; => y = (halfSvgSize - ysvg)  / scaleFactor
    // wxsvg * xsvg + wysvg * ysvg + csvg = 0
    // wx * x + wy * y + wc = 0
    // wx * (xsvg - halfSvgSize)  / scaleFactor + wy * (halfSvgSize - ysvg)  / scaleFactor + wc = 0
    // wx * (xsvg - halfSvgSize) + wy * (halfSvgSize - ysvg) + wc * scaleFactor = 0
    FloatType wxsvg = wx;
    FloatType wysvg = -wy;
    FloatType csvg = (wy-wx)*halfSvgSize + wc * scaleFactor;
    FloatType minspcxsvg = minspcx * scaleFactor + halfSvgSize;
    FloatType minspcysvg = halfSvgSize - minspcy * scaleFactor;
    // now intersect to find xminsvg, yminsvg, and so on
    // some may be NaN
    FloatType xsvgy0 = -csvg / wxsvg; // at ysvg = 0
    FloatType ysvgx0 = -csvg / wysvg; // at xsvg = 0
    // wxsvg * xsvg + wysvg * ysvg + csvg = 0
    FloatType xsvgymax = (-csvg -wysvg*svgSize) / wxsvg; // at ysvg = svgSize
    FloatType ysvgxmax = (-csvg -wxsvg*svgSize) / wysvg; // at xsvg = svgSize
    // NaN comparisons always fail, so use only positive tests and this is OK
    bool useLeft = (ysvgx0 >= 0) && (ysvgx0 <= svgSize);
    bool useRight = (ysvgxmax >= 0) && (ysvgxmax <= svgSize);
    bool useTop = (xsvgy0 >= 0) && (xsvgy0 <= svgSize);
    bool useBottom = (xsvgymax >= 0) && (xsvgymax <= svgSize);
    int sidescount = useLeft + useRight + useTop + useBottom;
    vector<Point2D> path;
//    if (sidescount==2) {
        svgfile << "<path style=\"fill:none;stroke:#000000;stroke-width:1px;z-index:1;\" d=\"M ";
        if (useLeft) {
            svgfile << 0 << "," << ysvgx0 << " L " << minspcxsvg<<","<<minspcysvg<<" L ";
            if (useTop) svgfile << xsvgy0 << "," << 0 << " ";
            if (useRight) svgfile << svgSize << "," << ysvgxmax << " ";
            if (useBottom) svgfile << xsvgymax << "," << svgSize << " ";
        }
        if (useTop) {
            svgfile << xsvgy0 << "," << 0 << " L " << minspcxsvg<<","<<minspcysvg<<" L ";
            if (useRight) svgfile << svgSize << "," << ysvgxmax << " ";
            if (useBottom) svgfile << xsvgymax << "," << svgSize << " ";
        }
        svgfile << "\" />" << endl;
//    }

    svgfile << "</svg>" << endl;
    svgfile.close();

    cairo_surface_destroy(surface);
    cairo_destroy(cr);

    return 0;
}
