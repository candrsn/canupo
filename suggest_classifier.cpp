#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>

// graphics lib
#include <cairo/cairo.h>

#include <strings.h>
#include <math.h>

#include "points.hpp"
#include "base64.hpp"

#include "dlib/svm.h"

using namespace std;

typedef dlib::matrix<FloatType, 0, 1> sample_type;

struct LinearSVM {

    typedef dlib::linear_kernel<sample_type> kernel_type;

    template <typename sample_type>
    struct cross_validation_objective {
        cross_validation_objective (
            const vector<sample_type>& samples_,
            const vector<FloatType>& labels_,
            int _nfolds
        ) : samples(samples_), labels(labels_), nfolds(_nfolds) {}

        double operator() (FloatType lognu) const {
            using namespace dlib;
            // see below for changes from dlib examples
            const FloatType nu = exp(lognu);

            // Make an SVM trainer and tell it what the parameters are supposed to be.
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_nu(nu);

            // Finally, perform 10-fold cross validation and then print and return the results.
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);

            return sum(result);
        }

        const vector<sample_type>& samples;
        const vector<FloatType>& labels;
        int nfolds;
    };


    FloatType crossValidate(int nfolds, const vector<sample_type>& samples, const vector<FloatType>& labels) {
        using namespace dlib;
        // taken from dlib examples
        
        // largest allowed nu: strictly below what's returned by maximum_nu
        FloatType max_nu = 0.999*maximum_nu(labels);
        FloatType min_nu = 1e-7;

        matrix<FloatType> gridsearchspace = logspace(log10(max_nu), log10(min_nu), 50); // nu parameter
        matrix<FloatType> best_result(2,1);
        best_result = 0;
        FloatType best_nu = 1;
        for (int col = 0; col < gridsearchspace.nc(); ++col) {
            const FloatType nu = gridsearchspace(0, col);
            svm_nu_trainer<kernel_type> trainer;
            trainer.set_kernel(kernel_type());
            trainer.set_nu(nu);
            matrix<FloatType> result = cross_validate_trainer(trainer, samples, labels, nfolds);
            if (sum(result) > sum(best_result)) {
                best_result = result;
                best_nu = nu;
            }
        }
        FloatType lnu = log(best_nu);

//        FloatType lnu = 0.5 * (lmin_nu + lmax_nu);

        FloatType best_score = find_max_single_variable(
            cross_validation_objective<sample_type>(samples, labels, nfolds), // Function to maximize
            lnu,              // starting point and result
            log(min_nu),          // lower bound
            log(max_nu),          // upper bound
            1e-2,
            50
        );
        return exp(lnu);
    }
    
    void train(FloatType nu, const vector<sample_type>& samples, const vector<FloatType>& labels) {
        using namespace dlib;
        svm_nu_trainer<kernel_type> trainer;
        trainer.set_nu(nu);
        trainer.set_kernel(kernel_type());
        int dim = samples.back().size();
        // linear kernel: convert the decision function to an hyperplane rather than support vectors
        // This is equivalent but way more efficient for later scene classification
        decision_function<kernel_type> decfun = trainer.train(samples, labels);
        weights.clear();
        weights.resize(dim+1, 0);
        matrix<FloatType> w(dim,1);
        w = 0;
        for (int i=0; i<decfun.alpha.nr(); ++i) {
            w += decfun.alpha(i) * decfun.basis_vectors(i);
        }
        for (int i=0; i<dim; ++i) weights[i] = w(i);
        weights[dim] = -decfun.b;
    }
    
    FloatType predict(const sample_type& data) {
        int dim = weights.size()-1;
        FloatType ret = weights[dim];
        for (int d=0; d<dim; ++d) ret += weights[d] * data(d);
        return ret;
    }
    
    vector<FloatType> weights;
};




int help(const char* errmsg = 0) {
    if (errmsg) cout << "Error: " << errmsg << endl;
cout << "\
suggest_classifier outfile.svg [ msc(non label) ...] : class1.msc ... - class2.msc ...\n\
"<<endl;
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
int read_msc_header(ifstream& mscfile, vector<FloatType>& scales) {
    int npts;
    mscfile.read((char*)&npts,sizeof(npts));
    if (npts<=0) help("invalid file");
    
    int nscales_thisfile;
    mscfile.read((char*)&nscales_thisfile, sizeof(nscales_thisfile));
    vector<FloatType> scales_thisfile(nscales_thisfile);
    for (int si=0; si<nscales_thisfile; ++si) mscfile.read((char*)&scales_thisfile[si], sizeof(FloatType));
    if (nscales_thisfile<=0) help("invalid file");
    
    // all files must be consistant
    if (scales.size() == 0) {
        scales = scales_thisfile;
    } else {
        if (scales.size() != nscales_thisfile) {cerr<<"input file mismatch: "<<endl; return 1;}
        for (int si=0; si<scales.size(); ++si) if (!fpeq(scales[si],scales_thisfile[si])) {cerr<<"input file mismatch: "<<endl; return 1;}
    }
    return npts;
}

void read_msc_data(ifstream& mscfile, int nscales, int npts, sample_type* data) {
    for (int pt=0; pt<npts; ++pt) {
        FloatType coord; // we do not care for the point coordinates
        mscfile.read((char*)&coord, sizeof(coord));
        mscfile.read((char*)&coord, sizeof(coord));
        mscfile.read((char*)&coord, sizeof(coord));
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
        ++data;
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
    
    if (argc<5) return help();
    
    ofstream svgfile(argv[1]);
    
    int arg_class1 = argc;
    for (int argi = 2; argi<argc; ++argi) if (!strcmp(argv[argi],":")) {
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
    
    // neutral files, if any
    int ndata_unlabeled = 0;
    vector<FloatType> scales;
    for (int argi = 2; argi<arg_class1-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        // read the file header
        int npts = read_msc_header(mscfile, scales);
        mscfile.close();
        ndata_unlabeled += npts;
    }
    int nscales = scales.size();
    int fdim = nscales * 2;
    if (nscales) undefsample.set_size(fdim,1);
    // fill data
    vector<sample_type> data_unlabeled(ndata_unlabeled, undefsample);
    int base_pt = 0;
    for (int argi = 2; argi<arg_class1-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        // read the file header (again)
        int npts = read_msc_header(mscfile, scales);
        // read data
        read_msc_data(mscfile,nscales,npts,&data_unlabeled[base_pt]);
        mscfile.close();
        base_pt += npts;
    }
    
    // class1 files
    int ndata_class1 = 0;
    for (int argi = arg_class1; argi<arg_class2-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales);
        mscfile.close();
        ndata_class1 += npts;
    }
    // class2 files
    int ndata_class2 = 0;
    for (int argi = arg_class2; argi<argc; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales);
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
        int npts = read_msc_header(mscfile, scales);
        read_msc_data(mscfile,nscales,npts,&samples[base_pt]);
        mscfile.close();
        base_pt += npts;
    }
    for (int argi = arg_class2; argi<argc; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        int npts = read_msc_header(mscfile, scales);
        read_msc_data(mscfile,nscales,npts,&samples[base_pt]);
        mscfile.close();
        base_pt += npts;
    }


    LinearSVM classifier;

    // shuffle before cross-validating to spread instances of each class
    dlib::randomize_samples(samples, labels);
    FloatType nu = classifier.crossValidate(10, samples, labels);
    classifier.train(nu, samples, labels);
    
    // get the projections of each sample on the first classifier direction
    vector<FloatType> proj1(nsamples);
    for (int i=0; i<nsamples; ++i) proj1[i] = classifier.predict(samples[i]);
    
    // we now have the first hyperplane and corresponding decision boundary
    // projection onto the orthogonal subspace and repeat SVM to get a 2D plot
    // The procedure is a bit like PCA except we seek the successive directions of maximal
    // separability instead of maximal variance
    
    // projection parameters
    vector<FloatType> nvec(fdim); // normal vector
    FloatType norm = 0;
    for (int i=0; i<fdim; ++i) {
        nvec[i] = classifier.weights[i];
        norm += nvec[i] * nvec[i];
    }
    norm = sqrt(norm);
    for (int i=0; i<fdim; ++i) nvec[i] /= norm;
    vector<FloatType> svec(fdim); // shift vector
    // dot product between normal and shift is given by the bias
    FloatType sndot = -classifier.weights[fdim] / norm;
    for (int i=0; i<fdim; ++i) svec[i] = sndot * nvec[i];
    
    // project the data onto the hyperplane so as to get the second direction
    LinearSVM ortho_classifier;
    for (int si=0; si<nsamples; ++si) {
        FloatType dotprod = 0;
        for(int i=0; i<fdim; ++i) dotprod += nvec[i] * (samples[si](i) - svec[i]);
        for(int i=0; i<fdim; ++i) samples[si](i) -= dotprod * nvec[i];
    }
    // already shuffled, and do not change order for the proj1 anyway
    nu = ortho_classifier.crossValidate(10, samples, labels);
    ortho_classifier.train(nu, samples, labels);

    vector<FloatType> proj2(nsamples);
    for (int i=0; i<nsamples; ++i) proj2[i] = ortho_classifier.predict(samples[i]);

    
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
    
    PointCloud<Point2D> cloud2Dg, cloud2Dc;
    cloud2Dg.prepare(xming,xmaxg,yming,ymaxg,nsamples+data_unlabeled.size());
    cloud2Dc.prepare(xminc,xmaxc,yminc,ymaxc,nsamples);
    for (int i=0; i<data_unlabeled.size(); ++i) cloud2Dg.insert(Point2D(
        classifier.predict(data_unlabeled[i]),
        ortho_classifier.predict(data_unlabeled[i])
    ));
    for (int i=0; i<nsamples; ++i) {
        cloud2Dg.insert(Point2D(proj1[i],proj2[i]));
        cloud2Dc.insert(Point2D(proj1[i],proj2[i]));
    }
    
    // search for lowest-density region of all points along the diagonal
    // between two high density regions of classified data
    // this is a way to handle unlabeled points and partially
    // compensate bad samples, ideally the lowest point is the origin
    FloatType absxymax = fabs(max(max(max(-xming,xmaxg),-yming),ymaxg));
    int nsearchpointm1 = 1000;
    // TODO: radius from probabilistic SVM and region of bad classif proba
    FloatType radius = absxymax / 100;
    // high density region of classified points in each direction from the origin
    FloatType maxdxy1 = 0, maxdxy2 = 0;
    int maxd = 0;
    for(int sp = 0; sp <= nsearchpointm1; ++sp) {
        FloatType x = -absxymax * sp / (FloatType)nsearchpointm1;
        vector<DistPoint<Point2D> > neighbors;
        cloud2Dc.findNeighbors(back_inserter(neighbors), Point2D(x,x), radius);
        if (neighbors.size()>maxd) {
            maxd = neighbors.size();
            maxdxy1 = x;
        }
    }
    maxd = 0;
    for(int sp = 0; sp <= nsearchpointm1; ++sp) {
        FloatType x = absxymax * sp / (FloatType)nsearchpointm1;
        vector<DistPoint<Point2D> > neighbors;
        cloud2Dc.findNeighbors(back_inserter(neighbors), Point2D(x,x), radius);
        if (neighbors.size()>maxd) {
            maxd = neighbors.size();
            maxdxy2 = x;
        }
    }
    // lowest global density region between the 2 max
    int mind = numeric_limits<int>::max();;
    FloatType mindxy = numeric_limits<FloatType>::max();
    for(int sp = 0; sp <= nsearchpointm1; ++sp) {
        FloatType x = maxdxy1 + (maxdxy2 - maxdxy1) * sp / (FloatType)nsearchpointm1;
        vector<DistPoint<Point2D> > neighbors;
        cloud2Dg.findNeighbors(back_inserter(neighbors), Point2D(x,x), radius);
        if (neighbors.size()<mind) {
            mind = neighbors.size();
            mindxy = x;
        }
    }
    
    // now we swipe a decision boundary in each direction around that point
    // and look for the lowest overall density along the boundary
    int nsearchdir = 360; // each half degree, as we swipe from 0 to 180 (unoriented lines)
    FloatType incr = max(xmaxg-xming, ymaxg-yming) / nsearchdir;
    int minsumd = numeric_limits<int>::max();
    FloatType minvx = 0, minvy = 0;
    for(int sd = 0; sd < nsearchdir; ++sd) {
        // use the parametric P = P0 + alpha*V formulation of a line
        // unit vector in the direction of the line
        FloatType vx = cos(M_PI * sd / nsearchdir);
        FloatType vy = sin(M_PI * sd / nsearchdir);
        int sumd = 0;
        for(int sp = -nsearchpointm1/2; sp < nsearchpointm1/2; ++sp) {
            int s = sp * incr;
            FloatType x = mindxy + vx * s;
            FloatType y = mindxy + vy * s;
            vector<DistPoint<Point2D> > neighbors;
            cloud2Dg.findNeighbors(back_inserter(neighbors), Point2D(x,y), radius);
            sumd += neighbors.size();
        }
        if (sumd<minsumd) {
            minsumd = sumd;
            minvx = vx;
            minvy = vy;
        }
    }
    
    // so we finally have the decision boundary in this 2D space
    // P = P0 + alpha * V : px = p0x + alpha * vx  and  py = p0y + alpha * vy
    // alpha = (px - p0x) / vx; // if vx is null see below
    // else py = p0y + (px - p0x) / vx * vy
    // py = p0y + px * vy/vx - p0x * vy/vx;
    // px * vy/vx - py + p0y - p0x * vy/vx = 0
    // equa: wx * px + wy * py + c = 0
    // with: wx = vy/vx; wy = -1; c = p0y - p0x * vy/vx
    // null vx just reverse roles as vy is then !=0 (v is unit vec)
    // wx = -1; wy = vx/vy; c = p0x - p0y * vx/vy
    FloatType wx = 0, wy = 0, c = 0;
    if (minvx!=0) { wx = minvy / minvx; wy = -1; c = mindxy - mindxy * wx;}
    else {wx = -1; wy = minvx / minvy; c = mindxy - mindxy * wy;}
    
    // Convert that in original space
    // equa: wx * px + wy * py + c = 0
    // find normal vector and homogeneous plane equation:
    FloatType normvec2d = sqrt(wx*wx+wy*wy);
    Point2D vecn2d(wx/normvec2d,wy/normvec2d);
    // orient it so it points toward class +1. Dot product with (1,1) shall be >0
    if (vecn2d.x+vecn2d.y < 0) {vecn2d *= -1.0; wx = -wx; wy = -wy; c = -c;}
    
    
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
    FloatType scaleFactorX = halfSvgSize / max(-minX,maxX);
    FloatType scaleFactorY = halfSvgSize / max(-minY,maxY);

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
    // first the unlabelled data, if any
    for (int i=0; i<ndata_unlabeled; ++i) {
        // we have to project this data as this was not done above
        FloatType x = classifier.predict(data_unlabeled[i]) * scaleFactorX + halfSvgSize;
        FloatType y = halfSvgSize - ortho_classifier.predict(data_unlabeled[i]) * scaleFactorY;
        cairo_arc(cr, x, y, 0.714, 0, 2*M_PI);
        cairo_stroke(cr);
    }
    // now plot the reference data. It is very well that it was randomised so we do not have one class on top of the other
    for (int i=0; i<nsamples; ++i) {
        FloatType x = proj1[i]*scaleFactorX + halfSvgSize;
        FloatType y = halfSvgSize - proj2[i]*scaleFactorY;
        if (labels[i]==1) cairo_set_source_rgba(cr, 1, 0, 0, 0.75);
        else cairo_set_source_rgba(cr, 0, 0, 1, 0.75);
        cairo_arc(cr, x, y, 0.714, 0, 2*M_PI);
        cairo_stroke(cr);
    }
    
    // draw lines on top of points
    double dashes[2]; 
    dashes[0] = dashes[1] = svgSize*0.01;
    cairo_set_dash(cr, dashes, 2, -svgSize*0.005);
    cairo_set_source_rgb(cr, 0.25,0.25,0.25);
    cairo_move_to(cr, 0,halfSvgSize);
    cairo_line_to(cr, svgSize,halfSvgSize);
    cairo_move_to(cr, halfSvgSize,0);
    cairo_line_to(cr, halfSvgSize,svgSize);
    cairo_stroke(cr);
    
    //cairo_surface_write_to_png (surface, argv[1]);
    std::vector<char> pngdata;
    pngdata.reserve(800*800*3); // need only large enough init size
    cairo_surface_write_to_png_stream(surface, png_copier, &pngdata);

    cairo_surface_destroy(surface);
    cairo_destroy(cr);
    
    // encode the png data into base64
    base64 codec;
    std::vector<char> base64pngdata(codec.get_max_encoded_size(pngdata.size()));
    int nbytes = codec.encode(&pngdata[0], pngdata.size(), &base64pngdata[0]);
    nbytes += codec.encode_end(&base64pngdata[nbytes]);
    
    // output the svg file
    svgfile << "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\""<< svgSize << "\" height=\""<< svgSize <<"\" >" << endl;
    // TODO: save classifier parameters as SVG comments, possibly base64 encoded text for binary values, or keep at least 10 decimals
    
    // include the image inline    
    svgfile << "<image xlink:href=\"data:image/png;base64,"<< &base64pngdata[0]
            << "\" width=\""<<svgSize<<"\" height=\""<<svgSize<<"\" x=\"0\" y=\"0\" />" << endl;
    
    // plot decision boundary as a path
    // xy space in plane => scale and then reverse
    // first find homogeneous equa in the 2D space
    // convert the decision boundary to SVG space
    // xsvg = x2d * scaleFactorX + halfSvgSize;
    // ysvg = halfSvgSize - y2d * scaleFactorY;
    // x2d = (xsvg - halfSvgSize) / scaleFactorX;
    // y2d = (halfSvgSize - ysvg) / scaleFactorY;
    // wx * x2d + wy * y2d + c = 0
    // wx * (xsvg - halfSvgSize) / scaleFactorX + wy * (halfSvgSize - ysvg) / scaleFactorY + c = 0
    // wx * xsvg - wy * ysvg + c -wx*halfSvgSize/scaleFactorX + wy * halfSvgSize / scaleFactorY = 0
    FloatType wxsvg = wx;
    FloatType wysvg = -wy;
    FloatType csvg = c -wx*halfSvgSize/scaleFactorX + wy*halfSvgSize/scaleFactorY;
    // wxsvg * xsvg + wysvg * ysvg + csvg = 0
    // now intersect to find xminsvg, yminsvg, and so on
    // some may be NaN
    FloatType xsvgy0 = -csvg / wxsvg; // at ysvg = 0
    FloatType ysvgx0 = -csvg / wysvg; // at xsvg = 0
    FloatType xsvgymax = (-csvg -wysvg*svgSize) / wxsvg; // at ysvg = svgSize
    FloatType ysvgxmax = (-csvg -wxsvg*svgSize) / wysvg; // at xsvg = svgSize
    // NaN comparisons always fail, so use only positive tests and this is OK
    bool useLeft = (ysvgx0 >= 0) && (ysvgx0 <= svgSize);
    bool useRight = (ysvgxmax >= 0) && (ysvgxmax <= svgSize);
    bool useTop = (xsvgy0 >= 0) && (xsvgy0 <= svgSize);
    bool useBottom = (xsvgymax >= 0) && (xsvgymax <= svgSize);
    int sidescount = useLeft + useRight + useTop + useBottom;
    if (sidescount==2) {
        svgfile << "<path style=\"fill:none;stroke:#000000;stroke-width:1px;\" d=\"M ";
        if (useLeft) {
            svgfile << 0 << "," << ysvgx0 << " L ";
            if (useTop) svgfile << xsvgy0 << "," << 0 << " ";
            if (useRight) svgfile << svgSize << "," << ysvgxmax << " ";
            if (useBottom) svgfile << xsvgymax << "," << svgSize << " ";
        }
        if (useTop) {
            svgfile << xsvgy0 << "," << 0 << " L ";
            if (useRight) svgfile << svgSize << "," << ysvgxmax << " ";
            if (useBottom) svgfile << xsvgymax << "," << svgSize << " ";
        }
        if (useBottom) {
            svgfile << xsvgymax << "," << svgSize << " L ";
            if (useRight) svgfile << svgSize << "," << ysvgxmax << " ";
            // else internal error
        }
        svgfile << "\" />" << endl;
    }

    svgfile << "</svg>" << endl;
    svgfile.close();

cout << radius * scaleFactorX << " " << radius * scaleFactorY << endl;
    return 0;
}
