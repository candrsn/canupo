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
int read_msc_header(ifstream& mscfile, vector<FloatType>& scales, int& ptnparams) {
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
    int ptnparams;
    
    // neutral files, if any
    int ndata_unlabeled = 0;
    vector<FloatType> scales;
    for (int argi = 2; argi<arg_class1-1; ++argi) {
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
    for (int argi = 2; argi<arg_class1-1; ++argi) {
        ifstream mscfile(argv[argi], ifstream::binary);
        // read the file header (again)
        int npts = read_msc_header(mscfile, scales, ptnparams);
        // read data
        read_msc_data(mscfile,nscales,npts,&data_unlabeled[base_pt], ptnparams);
        mscfile.close();
        base_pt += npts;
    }
    
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


    LinearSVM classifier;

    // shuffle before cross-validating to spread instances of each class
    dlib::randomize_samples(samples, labels);
    FloatType nu = classifier.crossValidate(10, samples, labels);
    classifier.train(10, nu, samples, labels);
    
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
    ortho_classifier.train(10, nu, samples, labels);

    vector<FloatType> proj2(nsamples);
    for (int i=0; i<nsamples; ++i) proj2[i] = ortho_classifier.predict(samples[i]);
    
    // Warning: this is in the projected space !
    // samples X were transformed
    // Xproj = X - (N . (X - S)) * N  with S shift vector
    // second hyperplane equa in original space : 
    // N_ortho . (Xproj - S_ortho) = 0
    // N_ortho . (X - (N . (X - S)) * N - S_ortho) = 0
    // But N_ortho . N = 0 by def
    // N_ortho . (X - S_ortho) = 0   // ok, equa directly applicable in original space
    vector<FloatType> nvec_ortho(fdim); // normal vector
    FloatType norm_ortho = 0;
    for (int i=0; i<fdim; ++i) {
        nvec_ortho[i] = ortho_classifier.weights[i];
        norm_ortho += nvec_ortho[i] * nvec_ortho[i];
    }
    norm_ortho = sqrt(norm_ortho);
    for (int i=0; i<fdim; ++i) nvec_ortho[i] /= norm_ortho;
    vector<FloatType> svec_ortho(fdim); // shift vector
    // dot product between normal and shift is given by the bias
    FloatType sndot_ortho = -ortho_classifier.weights[fdim] / norm_ortho;
    for (int i=0; i<fdim; ++i) svec_ortho[i] = sndot_ortho * nvec_ortho[i];
    
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
    int nsearchpointm1 = 1000;
    // TODO: radius from probabilistic SVM and region of bad classif proba
    FloatType radius = absxymax / 100;
    
    // now we swipe a decision boundary in each direction around the origin
    // and look for the lowest overall density along the boundary
    int nsearchdir = 720; // each quarter degree, as we swipe from 0 to 180 (unoriented lines)
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
            FloatType x = vx * s;
            FloatType y = vy * s;
            vector<DistPoint<Point2D> > neighbors;
            cloud2D.findNeighbors(back_inserter(neighbors), Point2D(x,y), radius);
            sumd += neighbors.size();
        }
        if (sumd<minsumd) {
            minsumd = sumd;
            minvx = vx;
            minvy = vy;
        }
    }
    
    // so we finally have the decision boundary in this 2D space
    // P = P0 + alpha * V : px = alpha * vx  and  py = alpha * vy, P0=origin
    // alpha = px / vx; // if vx is null see below
    // py = px / vx * vy
    // py = px * vy/vx
    // px * vy/vx - py = 0
    // equa: wx * px + wy * py + c = 0
    // with: wx = vy/vx; wy = -1; c = 0
    // null vx just reverse roles as vy is then !=0 (v is unit vec)
    FloatType wx = 0, wy = 0;
    if (minvx!=0) { wx = minvy / minvx; wy = -1; }
    else {wx = -1; wy = minvx / minvy; }
    
    // Convert that in original space
    // find normal vector and homogeneous plane equation:
    // equa: wx * px + wy * py = 0  as it passes through the origin
    FloatType normvec2d = sqrt(wx*wx+wy*wy);
    Point2D vecn2d(wx/normvec2d,wy/normvec2d);
    // orient it so it points toward class +1. Dot product with (1,1) shall be >0
    if (vecn2d.x+vecn2d.y < 0) {vecn2d *= -1.0; wx = -wx; wy = -wy;}
    
    
    // Xproj = X - (N . (X - S)) * N  with S shift vector
    // second hyperplane equa in original space : 
    // N_ortho . (Xproj - S_ortho) = 0
    // N_ortho . (X - (N . (X - S)) * N - S_ortho) = 0
    // But N_ortho . N = 0 by def
    // N_ortho . (X - S_ortho) = 0   // ok, equa directly applicable in original space
    
    // In the original space, we have the relation
    // N_final = vecn2d.x * N + vecn2d.y * N_ortho
    vector<FloatType> weights(fdim+1);
    for (int i=0; i<fdim; ++i) weights[i] = vecn2d.x * nvec[i] + vecn2d.y * nvec_ortho[i];

    // now the shift vector...
    
    // plane goes through (0,0) in 2D space
    // but x2d = N . (X - S)
    // and y2d = N_ortho . (Xproj - S_ortho)
    //     y2d = N_ortho . (X - (N . (X - S)) * N - S_ortho)
    //     y2d = N_ortho . (X - S_ortho)  as N_ortho . N = 0 
    
    // x2d + N.S  =  N.X
    // y2d + No.So = No.X  and x2d=y2d=0 here for one X in the decision boundary
    
    // if we have an orthonormal basis starting from N=N1, then N2, ...
    // X = s1.N1 + s2.N2 + ... satisfies the constraints
    //   all vectors are orthogonal : N1.X = s1, N2.X = s2, and so on
    
    // => simply continue the orthogonal decomposition using the SVM ?
    // No need ?
    // decision boundary is an hyperplane, of the form
    // N_dec . (X - S_dec) = 0
    // N_dec . X - N_dec . S_dec = 0
    // we seek here the bias term of the equa , b_dec = - N_dec . S_dec
    // b_dec = - N_dec . X
    // But X = s1.N1 + s2.N2 + Y  and N_dec = vecn2d.x * N1 + vecn2d.y * N2
    // N_dec . X = vecn2d.x * s1 + vecn2d.y * s2 + 0
    // we know s1 and s2 for the point at the origin in 2D space, also in the hyperplane
    // => s1 = sndot, s2 = sndot_ortho
    weights[fdim] = - (sndot * vecn2d.x + sndot_ortho * vecn2d.y);

    // Now that we have the weights for the linear classifier, compute some stats on the scales
    // the scales 2D planes are the basis vectors of the original space
    // => project the normal vector of the decision hyperplane on each 2D subspace
    // Then we know if that space contributes or not, given the norm of the projection
    // if null, then that 2D subspace is orthogonal to the normal vector
    // => within the hyperplane => no contribution
    // if 1, then that 2D subspace is fully contributing
    cout << "For the default classifier:" << endl;
    for (int s=0; s<nscales; ++s) {
        FloatType contrib = sqrt(
            weights[s*2]*weights[s*2]
          + weights[s*2+1]*weights[s*2+1]
        );
        // from dot-prod = cos to and angle between 0 and pi/2 => rescale in 0..1
        contrib = min(1.0, max(0.0, acos(contrib) / M_PI_2));
        cout << "Scale " << scales[s] << " has a contribution coefficient of " << contrib << endl;
    }
    
    
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
    const char* text = ">d eq. p(err)<1%";
    cairo_text_extents(cr, text, &extents);
    cairo_move_to(cr, svgSize - dprob - 20 - extents.width - extents.x_bearing, svgSize - 15 - extents.height/2 - extents.y_bearing);
    cairo_show_text(cr, text);
    cairo_move_to(cr, svgSize - dprob - 10, svgSize - 15);
    cairo_line_to(cr, svgSize - 10, svgSize - 15);
    dprob = -log(1.0/0.95 - 1.0) * scaleFactor;
    text = ">d eq. p(err)<5%";
    cairo_text_extents(cr, text, &extents);
    cairo_move_to(cr, svgSize - dprob - 20 - extents.width - extents.x_bearing, svgSize - 35 - extents.height/2 - extents.y_bearing);
    cairo_show_text(cr, text);
    cairo_move_to(cr, svgSize - dprob - 10, svgSize - 35);
    cairo_line_to(cr, svgSize - 10, svgSize - 35);
    dprob = -log(1.0/0.9 - 1.0) * scaleFactor;
    text = ">d eq. p(err)<10%";
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
    
    // include the image inline    
    svgfile << "<image xlink:href=\"data:image/png;base64,"<< &base64pngdata[0]
            << "\" width=\""<<svgSize<<"\" height=\""<<svgSize<<"\" x=\"0\" y=\"0\" style=\"z-index:0\" />" << endl;
    
    // include the reference points
    svgfile << "<circle cx=\""<< (refpt_pos.x*scaleFactor+halfSvgSize) <<"\" cy=\""<< (halfSvgSize-refpt_pos.y*scaleFactor) <<"\" r=\"2\" style=\"fill:none;stroke:#000000;stroke-width:1px;z-index:1;\" />" << endl;
    svgfile << "<circle cx=\""<< (refpt_neg.x*scaleFactor+halfSvgSize) <<"\" cy=\""<< (halfSvgSize-refpt_neg.y*scaleFactor) <<"\" r=\"2\" style=\"fill:none;stroke:#000000;stroke-width:1px;z-index:1;\" />" << endl;

    // plot decision boundary as a path
    // xy space in plane => scale and then reverse
    // first find homogeneous equa in the 2D space
    // convert the decision boundary to SVG space
    FloatType wxsvg = wx;
    FloatType wysvg = -wy;
    FloatType csvg = (wy-wx)*halfSvgSize;
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
    vector<Point2D> path;
    if (sidescount==2) {
        svgfile << "<path style=\"fill:none;stroke:#000000;stroke-width:1px;z-index:1;\" d=\"M ";
        // in each case we add a point at the origin to ease user edition of the line
        // in this config only left/right or top/bottom would be useful as we
        // pass through the origin, but we keep this older more generic code just in case
        if (useLeft) {
            svgfile << 0 << "," << ysvgx0 << " L " << halfSvgSize<<","<<halfSvgSize<<" L ";
            if (useTop) svgfile << xsvgy0 << "," << 0 << " ";
            if (useRight) svgfile << svgSize << "," << ysvgxmax << " ";
            if (useBottom) svgfile << xsvgymax << "," << svgSize << " ";
        }
        if (useTop) {
            svgfile << xsvgy0 << "," << 0 << " L " << halfSvgSize<<","<<halfSvgSize<<" L ";
            if (useRight) svgfile << svgSize << "," << ysvgxmax << " ";
            if (useBottom) svgfile << xsvgymax << "," << svgSize << " ";
        }
/*        if (useBottom) {
            svgfile << xsvgymax << "," << svgSize << " L " << halfSvgSize<<","<<halfSvgSize<<" L ";
            if (useRight) svgfile << svgSize << "," << ysvgxmax << " ";
            // else internal error
        }
*/
        svgfile << "\" />" << endl;
    }
    
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
    
    codec.reset_encoder();
    std::vector<char> base64commentdata(codec.get_max_encoded_size(binary_parameters.size()));
    nbytes = codec.encode(&binary_parameters[0], binary_parameters.size(), &base64commentdata[0]);
    nbytes += codec.encode_end(&base64commentdata[nbytes]);
    
    svgfile << "<!-- params " << &base64commentdata[0] << " -->\n" << endl;    

    svgfile << "</svg>" << endl;
    svgfile.close();

    return 0;
}
