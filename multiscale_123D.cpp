#include <iostream>
#include <fstream>

#include "points.hpp"
#include "svd.hpp"

using namespace std;

int main(int argc, char** argv) {

    if (argc<4) {
        cout << "Arguments required: data_file binary_feature_output_file radius1 [radius2...]" << endl;
        return 0;
    }

    vector<FloatType> radiusvec;
    for (int argi = 3; argi<argc; ++argi) {
        FloatType radius = atof(argv[3]);
        radiusvec.push_back(radius);
    }
    int nrad = radiusvec.size();

    cloud.load_txt(argv[1]);
    
    ofstream multiscalefile(argv[2], ofstream::binary);
    
    int npts = 0;

    // some informative file header
    // we will correct the effective npts later, just reserve the space now
    multiscalefile.write(reinterpret_cast<char*>(&npts), sizeof(int));
    int fdim = nrad * 2; // only a and b now, possibly more later
    multiscalefile.write(reinterpret_cast<char*>(&fdim), sizeof(int));
                                                                

    ofstream borderfield("borderfield.xyz");


    
    // TODO: new cloud struct
    
    
    Sequence seq = data.sequence();
    while (seq.hasNext()) {
        Point* p = seq.next();

        // this is so we can recover the data set even with the tree ordering
        // but this is useless as we can always reprovide the binary tree input file if needed
        // multiscalefile.write(reinterpret_cast<char*>(&(*p)[0]), sizeof(FloatType));
        // multiscalefile.write(reinterpret_cast<char*>(&(*p)[1]), sizeof(FloatType));
        // multiscalefile.write(reinterpret_cast<char*>(&(*p)[2]), sizeof(FloatType));
        // moreover we'll reproject the entire data set on the basis feature vectors at some point...
        
        vector<Point> neighbors;
        
        for (int radi = nrad-1; radi>=0; --radi) {
            FloatType radius = radiusvec[radi];

            vector<FloatType> svalues(3);
            // default is to be perfectly 3D if there are not enough neighbors
            // but then, it shall never happen as we detect that condition too now !
            svalues[0] = svalues[1] = svalues[2] = FloatType(1)/FloatType(3);
            
            // start from high dim to low dim, reuse neighbors
            if (radi==nrad-1) {
                neighbors.clear();
                data.FindInSphere(radius, neighbors, *p);
                // Problem: we must avoid the side effects when dealing with objects on the border of the scene
                // This is especially important when computing the multiscale parameters for representative objects
                // of each class: these are provided by the user in small-scene files
                // Ad-hoc solution : when the neighbors are only on one side, then we are at a border
                //                   => when the center of the local cloud is far off the current point !
                //  x-radius  border   x  c       x+radius
                //      |     |        |  |           |
                // We see that the center c is shifted away from x
                // If x is exactly on the border, then c is 1/2 radius away
                // Ad-hoc rule allowing for some natural fluctuation in the cloud: c shall be < 1/8 radius
                Point center; center[0] = center[1] = center[2] = 0;
                for(int i=0; i<neighbors.size(); ++i) for (int j=0; j<3; ++j) center[j] += neighbors[i][j];
                for (int j=0; j<3; ++j) center[j] /= neighbors.size();
                if (neighbors.size()<3) {
                    // center is too far away from real point, eliminate it
                    // eliminate isolated points with not enough neighbors even at the largest scale
                    radi=0;
                    continue;
                } else {
                    ++npts;
                }
                
                // Store the statistic about the border in the file, rather than eliminating border points here.
                // The feature generation step will deal with it, and allow the user to fine-tune
                // rather than setting an arbitrary value here
                // changing that value would mean doing all the computations again
                FloatType borderStat = point_dist(center, *p) / radius;
                multiscalefile.write(reinterpret_cast<char*>(&borderStat), sizeof(FloatType));

                // save the cloud so as to display the border property as a scalar field in CloudCompare
                borderfield << (*p)[0] << " " << (*p)[1] << " " << (*p)[2] << " " << borderStat << endl;
            }
            else {
                // keep only the neighbors within this lower radius
                // brute-force check, faster than rebuilding a local tree.
                // perhaps a better method shall be designed...
                FloatType radius2 = radius*radius;
                for(int i=0; i<neighbors.size();) {
                    if (point_dist2(neighbors[i],*p)<radius2) {
                        neighbors[i] = neighbors.back();
                        neighbors.pop_back();
                    }
                    else ++i;
                }
            }
            
            if (neighbors.size()<3) {
                //cout << "Warning: not enough points for 3D PCA, try increasing radius size" << endl;
                //solution : keep the value from the higher-level radius !
            } else {
                // TODO: subsamble randomly before doing the PCA (ex: keep at most 1000 neighbors in range)
                //       if the performances are not satisfying enough
                Point avg; avg[0] = 0; avg[1] = 0; avg[2] = 0;
                for (int i=0; i<neighbors.size(); ++i) for (int j=0; j<3; ++j) avg[j] += neighbors[i][j];
                for (int j=0; j<3; ++j) avg[j] /= neighbors.size();
                // compute PCA on the neighbors at this radius
                FloatType* A = new FloatType[neighbors.size() * 3];
                for (int j=0; j<3; ++j) for (int i=0; i<(int)neighbors.size(); ++i) {
                    A[j*neighbors.size()+i] = neighbors[i][j] - avg[j];
                }
                // SVD decomposition handled by LAPACK: we only need the singular values
                svd(neighbors.size(), 3, A, &svalues[0]);
                // convert to percent variance explained by each dim
                FloatType totalvar = 0;
                for (int i=0; i<3; ++i) {
                    svalues[i] = svalues[i] * svalues[i]; // / (neighbors.size() - 1);
                    totalvar += svalues[i];
                }
                for (int i=0; i<3; ++i) svalues[i] /= totalvar;
                delete[] A;
                //cout << neighbors.size() << ": " << svalues[0] << " " << svalues[1] << " " << svalues[2] << endl;
            }

            // transform svalue domain into equilateral triangle
            FloatType a = 0.5 * svalues[0] + 2.5 * svalues[1] - 0.5;
            // 3 * sqrt(3) / 2
            FloatType b = -2.59807621135332 * (svalues[0] + svalues[1] - 1);

    
    // another dimension reduction technique
    // possible sampling/discretising in the triangle space
/*    
    FloatType c = nsubdiv * (a + 0.577350269189626 * b); // sqrt(3)/3
    FloatType d = nsubdiv * (1.154700538379252 * b);     // sqrt(3)*2/3
    int cellx = (int)floor(c);
    int celly = (int)floor(d);
    int lower = (c - cellx) > (d - celly);
    if (cellx>=nsubdiv) {cellx=nsubdiv-1; lower = 1;}
    if (cellx<0) {cellx=0; lower = 1;}
    if (celly>=nsubdiv) {celly=nsubdiv-1; lower = 1;} // upper triangle cell = lower one
    if (celly<0) {celly=0; lower = 1;}
    if (celly>cellx) {celly=cellx; lower = 1;}
    int idx = ((cellx * (cellx+1) / 2) + celly) * 2 + lower;
*/  
    // but then we'll need a proper distance function based on indices
            
            multiscalefile.write(reinterpret_cast<char*>(&a), sizeof(FloatType));
            multiscalefile.write(reinterpret_cast<char*>(&b), sizeof(FloatType));

            //multiscalefile << " " << a << " " << b;
            //multiscalefile << endl;
        }
        
    }
    
    // rewrite the effective number of points
    multiscalefile.seekp(0);
    multiscalefile.write(reinterpret_cast<char*>(&npts), sizeof(int));
    multiscalefile.close();

#ifndef NDEBUG
    borderfield.close();
#endif
    
    return 0;
}


