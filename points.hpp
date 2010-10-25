#ifndef CANUPO_POINTS_HPP
#define CANUPO_POINTS_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <limits>
#include <vector>
#include <algorithm>

#include <boost/operators.hpp>

#include <math.h>
#include <stdlib.h>
#include <assert.h>

/// SOME USER-ADAPTABLE PARAMETERS

typedef double FloatType;

static const int TargetAveragePointDensityPerGridCell = 10;

/// /// /// ///

struct Point : boost::addable<Point, boost::subtractable<Point, boost::multipliable2<Point, FloatType, boost::dividable2<Point, FloatType> > > > {
    
    FloatType x,y,z;
    Point* next; // for the cell/grid structure

    // convenient but slow : avoid it !
    inline FloatType& operator[](int idx) {
        assert(idx>=0 && idx<3);
        return idx==0?x:(idx==1?y:z);
    }
    Point() : x(0),y(0),z(0),next(0) {}
    Point(FloatType _x, FloatType _y, FloatType _z) : x(_x),y(_y),z(_z), next(0) {}
    Point(Point* n) : x(0),y(0),z(0),next(n) {}
    
    // Baaah, how many times will similar code be rewriten ? Thanks boost::operators for easing the task
    inline Point& operator+=(const Point& v) {x+=v.x; y+=v.y; z+=v.z; return *this;}
    inline Point& operator-=(const Point& v) {x-=v.x; y-=v.y; z-=v.z; return *this;}
    inline Point& operator*=(const FloatType& f) {x*=f; y*=f; z*=f; return *this;}
    inline Point& operator/=(const FloatType& f) {x/=f; y/=f; z/=f; return *this;}
};

inline FloatType dist(const Point& a, const Point& b) {
    return sqrt(
        (a.x-b.x)*(a.x-b.x)
      + (a.y-b.y)*(a.y-b.y)
      + (a.z-b.z)*(a.z-b.z)
    );
}
inline FloatType dist2(const Point& a, const Point& b) {
    return (a.x-b.x)*(a.x-b.x)
      + (a.y-b.y)*(a.y-b.y)
      + (a.z-b.z)*(a.z-b.z)
    ;
}

// this struct accelerate the management of neighbors
// the std::multimaps are way too slow
struct DistPoint {
    FloatType distsq;
    Point* pt;
    bool operator<(const DistPoint& other) const {
        return distsq < other.distsq;
    }
    DistPoint(FloatType _distsq, Point* _pt) : distsq(_distsq), pt(_pt) {}
    DistPoint() : distsq(0), pt(0) {}
};

struct PointCloud {
    std::vector<Point> data; // avoids many mem allocations for individual points
    FloatType xmin, xmax, ymin, ymax;
    FloatType cellside;
    int ncellx;
    int ncelly;
    std::vector<Point*> grid; // cell lists are embedded in the points vector

    void load_txt(const char* filename) {
        using namespace std;
        data.clear();
        grid.clear();
        ifstream datafile(filename);
        string line;
        int npts = 0;
        // first pass to get the number of points and the bounds
        xmin = numeric_limits<FloatType>::max();
        xmax = -numeric_limits<FloatType>::max();
        ymin = numeric_limits<FloatType>::max();
        ymax = -numeric_limits<FloatType>::max();
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
            if (i<3) {
                cerr << "Invalid data file: " << filename << endl;
                exit(1);
            }
            xmin = min(xmin, point[0]);
            xmax = max(xmax, point[0]);
            ymin = min(ymin, point[1]);
            ymax = max(ymax, point[1]);
            ++npts;
        }
        FloatType sizex = xmax - xmin;
        FloatType sizey = ymax - ymin;
        
        cellside = sqrt(TargetAveragePointDensityPerGridCell * sizex * sizey / npts);
        ncellx = floor(sizex / cellside) + 1;
        ncelly = floor(sizey / cellside) + 1;
        
        // instanciate the points
        data.resize(npts);
        grid.resize(ncellx * ncelly);
        for (int i=0; i<grid.size(); ++i) grid[i] = 0;
        
        // second pass to load the data structure in place
        int ptidx=0;
        datafile.close();
        datafile.open(filename);
        while (datafile && !datafile.eof()) {
            getline(datafile, line);
            if (line.empty()) continue;
            stringstream linereader(line);
            FloatType value;
            int i = 0;
            while (linereader >> value) {
                data[ptidx][i] = value;
                if (++i==3) break;
            }
            // add this point to the cell grid list
            int cellx = floor((data[ptidx].x - xmin) / cellside);
            int celly = floor((data[ptidx].y - ymin) / cellside);
            data[ptidx].next = grid[celly * ncellx + cellx];
            grid[celly * ncellx + cellx] = &data[ptidx];
            ++ptidx;
        }
        datafile.close();
    }
    inline void load_txt(std::string s) {load_txt(s.c_str());}

    // TODO: save_bin / load_bin if txt files take too long to load
    
    template<typename OutputIterator>
    void findNeighbors(OutputIterator outit, const Point& center, FloatType radius) {
        int cx1 = floor((center.x - radius - xmin) / cellside);
        int cx2 = floor((center.x + radius - xmin) / cellside);
        int cy1 = floor((center.y - radius - ymin) / cellside);
        int cy2 = floor((center.y + radius - ymin) / cellside);
        if (cx1<0) cx1=0;
        if (cx2>=ncellx) cx2=ncellx-1;
        if (cy1<0) cy1=0;
        if (cy2>=ncelly) cy2=ncelly-1;
        double r2 = radius * radius;
        for (int cy = cy1; cy <= cy2; ++cy) for (int cx = cx1; cx <= cx2; ++cx) {
            for (Point* p = grid[cy * ncellx + cx]; p!=0; p=p->next) {
                FloatType d2 = dist2(center,*p);
                if (d2<=r2) (*outit++) = DistPoint(d2,p);
            }
        }
    }

    // returns the index of the nearest point in the cloud from the point given in argument
    // returns -1 iff the cloud is empty
    int findNearest(const Point& center) {
        int cx = floor((center.x - xmin) / cellside);
        int cy = floor((center.y - ymin) / cellside);
        // look for a non-empty cell in increasing distance. Once it is found, the nearest neighbor is necessarily within that radius
        int found_dcell = -1;
        for (int dcell = 0; dcell<std::max(ncellx,ncelly); ++dcell) {
            // loop only in the square at dcell distance from the center cell
            for (int cxi = cx-dcell; cxi <= cx + dcell; ++cxi) {
                // top
                if (cxi>=0 && cxi<ncellx && cy-dcell>=0 && cy-dcell<ncelly && grid[(cy-dcell) * ncellx + cxi]!=0) {found_dcell = dcell; break;}
                // bottom
                if (cxi>=0 && cxi<ncellx && cy+dcell>=0 && cy+dcell<ncelly && grid[(cy+dcell) * ncellx + cxi]!=0) {found_dcell = dcell; break;}
            }
            if (found_dcell!=-1) break;
            // left and right, omitting the corners
            for (int cyi = cy-dcell+1; cyi <= cy + dcell - 1; ++cyi) {
                // left
                if (cx-dcell>=0 && cx-dcell<ncellx && cyi>=0 && cyi<ncelly && grid[cyi * ncellx + cx - dcell]!=0) {found_dcell = dcell; break;}
                // right
                if (cx+dcell>=0 && cx+dcell<ncellx && cyi>=0 && cyi<ncelly && grid[cyi * ncellx + cx + dcell]!=0) {found_dcell = dcell; break;}
            }
            if (found_dcell!=-1) break;
        }
        if (found_dcell==-1) return -1;
        // neighbor necessarily within dcell+1 distance, limit case if we are very close to a cell edge
        int idx = -1;
        int cx1 = cx - found_dcell;
        int cx2 = cx + found_dcell;
        int cy1 = cy - found_dcell;
        int cy2 = cy + found_dcell;
        if (cx1<0) cx1=0;
        if (cx2>=ncellx) cx2=ncellx-1;
        if (cy1<0) cy1=0;
        if (cy2>=ncelly) cy2=ncelly-1;
        FloatType mind2 = std::numeric_limits<FloatType>::max();
        for (int cy = cy1; cy <= cy2; ++cy) for (int cx = cx1; cx <= cx2; ++cx) {
            for (Point* p = grid[cy * ncellx + cx]; p!=0; p=p->next) {
                FloatType d2 = dist2(center,*p);
                if (d2<=mind2) {
                    mind2 = d2;
                    idx = p - &data[0];
                }
            }
        }
        return idx;
    }

};
PointCloud cloud;


#endif
