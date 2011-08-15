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
#include <boost/algorithm/string.hpp>

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

/// SOME USER-ADAPTABLE PARAMETERS

#ifndef FLOAT_TYPE
#define FLOAT_TYPE float
#endif
typedef FLOAT_TYPE FloatType;

static const int TargetAveragePointDensityPerGridCell = 10;

// scenes are limited to 4G number of points, increase that to uint64_t if you need more
// using uint32_t indices into arrays saves a lot of memory
// compared to (memory-aligned, 8-bytes) pointers on 64-bit architectures
typedef uint32_t IndexType;

/// /// /// ///

template<class Base>
struct PointTemplate : public Base, boost::addable<PointTemplate<Base>, boost::subtractable<PointTemplate<Base>, boost::multipliable2<PointTemplate<Base>, FloatType, boost::dividable2<PointTemplate<Base>, FloatType> > > > {
    
    FloatType x,y,z;

    inline FloatType& operator[](int idx) {
        return reinterpret_cast<FloatType*>(this)[idx];
    }
    PointTemplate() : x(0),y(0),z(0) {}
    PointTemplate(FloatType _x, FloatType _y, FloatType _z) : x(_x),y(_y),z(_z) {}
    PointTemplate(PointTemplate* n) : x(0),y(0),z(0) {}
    
    // Baaah, how many times will similar code be rewriten ? Thanks boost::operators for easing the task
    inline PointTemplate& operator+=(const PointTemplate& v) {x+=v.x; y+=v.y; z+=v.z; return *this;}
    inline PointTemplate& operator-=(const PointTemplate& v) {x-=v.x; y-=v.y; z-=v.z; return *this;}
    inline PointTemplate& operator*=(const FloatType& f) {x*=f; y*=f; z*=f; return *this;}
    inline PointTemplate& operator/=(const FloatType& f) {x/=f; y/=f; z/=f; return *this;}
    
    inline FloatType norm2() const {return x*x + y*y + z*z;}
    inline FloatType norm() const {return sqrt(norm2());}
    inline FloatType dot(const PointTemplate& v) const {return x*v.x + y*v.y + z*v.z;}
    inline PointTemplate cross(const PointTemplate& v) const {return PointTemplate(y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x);}
    inline FloatType normalize() {double n = norm(); if (n>0) *this /= n;}

    enum {dim = 3};
};

template<class Base>
struct Point2DTemplate : public Base, boost::addable<Point2DTemplate<Base>, boost::subtractable<Point2DTemplate<Base>, boost::multipliable2<Point2DTemplate<Base>, FloatType, boost::dividable2<Point2DTemplate<Base>, FloatType> > > > {
    
    FloatType x,y;

    // convenient but slow : avoid it !
    inline FloatType& operator[](int idx) {
        assert(idx>=0 && idx<2);
        return idx==0?x:y;
    }
    Point2DTemplate() : x(0),y(0) {}
    Point2DTemplate(FloatType _x, FloatType _y) : x(_x),y(_y) {}
    Point2DTemplate(Point2DTemplate* n) : x(0),y(0) {}
    
    // Baaah, how many times will similar code be rewriten ? Thanks boost::operators for easing the task
    inline Point2DTemplate& operator+=(const Point2DTemplate& v) {x+=v.x; y+=v.y; return *this;}
    inline Point2DTemplate& operator-=(const Point2DTemplate& v) {x-=v.x; y-=v.y; return *this;}
    inline Point2DTemplate& operator*=(const FloatType& f) {x*=f; y*=f; return *this;}
    inline Point2DTemplate& operator/=(const FloatType& f) {x/=f; y/=f; return *this;}
    
    inline FloatType dot(const Point2DTemplate& v) {return x*v.x + y*v.y;}
    inline FloatType norm2() {return x*x + y*y;}
    inline FloatType norm() {return sqrt(x*x + y*y);}

    enum {dim = 2};
};

template<class Base>
std::ostream& operator<<(std::ostream &out, const PointTemplate<Base> &p){
    out << "(" << p.x << ", " << p.y << ", " << p.z << ")";
    return out;
}
template<class Base>
std::ostream& operator<<(std::ostream &out, const Point2DTemplate<Base> &p){
    out << "(" << p.x << ", " << p.y << ")";
    return out;
}

struct EmptyStruct {};
typedef PointTemplate<EmptyStruct> Point;
typedef Point2DTemplate<EmptyStruct> Point2D;

template<class PointType1, class PointType2, int Dim>
struct DistComput {
    // generates error if the dimension is not supported
};
// partial specialisation for dimensions 2 and 3
template<class PointType1, class PointType2>
struct DistComput<PointType1,PointType2,2> {
    inline static FloatType dist2(const PointType1& a, const PointType2& b) {
        return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
    }
};
template<class PointType1, class PointType2>
struct DistComput<PointType1,PointType2,3> {
    inline static FloatType dist2(const PointType1& a, const PointType2& b) {
        return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z);
    }
};

// user API function, selects the dimension automatically at compile time (no perf impact)
template<class PointType1, class PointType2>
inline FloatType dist2(const PointType1& a, const PointType2& b) {
    return DistComput<PointType1,PointType2,(int)PointType1::dim==(int)PointType2::dim?(int)PointType1::dim:-1>::dist2(a,b);
}
template<class PointType1, class PointType2>
inline FloatType dist(const PointType1& a, const PointType2& b) {
    return sqrt(dist2(a,b));
}


// this struct accelerate the management of neighbors
// the std::multimaps are way too slow
template<class PointType>
struct DistPoint {
    FloatType distsq;
    PointType* pt;
    bool operator<(const DistPoint& other) const {
        return distsq < other.distsq;
    }
    DistPoint(FloatType _distsq, PointType* _pt) : distsq(_distsq), pt(_pt) {}
    DistPoint() : distsq(0), pt(0) {}
};

template<class PointType>
struct PointCloud {
    std::vector<PointType> data; // avoids many mem allocations for individual points
    FloatType xmin, xmax, ymin, ymax;
    FloatType cellside;
    int ncellx;
    int ncelly;
    // external linkage, using uint32_t indices instead of PointType*
    // on 64-bit systems where pointers need to be aligned this can make a huge difference!
    // prev version: each point had an internal PointType* pointer to the next point in
    // the same grid cell
    // current version: external links, links[i] is the index into data giving the point
    // that would previously be pointed by the pointer at data[i]
    // however now 0 is a valid index => use IndexType(-1) as the invalid index marker
    std::vector<IndexType> links;
    std::vector<IndexType> grid; // cell lists are embedded in the points vector
    IndexType nextptidx;

    void prepare(FloatType _xmin, FloatType _xmax, FloatType _ymin, FloatType _ymax, int npts) {
        xmin = _xmin; xmax = _xmax;
        ymin = _ymin; ymax = _ymax;
        if (xmin==xmax) {xmin -= 0.5; xmax += 0.5;}
        if (ymin==ymax) {ymin -= 0.5; ymax += 0.5;}
        FloatType sizex = xmax - xmin;
        FloatType sizey = ymax - ymin;
        
        cellside = sqrt(TargetAveragePointDensityPerGridCell * sizex * sizey / npts);
        ncellx = floor(sizex / cellside) + 1;
        ncelly = floor(sizey / cellside) + 1;
        
        // instanciate the points
        data.resize(npts);
        links.resize(npts);
        grid.resize(ncellx * ncelly);
        for (int i=0; i<npts; ++i) links[i] = IndexType(-1);
        for (int i=0; i<grid.size(); ++i) grid[i] = IndexType(-1);
        nextptidx = 0;
    }

    void insert(const PointType& point) {
        // TODO: if necessary, reallocate data and update pointers. For now just assert
        assert(nextptidx<data.size());
        data[nextptidx] = point;
        // add this point to the cell grid list
        int cellx = floor((data[nextptidx].x - xmin) / cellside);
        int celly = floor((data[nextptidx].y - ymin) / cellside);
        
        //data[nextptidx].next = grid[celly * ncellx + cellx];
        //grid[celly * ncellx + cellx] = &data[nextptidx];
        links[nextptidx] = grid[celly * ncellx + cellx];
        grid[celly * ncellx + cellx] = nextptidx;
        ++nextptidx;
    }

    void load_txt(const char* filename, std::vector<std::vector<FloatType> >* additionalInfo = 0) {
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
        int linenum = 0;
        bool use4 = false;
        while (datafile && !datafile.eof()) {
            ++linenum;
            getline(datafile, line);
            if (line.empty() || boost::starts_with(line,"#") || boost::starts_with(line,";") || boost::starts_with(line,"!") || boost::starts_with(line,"//")) continue;
            stringstream linereader(line);
            PointType point;
            FloatType value;
            int i = 0;
            while (linereader >> value) {
                if (i<Point::dim) point[i] = value;
                if (++i==4) break;
            }
            if (i==4 && additionalInfo) {
                if (use4==false && !data.empty()) {
                    cout << "Warning: 4rth value met at line " << linenum << " but it was not present before." << endl;
                }
                use4 = true;
            }
            xmin = min(xmin, point[0]);
            xmax = max(xmax, point[0]);
            ymin = min(ymin, point[1]);
            ymax = max(ymax, point[1]);
            ++npts;
        }
        
        prepare(xmin, xmax, ymin, ymax, npts);
        if (additionalInfo) additionalInfo->resize(npts);
        
        // second pass to load the data structure in place
        datafile.close();
        datafile.open(filename);
        while (datafile && !datafile.eof()) {
            getline(datafile, line);
            if (line.empty() || boost::starts_with(line,"#") || boost::starts_with(line,";") || boost::starts_with(line,"!") || boost::starts_with(line,"//")) continue;
            stringstream linereader(line);
            PointType point;
            FloatType value;
            int i = 0;
            while (linereader >> value) {
                if (i<Point::dim) point[i] = value;
                if (i>=Point::dim && additionalInfo!=0) {
                    (*additionalInfo)[nextptidx].push_back(value);
                }
                ++i;
            }
            insert(point);
        }
        datafile.close();
    }
    inline void load_txt(std::string s, std::vector<std::vector<FloatType> >* additionalInfo = 0) {
        load_txt(s.c_str(), additionalInfo);
    }

    // TODO: save_bin / load_bin if txt files take too long to load
    
    template<typename OutputIterator, class SomePointType>
    void findNeighbors(OutputIterator outit, const SomePointType& center, FloatType radius) {
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
            for (IndexType p = grid[cy * ncellx + cx]; p!=IndexType(-1); p=links[p]) {
                FloatType d2 = dist2(center,data[p]);
                if (d2<=r2) (*outit++) = DistPoint<PointType>(d2,&data[p]);
            }
        }
    }

    // returns the index of the nearest point in the cloud from the point given in argument
    // The center point may be excluded from the search or included
    // The exclusion squared distance fixes the threshold at which points are considered the same
    // set it to 0 to include the search point (default)
    // returns -1 iff the cloud is empty
    template<class SomePointType>
    int findNearest(const SomePointType& center, FloatType exclusionDistSq = 0) {
        int cx = floor((center.x - xmin) / cellside);
        int cy = floor((center.y - ymin) / cellside);
        // look for a non-empty cell in increasing distance. Once it is found, the nearest neighbor is necessarily within that radius
        int found_dcell = -1;
        for (int dcell = 0; dcell<std::max(ncellx,ncelly); ++dcell) {
            // loop only in the square at dcell distance from the center cell
            for (int cxi = cx-dcell; cxi <= cx + dcell; ++cxi) {
                // top
                if (cxi>=0 && cxi<ncellx && cy-dcell>=0 && cy-dcell<ncelly && grid[(cy-dcell) * ncellx + cxi]!=IndexType(-1)) {
                    bool otherThanCenter = false;
                    for (IndexType pidx = grid[(cy-dcell) * ncellx + cxi]; pidx != IndexType(-1); pidx=links[pidx]) if (dist2(center,data[pidx])>=exclusionDistSq) {
                        otherThanCenter = true;
                        break;
                    }
                    if (otherThanCenter) {found_dcell = dcell; break;}
                }
                // bottom
                if (cxi>=0 && cxi<ncellx && cy+dcell>=0 && cy+dcell<ncelly && grid[(cy+dcell) * ncellx + cxi]!=IndexType(-1)) {
                    bool otherThanCenter = false;
                    for (IndexType pidx = grid[(cy+dcell) * ncellx + cxi]; pidx != IndexType(-1); pidx=links[pidx]) if (dist2(center,data[pidx])>=exclusionDistSq) {
                        otherThanCenter = true;
                        break;
                    }
                    if (otherThanCenter) {found_dcell = dcell; break;}
                }
            }
            if (found_dcell!=-1) break;
            // left and right, omitting the corners
            for (int cyi = cy-dcell+1; cyi <= cy + dcell - 1; ++cyi) {
                // left
                if (cx-dcell>=0 && cx-dcell<ncellx && cyi>=0 && cyi<ncelly && grid[cyi * ncellx + cx - dcell]!=IndexType(-1)) {
                    bool otherThanCenter = false;
                    for (IndexType pidx = grid[cyi * ncellx + cx - dcell]; pidx != IndexType(-1); pidx=links[pidx]) if (dist2(center,data[pidx])>=exclusionDistSq) {
                        otherThanCenter = true;
                        break;
                    }
                    if (otherThanCenter) {found_dcell = dcell; break;}
                }
                // right
                if (cx+dcell>=0 && cx+dcell<ncellx && cyi>=0 && cyi<ncelly && grid[cyi * ncellx + cx + dcell]!=IndexType(-1)) {
                    bool otherThanCenter = false;
                    for (IndexType pidx = grid[cyi * ncellx + cx + dcell]; pidx != IndexType(-1); pidx=links[pidx]) if (dist2(center,data[pidx])>=exclusionDistSq) {
                        otherThanCenter = true;
                        break;
                    }
                    if (otherThanCenter) {found_dcell = dcell; break;}
                }
            }
            if (found_dcell!=-1) break;
        }
        if (found_dcell==-1) {
#ifndef NDEBUG
            std::cerr << "Could not find dcell: cx=" << cx << ", cy=" << cy << ", ncellx=" << ncellx << ", ncelly=" << ncelly << ", center.x=" << center.x << ", center.y=" << center.y << ", xmin=" << xmin << ", xmax=" << xmax << ", ymin=" << ymin << ", ymax=" << ymax << std::endl;
#endif
            return -1;
        }
        // neighbor necessarily within dcell+1 distance, limit case if we are very close to a cell edge
        IndexType idx = IndexType(-1);
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
            for (IndexType p = grid[cy * ncellx + cx]; p!=IndexType(-1); p=links[p]) {
                FloatType d2 = dist2(center,data[p]);
                if (d2<exclusionDistSq) continue;
                if (d2<=mind2) {
                    mind2 = d2;
                    idx = p;
                }
            }
        }
        if (idx==IndexType(-1)) {
#ifndef NDEBUG
            std::cerr << "Could not find index: found_dcell=" << found_dcell << ", cx=" << cx << ", cy=" << cy << ", ncellx=" << ncellx << ", ncelly=" << ncelly << ", center.x=" << center.x << ", center.y=" << center.y << ", xmin=" << xmin << ", xmax=" << xmax << ", ymin=" << ymin << ", ymax=" << ymax << std::endl;
#endif
            return -1;
        }
        return idx;
    }

};
//PointCloud<Point> cloud;


#endif
