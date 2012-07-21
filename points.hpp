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
#include <boost/random/mersenne_twister.hpp>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

/// SOME USER-ADAPTABLE PARAMETERS

#ifndef FLOAT_TYPE
#define FLOAT_TYPE float
#endif
typedef FLOAT_TYPE FloatType;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
        return reinterpret_cast<FloatType*>(&x)[idx];
    }
    PointTemplate() : x(0),y(0),z(0) {}
    PointTemplate(FloatType _x, FloatType _y, FloatType _z) : x(_x),y(_y),z(_z) {}
    PointTemplate(PointTemplate* n) : x(0),y(0),z(0) {}
    
    // Baaah, how many times will similar code be rewriten ? Thanks boost::operators for easing the task
    inline PointTemplate& operator+=(const PointTemplate& v) {x+=v.x; y+=v.y; z+=v.z; return *this;}
    inline PointTemplate& operator-=(const PointTemplate& v) {x-=v.x; y-=v.y; z-=v.z; return *this;}
    inline PointTemplate& operator*=(const FloatType& f) {x*=f; y*=f; z*=f; return *this;}
    inline PointTemplate& operator/=(const FloatType& f) {x/=f; y/=f; z/=f; return *this;}
    
    inline PointTemplate& memmul(const PointTemplate& v) {x*=v.x; y*=v.y; z*=v.z; return *this;}
    inline FloatType norm2() const {return x*x + y*y + z*z;}
    inline FloatType norm() const {return sqrt(norm2());}
    inline FloatType dot(const PointTemplate& v) const {return x*v.x + y*v.y + z*v.z;}
    inline PointTemplate cross(const PointTemplate& v) const {return PointTemplate(y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x);}
    inline void normalize() {double n = norm(); if (n>0) *this /= n;}

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

// only classic notation supported, no fancy hex or the like that atof can handle
// Usage: for (char* x = line; *x!=0;) {value = fast_atof_next_token(x); ... }
// only classic notation supported, no fancy hex or the like that atof can handle
inline FloatType fast_atof_next_token(char* &str) {
    FloatType value = 0;
    FloatType neg = 1;
    if (*str==0) return 0;
    while ((*str==' ')||(*str=='\t')||(*str=='\n')||(*str=='\r')) {
        ++str; if (*str==0) return 0;
    }
    for (;;++str) {
        switch(*str) {
            default: ++str;// break on invalid characters
            case 0: return value*neg; // end of string
            case '-': neg = -1; continue;
            case '+': continue;
            case '0': value *= 10; continue;
            case '1': value = value * 10 + 1; continue;
            case '2': value = value * 10 + 2; continue;
            case '3': value = value * 10 + 3; continue;
            case '4': value = value * 10 + 4; continue;
            case '5': value = value * 10 + 5; continue;
            case '6': value = value * 10 + 6; continue;
            case '7': value = value * 10 + 7; continue;
            case '8': value = value * 10 + 8; continue;
            case '9': value = value * 10 + 9; continue;
            case '.': {
                FloatType tenpow = 10;
                ++str; if (*str==0) return value*neg; // useless terminal .
                while ((*str>='0')&&(*str<='9')) {
                    value += (*str - '0') / tenpow;
                    tenpow *= 10;
                    ++str; if (*str==0) return value*neg;
                } 
                // ignore unknown characters other than e or E
                if (*str!='e'&&*str!='E') {
                    while ((*str==' ')||(*str=='\t')||(*str=='\n')||(*str=='\r')) ++str;
                    return value*neg;
                }
            }
            case 'e':
            case 'E': {
                int exponum = 0;
                bool div = false;
                for (++str;;++str) {
                    switch(*str) {
                        default: ++str;
                        case 0: {
                            // non-recursive fast-exponentiation
                            // TODO: IEEE754 tricks... dependent of FloatType
                            FloatType expoval = 1;
                            FloatType tenpow = 10;
                            while (exponum!=0) {
                                if ((exponum&1)!=0) expoval *= tenpow;
                                tenpow *= tenpow;
                                exponum /= 2;
                            }
                            if (div) return value*neg/expoval; return value*neg*expoval;
                        }
                        case '+': div = false; continue;
                        case '-': div = true; continue;
                        case '0': exponum *= 10; continue;
                        case '1': exponum = exponum * 10 + 1; continue;
                        case '2': exponum = exponum * 10 + 2; continue;
                        case '3': exponum = exponum * 10 + 3; continue;
                        case '4': exponum = exponum * 10 + 4; continue;
                        case '5': exponum = exponum * 10 + 5; continue;
                        case '6': exponum = exponum * 10 + 6; continue;
                        case '7': exponum = exponum * 10 + 7; continue;
                        case '8': exponum = exponum * 10 + 8; continue;
                        case '9': exponum = exponum * 10 + 9; continue;
                    }
                }
                // shall never reach this point
                if (div) return value*neg;
                return value*neg;
            }
        }
    }
    // shall never reach this point
    return value*neg;
}

// argh, not all implementations have the newer getline, simulate it here
#ifdef DEFINE_GETLINE
    // implement the spec with minimal compliance for our needs...
    ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
        static const int chunck = 80;
        if (*lineptr==0) {
            *lineptr = (char*)malloc(chunck);
            *n = chunck;
        }
        int offset = 0;
        while (true) {
            int c = fgetc(stream);
            if (c==EOF) return -1;
            if (offset>=*n) {
                *n += chunck;
                *lineptr = (char*)realloc(*lineptr,*n);
                if (!*lineptr) return -1; // out of mem, shall set errno...
            }
            (*lineptr)[offset++]=(char)c;
            if (c=='\n') break;
        }
        // terminal 0
        if (offset>=*n) {
            *n += chunck;
            *lineptr = (char*)realloc(*lineptr,*n);
            if (!*lineptr) return -1; // out of mem, shall set errno...
        }
        (*lineptr)[offset]=0;
        return offset;
    }
#endif

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

    void prepare(FloatType _xmin, FloatType _xmax, FloatType _ymin, FloatType _ymax, size_t npts) {
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
        data.resize(npts); // without effect if data is already the correct size
        links.resize(npts);
        grid.resize(ncellx * ncelly);
        for (int i=0; i<npts; ++i) links[i] = IndexType(-1);
        for (int i=0; i<(int)grid.size(); ++i) grid[i] = IndexType(-1);
        nextptidx = 0;
    }

    void insert_data_at_index(size_t dataidx) {
        // add this point to the cell grid list
        int cellx = floor((data[dataidx].x - xmin) / cellside);
        int celly = floor((data[dataidx].y - ymin) / cellside);
        //data[nextptidx].next = grid[celly * ncellx + cellx];
        //grid[celly * ncellx + cellx] = &data[nextptidx];
        links[dataidx] = grid[celly * ncellx + cellx];
        grid[celly * ncellx + cellx] = dataidx;
    }

    void insert(const PointType& point) {
        // TODO: if necessary, reallocate data and update pointers. For now just assert
        assert(nextptidx<data.size());
        data[nextptidx] = point;
        insert_data_at_index(nextptidx);
        ++nextptidx;
    }

    void remove(int dataidx) {
        int cellx = floor((data[dataidx].x - xmin) / cellside);
        int celly = floor((data[dataidx].y - ymin) / cellside);
        
        // run through links list to find a good index
        // is this the list head ?
        if (grid[celly * ncellx + cellx]==dataidx) {
            // easy, just walk along
            grid[celly * ncellx + cellx] = links[dataidx];
        } else {
            // run through list, starting second pos
            IndexType previdx = grid[celly * ncellx + cellx];
            for (IndexType idx = links[previdx]; idx != IndexType(-1); previdx = idx, idx=links[idx]) {
                // found? => remove from list
                if (idx==dataidx) {
                    links[previdx] = links[idx];
                    break;
                }
            }
        }

        // now the tricky part.
        // we just removed an element, but it is still in the data vector
        // it may be free from links by anything, not appearing in the grid,
        // but still it pollutes the data vector.
        // => swap it with the last pos, reduce data vector
        // BUT we then need to update links using that last element
        // to use its new index
        
        // prepare the linkage at new pos
        int lastpos = data.size()-1;
        // process only if useful
        if (lastpos!=dataidx) {
            links[dataidx] = links[lastpos];
            // need to find previous element in list...
            cellx = floor((data[lastpos].x - xmin) / cellside);
            celly = floor((data[lastpos].y - ymin) / cellside);
            // run through links list to find a good index
            // is this the list head ?
            if (grid[celly * ncellx + cellx] == lastpos) {
                // update it to new pos
                grid[celly * ncellx + cellx] = dataidx;
            } else {
                // run through list, starting second pos
                IndexType previdx = grid[celly * ncellx + cellx];
                for (IndexType idx = links[previdx]; idx != IndexType(-1); previdx = idx, idx=links[idx]) {
                    // found? => update
                    if (idx==lastpos) {
                        links[previdx] = dataidx;
                        break;
                    }
                }
            }
        }
        
        // now we can finally update the data vector
        data[dataidx] = data[lastpos];
        data.pop_back();
        links.pop_back();
        nextptidx = lastpos;
    }
    
    size_t load_txt(const char* filename, std::vector<std::vector<FloatType> >* additionalInfo = 0, std::vector<size_t> *line_numbers = 0, int subsampling_factor = 0) {
        using namespace std;
        data.clear();
        grid.clear();
        FILE* fp = fopen(filename, "r");
        if (!fp) {std::cerr << "Could not load file: " << filename << std::endl; return 0;}
        // first pass to get the number of points and the bounds
        xmin = numeric_limits<FloatType>::max();
        xmax = -numeric_limits<FloatType>::max();
        ymin = numeric_limits<FloatType>::max();
        ymax = -numeric_limits<FloatType>::max();
        char* line = 0;
        size_t linelen = 0;
        int num_read = 0;
        size_t linenum = 0;
        boost::mt19937* rng = 0;
        if (subsampling_factor) rng = new boost::mt19937;
        while ((num_read = getline(&line, &linelen, fp)) != -1) {
            ++linenum;
            if (linelen==0 || line[0]=='#') continue;
            if (subsampling_factor && ((*rng)()%subsampling_factor>0)) continue;
            if (line_numbers) line_numbers->push_back(linenum);
            if (additionalInfo) additionalInfo->push_back(std::vector<FloatType>());
            PointType point;
            int i = 0;
            // atof & strtok are really too slow, not to mention alternatives...
            for (char* x = line; *x!=0;) {
                FloatType value = fast_atof_next_token(x);
                if (i<Point::dim) point[i] = value;
                else if (additionalInfo) {
                    additionalInfo->back().push_back(value);
                } else break;
                ++i;
            }
            data.push_back(point);
            xmin = min(xmin, point[0]);
            xmax = max(xmax, point[0]);
            ymin = min(ymin, point[1]);
            ymax = max(ymax, point[1]);
        }
        fclose(fp);
        prepare(xmin, xmax, ymin, ymax, data.size());
        nextptidx = data.size();
        for (size_t i = 0; i<data.size(); ++i) insert_data_at_index(i);
        return linenum;
    }
    inline size_t load_txt(std::string s, std::vector<std::vector<FloatType> >* additionalInfo = 0, std::vector<size_t> *line_numbers = 0, int subsampling_factor = 0) {
        return load_txt(s.c_str(), additionalInfo, line_numbers, subsampling_factor);
    }

    // TODO: save_bin / load_bin if txt files take too long to load
    
    template<typename OutputIterator, class SomePointType>
    void findNeighbors(OutputIterator outit, const SomePointType& center, FloatType radius) {
        applyToNeighbors(
            [&outit](FloatType d2, PointType* p) {(*outit++) = DistPoint<PointType>(d2,p);},
            center,
            radius
        );
    }
    
    template<typename FunctorType, class SomePointType>
    void applyToNeighbors(FunctorType functor, const SomePointType& center, FloatType radius) {
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
                if (d2<=r2) functor(d2,&data[p]);
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
