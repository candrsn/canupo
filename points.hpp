#ifndef CANUPO_POINTS_HPP
#define CANUPO_POINTS_HPP

#include "TNear.hpp"

#include <boost/array.hpp>

typedef double FloatType;

typedef boost::array<FloatType, 3> Point;

inline FloatType point_dist(const Point& a, const Point& b) {
    return sqrt(
        (a[0]-b[0])*(a[0]-b[0])
      + (a[1]-b[1])*(a[1]-b[1])
      + (a[2]-b[2])*(a[2]-b[2])
    );
}
inline FloatType point_dist2(const Point& a, const Point& b) {
    return (a[0]-b[0])*(a[0]-b[0])
      + (a[1]-b[1])*(a[1]-b[1])
      + (a[2]-b[2])*(a[2]-b[2])
    ;
}

typedef CNearTree<Point, FloatType, &point_dist>::Sequence Sequence;

#endif
