#ifndef CANUPO_CLASSIFIER_HPP
#define CANUPO_CLASSIFIER_HPP

#include "points.hpp"

struct Classifier {
    
    enum {gridsize = 100};
    
    int class1, class2;
    std::vector<FloatType> weights_axis1, weights_axis2;
    std::vector<Point2D> path;
    FloatType absmaxXY, axis_scale_ratio;
    
    struct LineDef {
        FloatType wx, wy, c;
    };
    std::vector<LineDef> pathlines;
    
    Point2D refpt_pos, refpt_neg;
    
    std::vector<FloatType> grid;

    void prepare() {
        using namespace std;
        // exchange refpt_pos and refpt_neg if necessary, the user may have moved them
        // dot product with (+1,+1) vector gives the classification sign
        if (refpt_pos.x + refpt_pos.y < 0) {
            Point2D tmp = refpt_pos;
            refpt_pos = refpt_neg;
            refpt_neg = tmp;
        }
        if (refpt_pos.x + refpt_pos.y < 0) {
            cerr << "Invalid reference points in the classifier" << endl;
            exit(1);
        }

        // compute the lines
        for(int i=0; i<path.size()-1; ++i) {
            LineDef ld;
            FloatType xdelta = path[i+1].x - path[i].x;
            FloatType ydelta = path[i+1].y - path[i].y;
            if (fabs(xdelta) > 1e-6) {
                // y = slope * x + bias
                ld.wy = -1;
                ld.wx = ydelta / xdelta; // slope
                ld.c = path[i].y - path[i].x * ld.wx;
            } else {
                if (fabs(ydelta) < 1e-6) {
                    cerr << "invalid path definition in classifier" << endl;
                    exit(1);
                }
                // just reverse the roles for a quasi-vertical line at x ~ cte
                ld.wx = -1;
                ld.wy = xdelta / ydelta; // is quasi null here, assuming ydelta != 0
                ld.c = path[i].x - path[i].y * ld.wy;
            }
            FloatType norm = sqrt(ld.wx*ld.wx + ld.wy*ld.wy);
            ld.wx /= norm; ld.wy /= norm; ld.c /= norm;
            pathlines.push_back(ld);
        }
    }

    FloatType classify2D_checkcondnum(FloatType a, FloatType b, Point2D& refpt, FloatType& condnumber) {
        using namespace std;
        Point2D pt(a,b);
        // consider each path line as a mini-classifier
        // the segments pt-refpt_pos  and  that segment line cross
        // iff each classifies the end point of the other in different classes
        // we only need a normal vector in each case, not necessary unit 1, as only the sign counts
        // normal vector <=> homogeneous equa
        LineDef ld;
            FloatType xdelta = refpt.x - a;
        FloatType ydelta = refpt.y - b;
        if (fabs(xdelta) > 1e-3) {
            // y = slope * x + bias
            ld.wy = -1;
            ld.wx = ydelta / xdelta; // slope
            ld.c = b - a * ld.wx;
        } else {
            // just reverse the roles for a quasi-vertical line at x ~ cte
            ld.wx = -1;
            ld.wy = xdelta / ydelta; // is quasi null here, assuming ydelta != 0
            ld.c = a - b * ld.wy;
        }
        FloatType norm = sqrt(ld.wx*ld.wx + ld.wy*ld.wy);
        ld.wx /= norm; ld.wy /= norm; ld.c /= norm;
        Point2D refsegn(ld.wx, ld.wy);
        Point2D refshift = refsegn * ld.c;
        FloatType closestDist = numeric_limits<FloatType>::max();
        int selectedSeg = -1;
        int numcross = 0;
        condnumber = -numeric_limits<FloatType>::max();
        for (int i=0; i<pathlines.size(); ++i) {
            Point2D n(pathlines[i].wx, pathlines[i].wy);
            condnumber = (FloatType)max(condnumber, (FloatType)fabs(n.dot(refsegn)));
            Point2D shift = n * pathlines[i].c;
            
            // Compute whether refpt-pt and that segment cross
            // 1. check whether the given pt and the refpt are on different sides of the classifier line
            bool pathseparates = n.dot(pt + shift) * n.dot(refpt + shift) < 0;
            bool refsegseparates;
            // first and last lines are projected to infinity
            if (i==0) {
                // projection of the end point on ref line
                // Point2D p = path[i+1] - refsegn * refsegn.dot(path[i+1] + refshift);
                // path[i+1] - p = refsegn * refsegn.dot(path[i+1] + refshift);
                // compute whether refsegn * refsegn.dot(path[i+1] + refshift); and path[i+1] - path[i]; are in the same dir
                Point2D to_infinity_and_beyond = path[i+1] - path[i];
                refsegseparates = to_infinity_and_beyond.dot(refsegn * refsegn.dot(path[i+1] + refshift))>0;
            } else if (i==pathlines.size()-1) {
                Point2D to_infinity_and_beyond = path[i] - path[i+1];
                refsegseparates = to_infinity_and_beyond.dot(refsegn * refsegn.dot(path[i] + refshift))>0;
            } else refsegseparates = refsegn.dot(path[i] + refshift) * refsegn.dot(path[i+1] + refshift) < 0;
            // crossing iif each segment/line separates the other
            numcross += refsegseparates && pathseparates;
            
            // closest distance from the point to that segment
            // 1. projection of the point of the line
            Point2D p = pt - n * n.dot(pt + n * pathlines[i].c);
            FloatType closestToSeg = numeric_limits<FloatType>::max();
            bool projwithin = true;
            // 2. Is the projection within the segment limit ? yes => closest
            if (i==0) {
                FloatType xdelta = path[i+1].x - p.x;
                FloatType ydelta = path[i+1].y - p.y;
                // use the more reliable delta
                if (fabs(xdelta)>fabs(ydelta)) {
                    // intersection is valid only if on the half-infinite side of the segment
                    projwithin &= (xdelta * (path[i+1].x - path[i].x)) > 0;
                } else {
                    projwithin &= (ydelta * (path[i+1].y - path[i].y)) > 0;
                }
            } else if (i==pathlines.size()-1) {
                // idem, just infinite on the other side
                FloatType xdelta = path[i].x - p.x;
                FloatType ydelta = path[i].y - p.y;
                if (fabs(xdelta)>fabs(ydelta)) {
                    projwithin &= (xdelta * (path[i].x - path[i+1].x)) > 0;
                } else {
                    projwithin &= (ydelta * (path[i].y - path[i+1].y)) > 0;
                }
            } else {
                // intersection is valid only within the segment boundaries
                projwithin &= (p.x >= min(path[i].x,path[i+1].x)) && (p.y >= min(path[i].y,path[i+1].y));
                projwithin &= (p.x <= max(path[i].x,path[i+1].x)) && (p.y <= max(path[i].y,path[i+1].y));
            }
            if (projwithin) closestToSeg = dist(Point2D(a,b), p);
            else {
                // 3. otherwise closest is the minimum of the distance to the segment ends
                if (i!=0) closestToSeg = dist(Point2D(a,b), Point2D(path[i].x,path[i].y));
                if (i!=pathlines.size()-1) closestToSeg = min(closestToSeg, dist(Point2D(a,b), Point2D(path[i+1].x,path[i+1].y)));
            }
            if (closestToSeg < closestDist) {
                selectedSeg = i;
                closestDist = closestToSeg;
            }
        }
        Point2D n(pathlines[selectedSeg].wx, pathlines[selectedSeg].wy);
        Point2D p = pt - n * n.dot(pt + n * pathlines[selectedSeg].c);
        Point2D delta = pt - p;
        if ((numcross&1)==0) return delta.norm();
        else return -delta.norm();
    }


    // classification in the 2D space
    FloatType classify2D(FloatType a, FloatType b) {
        FloatType condpos, condneg;
        FloatType predpos = classify2D_checkcondnum(a,b,refpt_pos,condpos);
        FloatType predneg = classify2D_checkcondnum(a,b,refpt_neg,condneg);
        // normal nearly aligned = bad conditionning, the lower the dot prod the better
        if (condpos<condneg) return predpos;
        return -predneg;
    }
    
    void project(FloatType* mscdata, FloatType& a, FloatType& b) {
        a = weights_axis1[weights_axis1.size()-1];
        b = weights_axis2[weights_axis2.size()-1];
        for (int d=0; d<weights_axis1.size()-1; ++d) {
            a += weights_axis1[d] * mscdata[d];
            b += weights_axis2[d] * mscdata[d];
        }
    }

    // classification in MSC space
    FloatType classify(FloatType* mscdata) {
        FloatType a,b;
        project(mscdata,a,b);
        return classify2D(a,b);
    }
};

#endif
