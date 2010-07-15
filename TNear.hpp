//*
//*  TNear.h
//*  NearTree
//*
//*  Copyright 2001, 2008 Larry Andrews.  All rights reserved
//*  Revised 12 Dec 2008 for sourceforge release -- H. J. Bernstein

//*  Changes by Nicolas Brodu, Jan 09, nicolas.brodu@at@numerimoire.net
//*  - Templatised the numeric type for the distances, default to double
//*  - Implemented a template meta-programming check for a "distance_to"
//*    optional member function so as to avoid temporaries. Global
//*    distance functions might also be used, ex for std::vector.
//*  - replaced #define by inner functions to avoid namespace pollution
//*  - replaced DBL_MIN by C++ equivalent, allowing template genericity

//**********************************************************************
//*                                                                    *
//* YOU MAY REDISTRIBUTE NearTree UNDER THE TERMS OF THE LGPL          *
//*                                                                    *
//**********************************************************************/

//************************* LGPL NOTICES *******************************
//*                                                                    *
//* This library is free software; you can redistribute it and/or      *
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

//  This is a revised release of 
//  template <typename T> class CNearTree;
//
// Nearest Neighbor algorithm after Kalantari and McDonald,
// (IEEE Transactions on Software Engineering, v. SE-9, pp.
//    631-634,1983)
//  modified to use recursion instead of a double-linked tree
//  and simplified so that it does a bit less checking for
//  things like is the distance to the right less than the
//  distance to the left; it was found that these checks made little
//  to no difference in timing.


// This template is used to contain a collection of objects. After the
// collection has been loaded into this structure, it can be quickly
// queried for which object is "closest" to some probe object of the
// same type. The major restriction on applicability of the near-tree
// is that the algorithm only works if the objects obey the triangle
// inequality. The triangle rule states that the length of any side of
// a triangle cannot exceed the sum of the lengths of the other two sides.


// The user of this class needs to provide at least the following
// functionality for the template to work. For the built-in
// numerics of C++, they are provided by the system.

//    Either:
//    .  operator NumericType( ); // conversion constructor from the templated class
//                                 to the user desired numeric type (double by default)
//    and                          (usually will return a "length")
//    .  operator- ( );         // geometrical (vector) difference of two objects

//    Or alternatively:
//    .  NumericType T::distance_to(const T& other) const; // member function of that
//                                                         // exact name and signature
//    or
//    .  NumericType a_dist_fun(const T& element1, const T& element2); // any global function
//    that last form is especially handy for built-in types, like std::vector

//    a copy constructor
//    a constructor would be nice
//    a destructor would be nice

// The provided interface is:
//
//    #include "TNear.h"
//
//    CNearTree( void )   // constructor
//       instantiated by something like:      CNearTree <v> vTree;
//       for some type v
//
//    void Insert( T& t )
//       where t is an object of the type v
//
//    bool NearestNeighbor ( const NumericType& dRadius,  T& tClosest,   const T& t ) const
//       dRadius is the largest radius within which to search; make it
//          very large if you want to include every point that was loaded; dRadius
//          is returned as the closest distance to the probe (or the search radius
//          if nothing is found)
//       tClosest is returned as the object that was found closest to the probe
//          point (if any were within radius dRadius of the probe)
//       t is the probe point, used to search in the group of points Insert'ed
//       return value is true if some object was found within the search radius, false otherwise
//
//    bool FarthestNeighbor ( T& tFarthest,   const T& t ) const
//       tFarthest is returned as the object that was found farthest to the probe
//          point
//       t is the probe point, used to search in the group of points Insert'ed
//       return value is true if some object was found, false otherwise
//
//    long FindInSphere ( const NumericType& dRadius,  std::vector<  T >& tClosest,   const T& t ) const
//       dRadius is the radius within which to search; make it very large if you want to
//           include every point that was loaded;
//       tClosest is returned as the vector of objects that were found within a radius dRadius
//          of the probe point
//       t is the probe point, used to search in the group of points Insert'ed
//       return value is the number of objects found within the search radius
//
//    ~CNearTree( void )  // destructor
//       invoked by  vTree.CNeartree<v>::~CNearTree
//       for an object vTree of some type v

// So a complete program is:
//
// #include "TNear.h"
// #include <cstdio>
// void main()
// {
//   CNearTree< double > dT;
//   double dNear;
//   dT.Insert( 1.5 );
//   if ( dT.NearestNeighbor( 10000.0,   dNear,  2.0 )) printf( "%f\n",double(dNear-2.0) );
// }
//
// and it should print 0.5 (that's how for 2.0 is from 1.5)
//
//
//-------------------------------------------------------------------------


#if !defined(TNEAR_H_INCLUDED)
#define TNEAR_H_INCLUDED

#include <limits>
#include <cmath>
#include <vector>

// Some template meta-programming to compute the distance between two elements
// according to the user-specified functions

namespace CNearTree_helpers {

#ifndef CNEARTREE_NO_PARTIAL_TEMPLATE
template <typename NumericType>
struct AbsVal {
    template <typename NT, int is_integer = 0>
    struct AbsValNT {
        inline static NT abs(NT x) {
            return std::fabs(x);
        }
    };
    template <typename NT>
    struct AbsValNT<NT,1> {
        inline static NT abs(NT x) {
            return std::abs(x);
        }
    };
    inline static NumericType abs(NumericType x) {
        return AbsValNT<NumericType,std::numeric_limits<NumericType>::is_integer>::abs(x);
    }
};

template <class C, typename NumericType>
struct DistanceTo {
    template <class BaseClass, typename FunctionType>
    struct Check_member_distance_to {
        template <FunctionType> struct Finder;
        template <class Base> static long sfinaeOverload(Finder<&Base::distance_to> *);
        template <class Base> static char sfinaeOverload(...);
        enum { found = sizeof(sfinaeOverload<BaseClass>(0)) == sizeof(long) };
    };

    template <typename X>
    struct Check_supports_member_func {
        template <class Base> static long sfinaeOverload(void(Base::*)(void));
        template <class Base> static char sfinaeOverload(...);
        enum { found = sizeof(sfinaeOverload<X>(0)) == sizeof(long) };
    };

    template <typename X, int is_class = 0>
    struct Selector_supports_member_func {
        enum { found = 0 };
    };
    template <typename X>
    struct Selector_supports_member_func<X,1> {
        enum { found = Check_member_distance_to<X, NumericType (X::*)(const X&) const>::found };
    };

    template <class BaseClass, int dist_method> struct Selector {
        inline static NumericType distance_to(const BaseClass& a,const BaseClass& b) {
            return AbsVal<NumericType>::abs( NumericType(a - b) );
        }
    };
    template <class BaseClass> struct Selector<BaseClass,1> {
        inline static NumericType distance_to(const BaseClass& a,const BaseClass& b) {
            return a.distance_to(b);
        }
    };

    inline static NumericType distance_to(const C& a, const C& b) {
        return Selector<C, Selector_supports_member_func<C, Check_supports_member_func<C>::found>::found >::distance_to(a,b);
    }
};
#else
// Support for old compilers - fallback to default CNearTree feature of requiring operators
template <class C, typename NumericType>
struct DistanceTo {
    inline static NumericType distance_to(const C& a, const C& b) {
        return NumericType(std::numeric_limits<NumericType>::is_integer ? std::abs(NumericType(a - b)) : std::fabs(NumericType(a - b)));
    }
};
#endif

template <class C, typename NumericType>
struct DistTypeMaker {
    typedef NumericType (*Distance)(const C&,const C&);
};

template <typename NumericType>
struct Lowest {
    // ggrmmbbl, standard behaviour differs between float and integer types...
    inline static NumericType lowest() {
        if (std::numeric_limits<NumericType>::is_integer) return std::numeric_limits<NumericType>::min();
        return -std::numeric_limits<NumericType>::max();
    }
};

}


/// The main NEAR tree class
/// T: type for the elements in the NEAR tree
/// NumericType: the numeric type for computing distances. Defaults to double
/// distance_to: A global or static distance function:  NumericType distance_to(const T&,const T&)
///              - element1.distance_to(element2) if T::distance_to exists
///              - NumericType(element1 - element2) otherwise
///              Tip: use a global function for vectors and other built-in type

template <
    typename T,
    typename NumericType = double,
    typename CNearTree_helpers::DistTypeMaker<T,NumericType>::Distance distance_to = &CNearTree_helpers::DistanceTo<T,NumericType>::distance_to
>
class CNearTree
{

   // Insert copies the input objects into a binary NEAR tree. When a node has
   // two entries, a descending node is used or created. The current datum is
   // put into the branch descending from the nearer of the two
   // objects in the current node.

   // NearestNeighbor retrieves the object nearest to some probe by descending
   // the tree to search out the appropriate object. Speed is gained
   // by pruning the tree if there can be no data below that are
   // nearer than the best so far found.

   // The tree is built in time O(n log n), and retrievals take place in
   // time O(log n).


    T *           m_ptLeft;         // left object (of type T) stored in this node
    T *           m_ptRight;        // right object (of type T) stored in this node
    NumericType   m_dMaxLeft;       // longest distance from the left object to
                                    // anything below it in the tree
    NumericType   m_dMaxRight;      // longest distance from the right object to
                                    // anything below it in the tree
    CNearTree *   m_pLeftBranch;    // tree descending from the left object
    CNearTree *   m_pRightBranch;   // tree descending from the right object


#ifdef CNEARTREE_SAFE_TRIANG
    inline static bool TRIANG(NumericType a, NumericType b, NumericType c) {
        return ((b+c)-a >= 0) || (b-(a-c) >= 0) || (c-(a-b) >= 0);
    }
#else
    inline static bool TRIANG(NumericType a, NumericType b, NumericType c) {
        return b+c-a >= 0;
    }
#endif


public:


//=======================================================================
//  CNearTree ( )
//
//  Default constructor for class CNearTree
//  creates an empty tree with no right or left node and with the dMax-below
//  set to negative values so that any match found will be stored since it will
//  greater than the negative value
//
//=======================================================================

   CNearTree(void)  // constructor
   {
      m_ptLeft       = 0;
      m_ptRight      = 0;
      m_pLeftBranch  = 0;
      m_pRightBranch = 0;
      m_dMaxLeft     = CNearTree_helpers::Lowest<NumericType>::lowest();
      m_dMaxRight    = CNearTree_helpers::Lowest<NumericType>::lowest();
   }  //  CNearTree constructor

//=======================================================================
//  ~CNearTree ( )
//
//  Destructor for class CNearTree
//
//=======================================================================

   ~CNearTree(void)  // destructor
   {
      delete m_pLeftBranch  ;  m_pLeftBranch  =0;
      delete m_pRightBranch ;  m_pRightBranch =0;
      delete m_ptLeft       ;  m_ptLeft       =0;
      delete m_ptRight      ;  m_ptRight      =0;

      m_dMaxLeft     = CNearTree_helpers::Lowest<NumericType>::lowest();
      m_dMaxRight    = CNearTree_helpers::Lowest<NumericType>::lowest();
   }  //  ~CNearTree

//=======================================================================
//  empty ( )
//
//  Test for an empty CNearTree
//
//=======================================================================
    
    bool empty( ) const
    {
        return( m_ptLeft == 0 );
    }
    
//=======================================================================
//  void Insert ( const T& t )
//
//  Function to insert some "point" as an object into a CNearTree for
//  later searching
//
//     t is an object of the templated type which is to be inserted into a
//     Neartree
//
//  Three possibilities exist: put the datum into the left
//  postion (first test),into the right position, or else
//  into a node descending from the nearer of those positions
//  when they are both already used.
//
//=======================================================================
   void Insert( const T& t )
   {
      // do a bit of precomputing if it is possible so that we can
      // reduce the number of calls to operator 'NumericType' as much as possible;
      // 'NumericType' might use square roots in some cases
      NumericType dTempRight =  0;
      NumericType dTempLeft  =  0;

      if ( m_ptRight  != 0 )
      {
         dTempRight  = distance_to(t, *m_ptRight);
         dTempLeft   = distance_to(t, *m_ptLeft );
      }

      if ( m_ptLeft == 0 )
      {
         m_ptLeft = new T( t );
      }
      else if ( m_ptRight == 0 )
      {
         m_ptRight   = new T( t );
      }
      else if ( dTempLeft > dTempRight )
      {
         if ( m_pRightBranch == 0 ) m_pRightBranch = new CNearTree;
         // note that the next line assumes that m_dMaxRight is negative for a new node
         if ( m_dMaxRight < dTempRight ) m_dMaxRight = dTempRight;
         m_pRightBranch->Insert( t );
      }
      else  // ((NumericType)(t - *m_tLeft) <= (NumericType)(t - *m_tRight) )
      {
         if ( m_pLeftBranch  == 0 ) m_pLeftBranch  = new CNearTree;
         // note that the next line assumes that m_dMaxLeft is negative for a new node
         if ( m_dMaxLeft < dTempLeft ) m_dMaxLeft  = dTempLeft;
         m_pLeftBranch->Insert( t );
      }

   }  //  Insert

//=======================================================================
//  bool NearestNeighbor ( const NumericType& dRadius,  T& tClosest,   const T& t ) const
//
//  Function to search a Neartree for the object closest to some probe point, t. This function
//  is only here so that the function Nearest can be called without having the radius const.
//  This was necessary because Nearest is recursive, but needs to keep the current radius.
//
//    dRadius is the maximum search radius - any point farther than dRadius from the probe
//             point will be ignored
//    tClosest is an object of the templated type and is the returned nearest point
//             to the probe point that can be found in the Neartree
//    t  is the probe point
//
//    the return value is true only if a point was found
//
//=======================================================================
   bool NearestNeighbor ( const NumericType& dRadius,  T& tClosest,   const T& t, bool acceptQueryPoint = false ) const
   {
      if( dRadius < NumericType(0) ) 
      {
         return( false );
      }
      else if( this->empty( ) )
      {
         return( false );
      }
      else
      {
        NumericType dSearchRadius = dRadius;
        return ( const_cast<CNearTree*>(this)->Nearest ( dSearchRadius, tClosest, t, acceptQueryPoint ) );
      }
   }  //  NearestNeighbor

//=======================================================================
//  bool FarthestNeighbor ( const NumericType& dRadius,  T& tClosest,   const T& t ) const
//
//  Function to search a Neartree for the object closest to some probe point, t. This function
//  is only here so that the function FarthestNeighbor can be called without the user
//  having to input a search radius and so the search radius can be guaranteed to be
//  negative at the start.
//
//    tFarthest is an object of the templated type and is the returned farthest point
//             from the probe point that can be found in the Neartree
//    t  is the probe point
//
//    the return value is true only if a point was found (should only be false for
//             an empty tree)
//
//=======================================================================
   bool FarthestNeighbor ( T& tFarthest,   const T& t ) const
   {
      if( this->empty( ) )
      {
         return( false );
      }
      else
      {
        NumericType dSearchRadius = CNearTree_helpers::Lowest<NumericType>::lowest();
        return ( const_cast<CNearTree*>(this)->Farthest ( dSearchRadius, tFarthest, t ) );
      }
   }  //  FarthestNeighbor

//=======================================================================
//  long FindInSphere ( const NumericType& dRadius,  std::vector<  T >& tClosest,   const T& t ) const
//
//  Function to search a Neartree for the set of objects closer to some probe point, t,
//  than dRadius. This is only here so that tClosest can be cleared before starting the work.
//
//    dRadius is the maximum search radius - any point farther than dRadius from the probe
//             point will be ignored
//    tClosest is a vector of objects of the templated type and is the returned set of nearest points
//             to the probe point that can be found in the Neartree
//    t  is the probe point
//    return value is the number of points found within dRadius of the probe point
//
//=======================================================================
   long FindInSphere ( const NumericType& dRadius,  std::vector<  T >& tClosest,   const T& t ) const
   {
      // clear the contents of the return vector so that things don't accidentally accumulate
      tClosest.clear( );
      return ( const_cast<CNearTree*>(this)->InSphere( dRadius, tClosest, t ) );
   }  //  FindInSphere

   private:
//=======================================================================
//  long InSphere ( const NumericType& dRadius,  std::vector<  T >& tClosest,   const T& t ) const
//
//  Private function to search a Neartree for the object closest to some probe point, t.
//  This function is only called by FindInSphere.
//
//    dRadius is the search radius
//    tClosest is a vector of objects of the templated type found within dRadius of the
//         probe point
//    t  is the probe point
//    the return value is the number of points found within dRadius of the probe
//
//=======================================================================
    long InSphere ( const NumericType& dRadius,  std::vector<  T >& tClosest,   const T& t ) const
    {
        std::vector <CNearTree<T,NumericType,distance_to>* > sStack;
        long lReturn = 0;
        enum  { left, right, end } eDir;
        eDir = left; // examine the left nodes first
        CNearTree* pt = const_cast<CNearTree*>(this);
        if (!(pt->m_ptLeft)) return false; // test for empty
        while ( ! ( eDir == end && sStack.empty( ) ) )
        {
            if ( eDir == right )
            {
                const NumericType dDR = distance_to( t, *(pt->m_ptRight) );
                if ( dDR <= dRadius )
                {
                    ++lReturn;
                    tClosest.push_back( *pt->m_ptRight);
                }
                if ( pt->m_pRightBranch != 0 && (TRIANG(dDR,pt->m_dMaxRight,dRadius)))
                { // we did the left and now we finished the right, go down
                    pt = pt->m_pRightBranch;
                    eDir = left;
                }
                else
                {
                    eDir = end;
                }
            }
            if ( eDir == left )
            {
                const NumericType dDL = distance_to( t, *(pt->m_ptLeft) );
                if ( dDL <= dRadius )
                {
                    ++lReturn;
                    tClosest.push_back( *pt->m_ptLeft);
                }
                if ( pt->m_ptRight != 0 ) // only stack if there's a right object
                {
                    sStack.push_back( pt );
                }
                if ( pt->m_pLeftBranch != 0 && (TRIANG(dDL,pt->m_dMaxLeft,dRadius)))
                { // we did the left, go down
                    pt = pt->m_pLeftBranch;
                }
                else
                {
                    eDir = end;
                }
            }
            
            if ( eDir == end && !sStack.empty( ) )
            {
                pt = sStack.back( );
                sStack.pop_back( );
                eDir = right;
            }
        }
        while ( !sStack.empty( ) ) // for safety !!!
            sStack.pop_back( );
        return ( lReturn );
    }  //  InSphere

//=======================================================================
//  bool Nearest ( NumericType& dRadius,  T& tClosest,   const T& t ) const
//
//  Private function to search a Neartree for the object closest to some probe point, t.
//  This function is only called by NearestNeighbor.
//
//    dRadius is the smallest currently known distance of an object from the probe point.
//    tClosest is an object of the templated type and is the returned closest point
//             to the probe point that can be found in the Neartree
//    t  is the probe point
//    the return value is true only if a point was found within dRadius
//
//=======================================================================
   bool Nearest ( NumericType& dRadius,  T& tClosest,   const T& t , bool acceptQueryPoint) const
   {
      std::vector <CNearTree<T,NumericType,distance_to>* > sStack;;
      enum  { left, right, end } eDir;
      eDir = left; // examine the left nodes first
      CNearTree* pt = const_cast<CNearTree*>(this);
      T* pClosest = 0;
      if (!(pt->m_ptLeft)) return false; // test for empty
      while ( ! ( eDir == end && sStack.empty( ) ) )
      {
         if ( eDir == right )
         {
            const NumericType dDR = distance_to(t, *(pt->m_ptRight) );
            if (( dDR < dRadius ) && (acceptQueryPoint || (dDR!=0)))
            {
               dRadius = dDR;
               pClosest = pt->m_ptRight;
            }
            if ( pt->m_pRightBranch != 0 && (TRIANG(dDR,pt->m_dMaxRight,dRadius)))
            { // we did the left and now we finished the right, go down
               pt = pt->m_pRightBranch;
               eDir = left;
            }
            else
            {
               eDir = end;
            }
         }
         if ( eDir == left )
         {
            const NumericType dDL = distance_to(t, *(pt->m_ptLeft) );
            if (( dDL < dRadius ) && (acceptQueryPoint || (dDL!=0)))
            {
               dRadius = dDL;
               pClosest = pt->m_ptLeft;
            }
            if ( pt->m_ptRight != 0 ) // only stack if there's a right object
            {
               sStack.push_back( pt );
            }
            if ( pt->m_pLeftBranch != 0 && (TRIANG(dDL,pt->m_dMaxLeft,dRadius)))
            { // we did the left, go down
               pt = pt->m_pLeftBranch;
            }
            else
            {
               eDir = end;
            }
         }

         if ( eDir == end && !sStack.empty( ) )
         {
            pt = sStack.back( );
            sStack.pop_back( );
            eDir = right;
         }
      }
      while ( !sStack.empty( ) ) // for safety !!!
         sStack.pop_back( );
      if ( pClosest != 0 )
         tClosest = *pClosest;
      return ( pClosest != 0 );
   };   // Nearest

//=======================================================================
//  bool Farthest ( NumericType& dRadius,  T& tFarthest,   const T& t ) const
//
//  Private function to search a Neartree for the object farthest from some probe point, t.
//  This function is only called by FarthestNeighbor.
//
//    dRadius is the largest currently known distance of an object from the probe point.
//    tFarthest is an object of the templated type and is the returned farthest point
//             from the probe point that can be found in the Neartree
//    t  is the probe point
//    the return value is true only if a point was found (should only be false for
//             an empty tree)
//
//=======================================================================
   bool Farthest ( NumericType& dRadius,  T& tFarthest,   const T& t )
   {
      std::vector <CNearTree<T,NumericType,distance_to>* > sStack;;
      enum  { left, right, end } eDir;
      eDir = left; // examine the left nodes first
      CNearTree* pt = this;
      T* pFarthest = 0;
      if (!(pt->m_ptLeft)) return false; // test for empty
      while ( ! ( eDir == end && sStack.empty( ) ) )
      {
         if ( eDir == right )
         {
            const NumericType dDR = distance_to( t, *(pt->m_ptRight) );
            if ( dDR >= dRadius )
            {
               dRadius = dDR;
               pFarthest = pt->m_ptRight;
            }
            if ( pt->m_pRightBranch != 0 && TRIANG(dRadius,dDR,pt->m_dMaxRight))
            { // we did the left and now we finished the right, go down
               pt = pt->m_pRightBranch;
               eDir = left;
            }
            else
            {
               eDir = end;
            }
         }
         if ( eDir == left )
         {
            const NumericType dDL = distance_to( t, *(pt->m_ptLeft) );
            if ( dDL >= dRadius )
            {
               dRadius = dDL;
               pFarthest = pt->m_ptLeft;
            }
            if ( pt->m_ptRight != 0 ) // only stack if there's a right object
            {
               sStack.push_back( pt );
            }
            if ( pt->m_pLeftBranch != 0 && TRIANG(dRadius,dDL,pt->m_dMaxLeft) )
            { // we did the left, go down
               pt = pt->m_pLeftBranch;
            }
            else
            {
               eDir = end;
            }
      }

         if ( eDir == end && !sStack.empty( ) )
         {
            pt = sStack.back( );
            sStack.pop_back( );
            eDir = right;
         }
      }
      while ( !sStack.empty( ) ) // for safety !!!
         sStack.pop_back( );
      if ( pFarthest != 0 )
         tFarthest = *pFarthest;
      return ( pFarthest != 0 );
   };   // Farthest

public:

    // NB100714: quick and dirty forward iterator. Non-standard
    struct Sequence {
        std::vector<CNearTree*> parents;
        CNearTree* node;
        bool leftDone, rightDone;
        inline Sequence(CNearTree* root) : node(root), leftDone(false), rightDone(false) {}
        bool hasNext() {
            // see next() for the logic
            if (!node) return false;
            if (!leftDone && node->m_ptLeft) return true;
            if (!rightDone && node->m_ptRight) return true;
            // recursion condition : there is some point down there by construction
            if (node->m_pLeftBranch || node->m_pRightBranch) return true;
            // leaf node, no parent = the end
            if (parents.empty()) return false;
            return true; // recursion condition
        }
        T* next() {
            if (!node) return 0;
            // process left point first.
            if (!leftDone) {
                if (node->m_ptLeft) {leftDone = true; return node->m_ptLeft;}
                // not left done, but nothing on left anyway => go to right, if any
                leftDone = true;
            }
            if (!rightDone) {
                if (node->m_ptRight) {rightDone = true; return node->m_ptRight;}
                // not right done, but nothing on right anyway => go to recurse now
                rightDone = true;
            }
            // recurse to left branch, then right branch
            if (node->m_pLeftBranch) {
                // we only push in stack parents that have unprocessed right branch
                // otherwise, jump directly to next parent, if any
                if (node->m_pRightBranch) parents.push_back(node);
                leftDone = false;
                rightDone = false;
                node = node->m_pLeftBranch;
                // recurse now
                return next();
            }
            if (node->m_pRightBranch) {
                // don't push this node as parent, as we'll have done it completely on return
                // but recurse to right branch. On return, go directly to an unprocessed parent
                leftDone = false;
                rightDone = false;
                node = node->m_pRightBranch;
                // recurse now
                return next();
            }
            // leaf node at this point. Unpop parent, if any
            if (parents.empty()) {node = 0; return 0;} // end, user has ignored !hasNext()...
            node = parents.back(); parents.pop_back();
            // parents have both left and right points done, always, when poping back
            // as well as left branch completely explored. then get down in tree on the right.
            // Parents are also ONLY pushed if they have unexplored right branch
            // and if it exists, then there IS by construction some point down there
            leftDone = false;
            rightDone = false;
            node = node->m_pRightBranch;
            // recurse now
            return next();
        }
    };

    Sequence sequence() {
        return Sequence(this);
    }

}; // template class TNear

#endif // !defined(TNEAR_H_INCLUDED)
