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
/* Work in progress: centralize the msc/prm/etc... routines to ensure
   no inconsistency in case of file format change
*/

#ifndef CANUPO_HELPERS_HPP
#define CANUPO_HELPERS_HPP

#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/unistd.h>

#ifndef NO_MMAP
#include <sys/mman.h>
#endif

// For the FloatType def => TODO move that def here ?
#include "points.hpp"

bool fpeq(FloatType a, FloatType b) {
    static const FloatType epsilon = 1e-6;
    if (b==0) return fabs(a)<epsilon;
    FloatType ratio = a/b;
    return ratio>1-epsilon && ratio<1+epsilon;
}

#ifdef NO_MMAP
struct MSCFile {
    std::ifstream realfile;
    MSCFile(const char* name) {
        realfile.open(name, std::ifstream::binary);
    }
    ~MSCFile() {realfile.close();}
    template<typename T> void read(T& value) {
        realfile.read((char*)&value, sizeof(T));
    }
};
#else
struct MSCFile {
    int fd;
    char* memzone;
    off_t zone_size;
    off_t offset;
    MSCFile(const char* name) {
        fd = open(name, O_RDONLY);
        struct stat file_stats;
        fstat(fd, &file_stats);
        zone_size = file_stats.st_size;
        memzone = (char*)mmap(0,zone_size,PROT_READ,MAP_SHARED,fd,0);
        if (memzone==(char*)(-1)) {
            close(fd);
            perror("Error with mmap");
            exit(1);
        }
        offset = 0;
    }
    ~MSCFile() {
        munmap(memzone, zone_size); close(fd);
    }
    template<typename T> void read(T& value) {
        value = *reinterpret_cast<T*>(&memzone[offset]);
        offset += sizeof(T);
    }
};
#endif


// if vector is empty, fill it
// otherwise check the vectors match
int read_msc_header(MSCFile& mscfile, std::vector<FloatType>& scales, int& ptnparams) {
    using namespace std;
    int npts;
    mscfile.read(npts);
    if (npts<=0) {
        cerr << "invalid msc file (negative or null number of points)" << endl;
        exit(1);
    }
    
    int nscales_thisfile;
    mscfile.read(nscales_thisfile);
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
    for (int si=0; si<nscales_thisfile; ++si) mscfile.read(scales_thisfile[si]);

    // all files must be consistant
    if (scales.size() == 0) {
        scales = scales_thisfile;
    } else {
        if (scales.size() != nscales_thisfile) {
            cerr<<"input file mismatch: "<<endl; exit(1);
        }
        for (int si=0; si<scales.size(); ++si) if (!fpeq(scales[si],scales_thisfile[si])) {cerr<<"input file mismatch: "<<endl; exit(1);}
    }
    
    mscfile.read(ptnparams);

    return npts;
}

void read_msc_data(MSCFile& mscfile, int nscales, int npts, FloatType* data, int ptnparams, bool convert_from_tri_to_2D = false) {
    for (int pt=0; pt<npts; ++pt) {
        FloatType param;
        // we do not care for the point coordinates and other parameters
        for (int i=0; i<ptnparams; ++i) {
            mscfile.read(param);
        }
        for (int s=0; s<nscales; ++s) {
            mscfile.read(data[s*2]);
            mscfile.read(data[s*2+1]);
            if (convert_from_tri_to_2D) {
                FloatType c = 1 - data[s*2] - data[s*2+1];
                FloatType x = data[s*2+1] + c / 2;
                FloatType y = c * sqrt(3)/2;
                data[s*2] = x;
                data[s*2+1] = y;
            }
        }
        // we do not care for number of neighbors and average dist between nearest neighbors
        int fooi;
        for (int i=0; i<nscales; ++i) mscfile.read(fooi);
        data += nscales*2;
    }
}


#endif
