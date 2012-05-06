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
#ifndef CANUPO_PNG_UTILITY
#define CANUPO_PNG_UTILITY

#include <iostream>
#include <fstream>
#include <vector>

// graphics lib
#include <cairo/cairo.h>

#include "png++/png.hpp"

int ppmwrite(cairo_surface_t *surface, const char* filename) {
    using namespace std;
    int height = cairo_image_surface_get_height(surface);
    int width = cairo_image_surface_get_width(surface);
    int stride = cairo_image_surface_get_stride(surface);
    unsigned char* data = cairo_image_surface_get_data(surface);
    ofstream ppmfile(filename);
    ppmfile << "P3 " << width << " " << height << " " << 255 << endl;
    for (int row = 0; row<height; ++row) {
        for (int col = 0; col<width*4; col+=4) {
            ppmfile << (int)data[col+2] << " " << (int)data[col+1] << " " << (int)data[col+0] << " ";
        }
        data += stride;
    }
}

/* bugged under mingw
cairo_status_t png_copier(void *closure, const unsigned char *data, unsigned int length) {
    std::vector<char>* pngdata = (std::vector<char>*)closure;
    int cursize = pngdata->size();
    pngdata->resize(cursize + length); // use reserve() before, or this will be slow
    memcpy(&(*pngdata)[cursize], data, length);
    return CAIRO_STATUS_SUCCESS;
}
*/
struct pngpp_ostream {
    std::vector<char> data;
    void write(char const* c, size_t s) {
        for (int i=0; i<s; ++i) data.push_back(c[i]);
    }
    void flush() {}
    bool good() {return true;}
};

// assumes the surface is in ARGB mode
void surface_to_png(cairo_surface_t *surface, std::vector<char>& pngdata) {
    int height = cairo_image_surface_get_height(surface);
    int width = cairo_image_surface_get_width(surface);
    int stride = cairo_image_surface_get_stride(surface);
    unsigned char* data = cairo_image_surface_get_data(surface);
    png::image<png::rgb_pixel> image(width, height);
    int offset = 0;
    for (size_t y = 0; y < image.get_height(); ++y) {
        int rowoffset = offset;
        for (size_t x = 0; x < image.get_width(); ++x) {
            image[y][x] = png::rgb_pixel(data[rowoffset + 2], data[rowoffset + 1], data[rowoffset + 0]);
            rowoffset += 4;
        }
        offset += stride;
    }
    pngpp_ostream pngsstream;
    image.write_stream(pngsstream);
    pngdata.swap(pngsstream.data);
}

#endif
