// This file contains helper functions and interfaces for reading/writing HDF5
#ifndef included_BMP_Writer_h
#define included_BMP_Writer_h

#include "AMP/utils/Array.h"


namespace AMP::Graphics {


/**
 * \brief Write an array to a .bmp
 * \details This function writes a scaled image as a .bmp file
 * @param[in] file      The file name to write
 * @param[in] data      The array with data
 */
void writeBMP( const std::string &file, const AMP::Array<double> &data );


/**
 * \brief Write an array to a .bmp
 * \details This function writes an image as a .bmp file
 * @param[in] file      The file name to write
 * @param[in] data      The array with data
 */
void writeBMP( const std::string &file, const AMP::Array<uint8_t> &data );


} // namespace AMP::Graphics

#endif
