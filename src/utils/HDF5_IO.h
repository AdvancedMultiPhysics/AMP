// This file contains helper functions and interfaces for reading/writing HDF5
#ifndef included_AMP_HDF5_h
#define included_AMP_HDF5_h

#include "AMP/utils/ArraySize.h"
#include "AMP/utils/string_view.h"

#include <cstring>


namespace AMP {


// Include the headers and define some basic types
#ifdef USE_HDF5
// Using HDF5
#include "hdf5.h"
#else
// Not using HDF5
typedef int hid_t;
typedef size_t hsize_t;
#endif


enum class Compression : uint8_t { None, GZIP, SZIP };


/**
 * \brief Open an HDF5 file
 * \details This function opens and HDF5 file for reading/writing.
 *     Once complete, we must close the file using closeHDF5
 * @param[in] filename  File to open
 * @param[in] mode      C string containing a file access mode. It can be:
 *                      "r"    read: Open file for input operations. The file must exist.
 *                      "w"    write: Create an empty file for output operations.
 *                          If a file with the same name already exists, its contents
 *                          are discarded and the file is treated as a new empty file.
 *                      "rw" read+write: Open file for reading and writing.  The file must exist.
 * @param[in] compress  Default compression
 * @return              Return a handle to the file.
 */
hid_t openHDF5( const AMP::string_view &filename,
                const char *mode,
                Compression compress = Compression::None );


/**
 * \brief Open an HDF5 file
 * \details This function opens and HDF5 file for reading/writing
 * @param[in] fid       File to open
 */
void closeHDF5( hid_t fid );


/**
 * \brief Retrun the the default compression
 * \details This function returns the default compression used when the file was created
 * @param[in] fid       File/Group id
 */
Compression defaultCompression( hid_t fid );


/**
 * \brief Open an HDF5 file
 * \details This function create a chunk for HDF5
 * @param[in] dims      Chunk size
 * @param[in] compress  Compression to use
 * @return              Return a handle to the file.
 */
hid_t createChunk( const std::vector<hsize_t> &dims, Compression compress );


/**
 * \brief Write a structure to HDF5
 * \details This function writes a C++ class/struct to HDF5.
 *    This is a templated function and users can impliment their own data
 *    types by creating explicit instantiations for a given type.
 *    There is no default instantiation except when compiled without HDF5 which is a no-op.
 * @param[in] fid       File or group to write to
 * @param[in] name      The name of the variable
 * @param[in] data      The structure to write
 */
template<class T>
void writeHDF5( hid_t fid, const AMP::string_view &name, const T &data );


/**
 * \brief Read a structure from HDF5
 * \details This function reads a C++ class/struct from HDF5.
 *    This is a templated function and users can impliment their own data
 *    types by creating explicit instantiations for a given type.
 *    There is no default instantiation except when compiled without HDF5 which is a no-op.
 * @param[in] fid       File or group to read from
 * @param[in] name      The name of the variable
 * @param[out] data     The structure to read
 */
template<class T>
void readHDF5( hid_t fid, const AMP::string_view &name, T &data );


/**
 * \brief Check if group exists
 * \details This function checks if an HDF5 group exists in the file
 * @param[in] fid       ID of group or database to read
 * @param[in] name      The name of the group
 */
bool H5Gexists( hid_t fid, const AMP::string_view &name );


/**
 * \brief Check if dataset exists
 * \details This function checks if an HDF5 dataset exists in the file
 * @param[in] fid       File to open
 * @param[in] name      The name of the dataset
 */
bool H5Dexists( hid_t fid, const AMP::string_view &name );


/**
 * \brief Get HDF5 data type
 * \details This function returns the id of the data type
 */
template<class T>
hid_t getHDF5datatype();


// Default no-op implimentations for use without HDF5
// clang-format off
#ifndef USE_HDF5
template<class T> void readHDF5( hid_t, const AMP::string_view&, T& ) {}
template<class T> void writeHDF5( hid_t, const AMP::string_view&, const T& ) {}
template<class T> void readHDF5Array( hid_t, const AMP::string_view&, AMP::Array<T>& ) {}
template<class T> void writeHDF5Array( hid_t, const AMP::string_view&, const AMP::Array<T>& ) {}
template<class T> hid_t getHDF5datatype() { return 0; }
#endif
// clang-format on


} // namespace AMP

#endif