// This file contains helper functions and interfaces for reading/writing HDF5
#ifndef included_AMP_HDF5_h
#define included_AMP_HDF5_h

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/ArraySize.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <vector>


// Include the headers and define some basic types
#ifdef AMP_USE_HDF5
    // Using HDF5
    #include "hdf5.h"
#else
// Not using HDF5
typedef int64_t hid_t;
typedef size_t hsize_t;
#endif


namespace AMP::IO {


enum class Compression : uint8_t { None, GZIP, SZIP };


/**
 * \brief Is HDF5 enabled
 * \details This function returns true if HDF5 support is enabled (AMP was compiled with HDF5)
 */
bool HDF5enabled();


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
 * @param[in] compress      Default compression
 * @return                  Return a handle to the file.
 */
hid_t openHDF5( const std::string &filename,
                const char *mode,
                Compression compress = Compression::None );


/**
 * \brief Open an HDF5 file
 * \details This function opens and HDF5 file for reading/writing
 * @param[in] fid           File to open
 * @param[in] printLeaks    Print the resource leaks
 */
void closeHDF5( hid_t fid, bool printLeaks = false );


/**
 * \brief Retrun the the default compression
 * \details This function returns the default compression used when the file was created
 * @param[in] fid           File/Group id
 */
Compression defaultCompression( hid_t fid );


/**
 * \brief Open an HDF5 file
 * \details This function create a chunk for HDF5
 * @param[in] dims          Chunk size
 * @param[in] compress      Compression to use
 * @param[in] objSize       Optional number of bytes of an object
 * @return                  Return a handle to the file.
 */
hid_t createChunk( AMP::ArraySize dims, Compression compress, size_t objSize = 0 );


/**
 * \brief Write a structure to HDF5
 * \details This function writes a C++ class/struct to HDF5.
 *    This is a templated function and users can implement their own data
 *    types by creating explicit instantiations for a given type.
 *    There is no default instantiation except when compiled without HDF5 which is a no-op.
 * @param[in] fid           File or group to write to
 * @param[in] name          The name of the variable
 * @param[in] data          The structure to write
 */
template<class T>
void writeHDF5( hid_t fid, const std::string &name, const T &data );


/**
 * \brief Read a structure from HDF5
 * \details This function reads a C++ class/struct from HDF5.
 *    This is a templated function and users can implement their own data
 *    types by creating explicit instantiations for a given type.
 *    There is no default instantiation except when compiled without HDF5 which is a no-op.
 * @param[in] fid           File or group to read from
 * @param[in] name          The name of the variable
 * @param[out] data         The structure to read
 */
template<class T>
void readHDF5( hid_t fid, const std::string &name, T &data );


/**
 * \brief Read a structure from HDF5
 * \details This function reads a C++ class/struct from HDF5.
 *    This is a templated function and users can implement their own data
 *    types by creating explicit instantiations for a given type.
 *    There is no default instantiation except when compiled without HDF5 which is a no-op.
 * @param[in] fid           File or group to read from
 * @param[in] name          The name of the variable
 * @param[in] comm          The communicator of the object
 */
template<class T>
std::unique_ptr<T> readHDF5( hid_t fid, const std::string &name, AMP_MPI comm = AMP_COMM_SELF );


/**
 * \brief Write data to HDF5
 * \details This function writes a fixed number of bytes from HDF5.
 * @param[in] fid           File or group to write to
 * @param[in] name          The name of the variable
 * @param[in] N_bytes       The number of bytes to write
 * @param[in] data          The data to write
 */
void writeHDF5( hid_t fid, const std::string &name, size_t N_bytes, const void *data );


/**
 * \brief Read data from HDF5
 * \details This function reads a fixed number of bytes from HDF5.
 * @param[in] fid           File or group to write to
 * @param[in] name          The name of the variable
 * @param[in] N_bytes       The number of bytes to write
 * @param[out] data         The data to read
 */
void readHDF5( hid_t fid, const std::string &name, size_t N_bytes, void *data );


/**
 * \brief Check if group exists
 * \details This function checks if an HDF5 group exists in the file
 * @param[in] fid           ID of group or database to read
 * @param[in] name          The name of the group
 */
bool H5Gexists( hid_t fid, const std::string &name );


/**
 * \brief Check if dataset exists
 * \details This function checks if an HDF5 dataset exists in the file
 * @param[in] fid           File to open
 * @param[in] name          The name of the dataset
 */
bool H5Dexists( hid_t fid, const std::string &name );


/**
 * \brief Create a group
 * \details This function creates a new HDF5 group
 * @param[in] fid       File or group to write to
 * @param[in] name      The name of the group
 */
hid_t createGroup( hid_t fid, const std::string &name );


/**
 * \brief Open a group
 * \details This function opens an HDF5 group
 * @param[in] fid       File or group to write to
 * @param[in] name      The name of the group
 */
hid_t openGroup( hid_t fid, const std::string &name );


/**
 * \brief Close a group
 * \details This function closes an HDF5 group
 * @param[in] fid       Group to close
 */
void closeGroup( hid_t fid );


/**
 * \brief Close a group
 * \details This function closes an HDF5 group
 * \return  Returns the list of open objects associated with the file:
 *          [files,datasets,groups,datatypes,attributes]
 * @param[in] fid       Group to close
 */
std::tuple<std::vector<hid_t>,
           std::vector<hid_t>,
           std::vector<hid_t>,
           std::vector<hid_t>,
           std::vector<hid_t>>
openObjects( hid_t fid );


/**
 * \brief Get HDF5 data type
 * \details This function returns the id of the data type
 */
template<class T>
hid_t getHDF5datatype();


} // namespace AMP::IO

#endif
