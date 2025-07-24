// This file contains helper functions and interfaces for reading/writing HDF5
#ifndef included_AMP_HDF5_Class_h
#define included_AMP_HDF5_Class_h

#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "AMP/AMP_TPLs.h"
#include "AMP/IO/HDF.h"
#include "AMP/utils/Array.h"


namespace AMP::IO {


//! Class to wrap HDF5 data
class HDF5data : public std::enable_shared_from_this<HDF5data>
{
public:
    //! Virtual destructor
    virtual ~HDF5data() {}

    //! Return a string identifying the class
    virtual std::string type() const = 0;

    //! Number of entries
    virtual size_t size() const = 0;

    //! Get the data ith block and jth child
    virtual std::shared_ptr<HDF5data> getData( size_t i, const std::string_view &name ) = 0;

    //! Get the data ith block and jth child
    std::shared_ptr<const HDF5data> getData( size_t i, const std::string_view &name ) const;

    //! Get the variable names
    virtual std::vector<std::string> getNames() const = 0;

    //! Get the data
    virtual AMP::ArraySize getDataSize() const = 0;

    //! Get the data
    template<class TYPE>
    void getData( AMP::Array<TYPE> &data ) const;

    //! Print information about the data
    virtual void print( int level = 1, const std::string_view &prefix = "" ) const = 0;

    //! Return the name of the variable
    inline const std::string &name() const { return d_name; }

    //! Check if two objects are equal
    virtual bool operator==( const HDF5data &rhs ) const = 0;

    //! Check if two objects are not equal
    inline bool operator!=( const HDF5data &rhs ) const { return !operator==( rhs ); }


protected:
    HDF5data( hid_t fid, const std::string_view &name ) : d_fid( fid ), d_name( name ) {}

    hid_t d_fid;
    std::string d_name;
};


/**
 * \brief Read data from HDF5
 * \details This function reads arbitrary from HDF5.
 *    The user then requests the individual data from the class.
 * @param[in] fid       File or group to read from
 * @param[in] name      The name of the variable
 * @return              The structure to read
 */
std::unique_ptr<HDF5data> readHDF5( hid_t fid, const std::string_view &name );


} // namespace AMP::IO


#endif
