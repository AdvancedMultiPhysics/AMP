#include "AMP/utils/MeshPoint.h"
#include "AMP/IO/HDF.hpp"

#include <sstream>


/****************************************************************
 * Print the point                                               *
 ****************************************************************/
void AMP::Mesh::MeshPoint::print( std::ostream &os ) const
{
    if ( d_ndim == 0 ) {
        os << "()";
    } else {
        os << "(" << d_data[0];
        for ( int d = 1; d < d_ndim; d++ )
            os << "," << d_data[d];
        os << ")";
    }
}
std::string AMP::Mesh::MeshPoint::print() const
{
    std::ostringstream stream;
    print( stream );
    return stream.str();
}
std::ostream &AMP::Mesh::operator<<( std::ostream &out, const AMP::Mesh::MeshPoint &x )
{
    x.print( out );
    return out;
}


/********************************************************
 * Explicit instantiations                               *
 ********************************************************/
using Array1 = std::array<AMP::Mesh::MeshPoint, 1>;
using Array2 = std::array<AMP::Mesh::MeshPoint, 2>;
using Array3 = std::array<AMP::Mesh::MeshPoint, 3>;
using Array4 = std::array<AMP::Mesh::MeshPoint, 4>;
using Array6 = std::array<AMP::Mesh::MeshPoint, 6>;
template void AMP::IO::writeHDF5<Array1>( hid_t, std::string const &, Array1 const & );
template void AMP::IO::writeHDF5<Array2>( hid_t, std::string const &, Array2 const & );
template void AMP::IO::writeHDF5<Array3>( hid_t, std::string const &, Array3 const & );
template void AMP::IO::writeHDF5<Array4>( hid_t, std::string const &, Array4 const & );
template void AMP::IO::writeHDF5<Array6>( hid_t, std::string const &, Array6 const & );
template void AMP::IO::readHDF5<Array1>( hid_t, std::string const &, Array1 & );
template void AMP::IO::readHDF5<Array2>( hid_t, std::string const &, Array2 & );
template void AMP::IO::readHDF5<Array3>( hid_t, std::string const &, Array3 & );
template void AMP::IO::readHDF5<Array4>( hid_t, std::string const &, Array4 & );
template void AMP::IO::readHDF5<Array6>( hid_t, std::string const &, Array6 & );
