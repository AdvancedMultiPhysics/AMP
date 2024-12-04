#include "AMP/utils/MeshPoint.h"
#include "AMP/IO/HDF5.hpp"


/********************************************************
 * Explicit instantiations                               *
 ********************************************************/
template void AMP::IO::writeHDF5<std::array<AMP::Mesh::MeshPoint<double>, 1>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 1> const & );
template void AMP::IO::writeHDF5<std::array<AMP::Mesh::MeshPoint<double>, 2>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 2> const & );
template void AMP::IO::writeHDF5<std::array<AMP::Mesh::MeshPoint<double>, 3>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 3> const & );
template void AMP::IO::writeHDF5<std::array<AMP::Mesh::MeshPoint<double>, 4>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 4> const & );
template void AMP::IO::writeHDF5<std::array<AMP::Mesh::MeshPoint<double>, 6>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 6> const & );
template void AMP::IO::readHDF5<std::array<AMP::Mesh::MeshPoint<double>, 1>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 1> & );
template void AMP::IO::readHDF5<std::array<AMP::Mesh::MeshPoint<double>, 2>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 2> & );
template void AMP::IO::readHDF5<std::array<AMP::Mesh::MeshPoint<double>, 3>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 3> & );
template void AMP::IO::readHDF5<std::array<AMP::Mesh::MeshPoint<double>, 4>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 4> & );
template void AMP::IO::readHDF5<std::array<AMP::Mesh::MeshPoint<double>, 6>>(
    long, std::string const &, std::array<AMP::Mesh::MeshPoint<double>, 6> & );