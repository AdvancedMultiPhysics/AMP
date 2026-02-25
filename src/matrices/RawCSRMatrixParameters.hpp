#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/RawCSRMatrixParameters.h"

namespace AMP::LinearAlgebra {


template<typename Config>
void RawCSRMatrixParameters<Config>::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    MatrixParametersBase::registerChildObjects( manager );
}

template<typename Config>
void RawCSRMatrixParameters<Config>::writeRestart( int64_t fid ) const
{
    MatrixParametersBase::writeRestart( fid );
    IO::writeHDF5( fid, "first_row", d_first_row );
    IO::writeHDF5( fid, "last_row", d_last_row );
    IO::writeHDF5( fid, "first_col", d_first_col );
    IO::writeHDF5( fid, "last_col", d_last_col );

    // not recreating d_diag and d_off_diag at present
}

template<typename Config>
RawCSRMatrixParameters<Config>::RawCSRMatrixParameters( int64_t fid,
                                                        AMP::IO::RestartManager *manager )
    : MatrixParametersBase( fid, manager )
{
    IO::readHDF5( fid, "first_row", d_first_row );
    IO::readHDF5( fid, "last_row", d_last_row );
    IO::readHDF5( fid, "first_col", d_first_col );
    IO::readHDF5( fid, "last_col", d_last_col );

    // not recreating d_diag and d_off_diag at present
    //    d_diag.writeRestart( fid );
    //    d_off_diag.writeRestart( fid );
    //    AMP_ERROR( "Not complete" );
}

} // namespace AMP::LinearAlgebra
