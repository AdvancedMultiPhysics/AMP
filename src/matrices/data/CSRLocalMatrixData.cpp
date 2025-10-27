#include "AMP/matrices/data/CSRLocalMatrixData.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/utils/Memory.h"

#define CSR_INST_RESTART_STORE_TYPE( mode )                                                    \
    template<>                                                                                 \
    AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::CSRLocalMatrixData<             \
        AMP::LinearAlgebra::config_mode_t<AMP::LinearAlgebra::mode>>>::                        \
        DataStoreType( std::shared_ptr<const AMP::LinearAlgebra::CSRLocalMatrixData<           \
                           AMP::LinearAlgebra::config_mode_t<AMP::LinearAlgebra::mode>>> data, \
                       AMP::IO::RestartManager *manager )                                      \
        : d_data( data )                                                                       \
    {                                                                                          \
        d_hash = data->getID();                                                                \
        d_data->registerChildObjects( manager );                                               \
    }

CSR_CONFIG_FORALL( CSR_INST_RESTART_STORE_TYPE )

#define CSR_INST_RESTART_WRITE( mode )                                                          \
    template<>                                                                                  \
    void AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::CSRLocalMatrixData<         \
        AMP::LinearAlgebra::config_mode_t<AMP::LinearAlgebra::mode>>>::write( hid_t fid,        \
                                                                              const std::string \
                                                                                  &name ) const \
    {                                                                                           \
        hid_t gid = createGroup( fid, name );                                                   \
        writeHDF5( gid, "ClassType", d_data->type() );                                          \
        d_data->writeRestart( gid );                                                            \
        closeGroup( gid );                                                                      \
    }

CSR_CONFIG_FORALL( CSR_INST_RESTART_WRITE )

#define CSR_INST_RESTART_READ( mode )                                                      \
    template<>                                                                             \
    std::shared_ptr<AMP::LinearAlgebra::CSRLocalMatrixData<                                \
        AMP::LinearAlgebra::config_mode_t<AMP::LinearAlgebra::mode>>>                      \
    AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::CSRLocalMatrixData<         \
        AMP::LinearAlgebra::config_mode_t<AMP::LinearAlgebra::mode>>>::                    \
        read( hid_t fid, const std::string &name, AMP::IO::RestartManager *manager ) const \
    {                                                                                      \
        hid_t gid = openGroup( fid, name );                                                \
        std::string type;                                                                  \
        AMP::IO::readHDF5( gid, "ClassType", type );                                       \
        auto localMatrixData = std::make_shared<AMP::LinearAlgebra::CSRLocalMatrixData<    \
            AMP::LinearAlgebra::config_mode_t<AMP::LinearAlgebra::mode>>>( gid, manager ); \
        closeGroup( gid );                                                                 \
        return localMatrixData;                                                            \
    }

CSR_CONFIG_FORALL( CSR_INST_RESTART_READ )
