#ifndef included_AMP_RestartManager_hpp
#define included_AMP_RestartManager_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UtilityMacros.h"


namespace AMP::Geometry {
class Geometry;
}
namespace AMP::Mesh {
class Mesh;
class MeshIterator;
} // namespace AMP::Mesh
namespace AMP::Discretization {
class DOFManager;
}
namespace AMP::LinearAlgebra {
class Vector;
class Matrix;
} // namespace AMP::LinearAlgebra
namespace AMP::Operator {
class Operator;
}
namespace AMP::Solver {
class SolverStrategy;
}
namespace AMP::TimeIntegrator {
class TimeIntegrator;
}
namespace SAMRAI::tbox {
class Serializable;
}


namespace AMP::IO {


/********************************************************
 *  Register data with the manager                       *
 ********************************************************/
template<class TYPE>
RestartManager::DataStoreType<TYPE>::DataStoreType( hid_t fid,
                                                    uint64_t hash,
                                                    RestartManager *manager )
{
    d_hash = hash;
    d_data = this->read( fid, hash2String( hash ), manager );
}
template<class TYPE>
void RestartManager::registerData( const TYPE &data, const std::string &name )
{
    AMP_INSIST( d_fid == hid_t( -1 ),
                "We cannot register new items while a restart file is being read" );
    AMP_INSIST( !name.empty(), "Cannot register an object without a name" );
    auto id = registerObject<TYPE>( data );
    auto it = d_names.find( name );
    if ( it == d_names.end() ) {
        d_names[name] = id;
    } else if ( it->second != id ) {
        AMP_ERROR( "Two objects registered with the same name" );
    }
}
template<class TYPE>
uint64_t RestartManager::registerObject( const TYPE &data )
{
    AMP_INSIST( d_fid == hid_t( -1 ),
                "We cannot register new items while a restart file is being read" );
    using AMP::Discretization::DOFManager;
    using AMP::LinearAlgebra::Matrix;
    using AMP::LinearAlgebra::Vector;
    using AMP::Mesh::Mesh;
    using AMP::Mesh::MeshIterator;
    using AMP::Solver::SolverStrategy;
    using AMP::TimeIntegrator::TimeIntegrator;
    if constexpr ( std::is_same_v<TYPE, AMP::AMP_MPI> ) {
        return registerComm( data );
    } else if constexpr ( is_string_v<TYPE> ) {
        return registerObject( std::make_shared<const std::string>( data ) );
    } else if constexpr ( !AMP::is_shared_ptr_v<TYPE> ) {
        return registerObject( std::make_shared<const TYPE>( data ) );
    } else if constexpr ( !std::is_const_v<typename TYPE::element_type> ) {
        return registerObject( std::const_pointer_cast<const typename TYPE::element_type>( data ) );
    } else {
        if ( !data )
            return 0;
        std::shared_ptr<DataStore> obj;
        using TYPE2 = remove_cvref_t<typename TYPE::element_type>;
        if constexpr ( std::is_base_of_v<DataStore, TYPE2> || std::is_same_v<TYPE2, DataStore> ) {
            obj = data;
        } else if constexpr ( std::is_base_of_v<Mesh, TYPE2> && !std::is_same_v<TYPE2, Mesh> ) {
            obj = create( std::dynamic_pointer_cast<const Mesh>( data ) );
        } else if constexpr ( std::is_base_of_v<MeshIterator, TYPE2> &&
                              !std::is_same_v<TYPE2, MeshIterator> ) {
            obj = create( std::dynamic_pointer_cast<const MeshIterator>( data ) );
        } else if constexpr ( std::is_base_of_v<DOFManager, TYPE2> &&
                              !std::is_same_v<TYPE2, DOFManager> ) {
            obj = create( std::dynamic_pointer_cast<const DOFManager>( data ) );
        } else if constexpr ( std::is_base_of_v<Vector, TYPE2> && !std::is_same_v<TYPE2, Vector> ) {
            obj = create( std::dynamic_pointer_cast<const Vector>( data ) );
        } else if constexpr ( std::is_base_of_v<Matrix, TYPE2> && !std::is_same_v<TYPE2, Matrix> ) {
            obj = create( std::dynamic_pointer_cast<const Matrix>( data ) );
        } else if constexpr ( std::is_base_of_v<SolverStrategy, TYPE2> &&
                              !std::is_same_v<TYPE2, SolverStrategy> ) {
            obj = create( std::dynamic_pointer_cast<const SolverStrategy>( data ) );
        } else if constexpr ( std::is_base_of_v<TimeIntegrator, TYPE2> &&
                              !std::is_same_v<TYPE2, TimeIntegrator> ) {
            obj = create( std::dynamic_pointer_cast<const TimeIntegrator>( data ) );
        } else if constexpr ( std::is_base_of_v<SAMRAI::tbox::Serializable, TYPE2> ) {
            obj = std::make_shared<SAMRAIDataStore<TYPE2>>( data, this );
        } else {
            obj = create<TYPE2>( data );
        }
        auto hash = obj->getHash();
        AMP_ASSERT( hash != 0 );
        d_data[hash] = obj;
        return hash;
    }
}
template<class TYPE>
RestartManager::DataStorePtr RestartManager::create( std::shared_ptr<const TYPE> data )
{
    return std::make_shared<DataStoreType<TYPE>>( data, this );
}
template<class TYPE>
std::shared_ptr<TYPE> RestartManager::getData( const std::string &name )
{
    auto it = d_names.find( name );
    AMP_INSIST( it != d_names.end(), "Object not found: " + name );
    auto hash = it->second;
    return getData<TYPE>( hash );
}
template<class TYPE>
std::shared_ptr<TYPE> RestartManager::getData( uint64_t hash )
{
    if ( hash == 0 )
        return std::shared_ptr<TYPE>();
    using AMP::Discretization::DOFManager;
    using AMP::LinearAlgebra::Matrix;
    using AMP::LinearAlgebra::Vector;
    using AMP::Mesh::Mesh;
    using AMP::Mesh::MeshIterator;
    using AMP::Solver::SolverStrategy;
    using AMP::TimeIntegrator::TimeIntegrator;
    if constexpr ( std::is_base_of_v<Mesh, TYPE> && !std::is_same_v<TYPE, Mesh> ) {
        auto data = getData<Mesh>( hash );
        return std::dynamic_pointer_cast<TYPE>( data );
    } else if constexpr ( std::is_base_of_v<MeshIterator, TYPE> &&
                          !std::is_same_v<TYPE, MeshIterator> ) {
        auto data = getData<MeshIterator>( hash );
        return std::dynamic_pointer_cast<TYPE>( data );
    } else if constexpr ( std::is_base_of_v<DOFManager, TYPE> &&
                          !std::is_same_v<TYPE, DOFManager> ) {
        auto data = getData<DOFManager>( hash );
        return std::dynamic_pointer_cast<TYPE>( data );
    } else if constexpr ( std::is_base_of_v<Vector, TYPE> && !std::is_same_v<TYPE, Vector> ) {
        auto data = getData<Vector>( hash );
        return std::dynamic_pointer_cast<TYPE>( data );
    } else if constexpr ( std::is_base_of_v<Matrix, TYPE> && !std::is_same_v<TYPE, Matrix> ) {
        auto data = getData<Matrix>( hash );
        return std::dynamic_pointer_cast<TYPE>( data );
    } else if constexpr ( std::is_base_of_v<SolverStrategy, TYPE> &&
                          !std::is_same_v<TYPE, SolverStrategy> ) {
        auto data = getData<SolverStrategy>( hash );
        return std::dynamic_pointer_cast<TYPE>( data );
    } else if constexpr ( std::is_base_of_v<TimeIntegrator, TYPE> &&
                          !std::is_same_v<TYPE, TimeIntegrator> ) {
        auto data = getData<TimeIntegrator>( hash );
        return std::dynamic_pointer_cast<TYPE>( data );
    } else if constexpr ( std::is_base_of_v<SAMRAI::tbox::Serializable, TYPE> ) {
        return getSAMRAIData<TYPE>( hash );
    } else {
        auto it = d_data.find( hash );
        if ( it == d_data.end() ) {
            auto obj     = std::make_shared<DataStoreType<TYPE>>( d_fid, hash, this );
            d_data[hash] = obj;
            it           = d_data.find( hash );
        }
        auto data = std::dynamic_pointer_cast<DataStoreType<TYPE>>( it->second );
        return data->getData();
    }
}


} // namespace AMP::IO


#endif
