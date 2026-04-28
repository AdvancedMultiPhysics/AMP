#include "AMP/IO/RestartManager.h"
#include "AMP/IO/RestartManager.hpp"
#include "AMP/utils/Database.h"

#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/hier/PatchDataRestartManager.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/MemoryDatabase.h"
#include "SAMRAI/tbox/NullDatabase.h"
#include "SAMRAI/tbox/RestartManager.h"


/********************************************************
 * Special class to wrap SAMRAI's restart manager and    *
 * enable writing/reading the root database              *
 ********************************************************/
struct SAMRAIRestartManager {
    static constexpr uint64_t id = 0xC79C837241B11E7B;
    static int count;
    SAMRAIRestartManager() = delete;
    SAMRAIRestartManager( AMP::IO::RestartManager * )
    {
        AMP_ASSERT( count++ == 0 );
        resetRestartManagerSAMRAI();
    }
    SAMRAIRestartManager( const AMP::Database &db )
    {
        AMP_ASSERT( count++ == 0 );
        resetRestartManagerSAMRAI();
        auto root          = db.cloneToSAMRAI();
        auto SAMRAImanager = SAMRAI::tbox::RestartManager::getManager();
        SAMRAImanager->setRootDatabase( root );
    }
    ~SAMRAIRestartManager()
    {
        AMP_ASSERT( --count == 0 );
        resetRestartManagerSAMRAI();
    }
    static void resetRestartManagerSAMRAI()
    {
        auto SAMRAImanager = SAMRAI::tbox::RestartManager::getManager();
        SAMRAImanager->setRootDatabase( std::make_shared<SAMRAI::tbox::MemoryDatabase>( "root" ) );
        SAMRAImanager->clearRestartItems();
    }
};
int SAMRAIRestartManager::count = 0;
void registerSAMRAIObject( AMP::IO::RestartManager *manager,
                           std::shared_ptr<const SAMRAI::tbox::Serializable> obj,
                           const std::string &name )
{
    // Register the SAMRAI RestartManager with AMP's Restart Manager
    // Note: this will initialize SAMRAI's RestartManager
    if ( !manager->isRegistered( SAMRAIRestartManager::id ) )
        manager->registerObject( std::make_shared<SAMRAIRestartManager>( manager ) );
    // Register the object with SAMRAI's Restart Manager
    auto SAMRAImanager = SAMRAI::tbox::RestartManager::getManager();
    SAMRAImanager->registerRestartItem( name,
                                        const_cast<SAMRAI::tbox::Serializable *>( obj.get() ) );
}
void loadSAMRAIManager( AMP::IO::RestartManager *manager )
{
    // Load the data for SAMRAI's RestartManager
    manager->getData<SAMRAIRestartManager>( SAMRAIRestartManager::id );
}
template<>
AMP::IO::RestartManager::DataStoreType<SAMRAIRestartManager>::DataStoreType(
    std::shared_ptr<const SAMRAIRestartManager> obj, RestartManager * )
{
    this->d_hash = SAMRAIRestartManager::id;
    this->d_data = obj;
}
template<>
void AMP::IO::RestartManager::DataStoreType<SAMRAIRestartManager>::write(
    hid_t fid, const std::string &name ) const
{
    auto SAMRAImanager = SAMRAI::tbox::RestartManager::getManager();
    SAMRAImanager->writeRestartToDatabase();
    auto db = std::make_shared<AMP::Database>( *SAMRAImanager->getRootDatabase() );
    db->setDefaultAddKeyBehavior( AMP::Database::Check::Overwrite, true );
    AMP::IO::writeHDF5( fid, name, *db );
}
template<>
std::shared_ptr<SAMRAIRestartManager>
AMP::IO::RestartManager::DataStoreType<SAMRAIRestartManager>::read( hid_t fid,
                                                                    const std::string &name,
                                                                    RestartManager * ) const
{
    AMP_INSIST( SAMRAIRestartManager::count == 0,
                "RestartManager must be reset before loading data when working with SAMRAI" );
    AMP::Database db;
    AMP::IO::readHDF5( fid, name, db );
    db.setDefaultAddKeyBehavior( AMP::Database::Check::Overwrite, true );
    return std::make_shared<SAMRAIRestartManager>( db );
}


/********************************************************
 * CartesianGridGeometry                                 *
 ********************************************************/
template<>
AMP::IO::RestartManager::SAMRAIDataStore<SAMRAI::geom::CartesianGridGeometry>::SAMRAIDataStore(
    std::shared_ptr<const SAMRAI::geom::CartesianGridGeometry> data, RestartManager *manager )
{
    this->d_hash = reinterpret_cast<uint64_t>( data.get() );
    this->d_data = data;

    // Register the object (this will register it with SAMRAI's restart manager)
    if ( !manager->isRegistered( this->d_hash ) ) {
        registerSAMRAIObject( manager, data, hash2String( this->d_hash ) );
    }
}
template<>
void AMP::IO::RestartManager::SAMRAIDataStore<SAMRAI::geom::CartesianGridGeometry>::write(
    hid_t fid, const std::string &name ) const
{
    auto gid = AMP::IO::createGroup( fid, name );
    AMP::IO::writeHDF5( gid, "dim", static_cast<int>( this->d_data->getDim().getValue() ) );
    AMP::IO::closeGroup( gid );
}

template<>
std::shared_ptr<SAMRAI::geom::CartesianGridGeometry>
AMP::IO::RestartManager::SAMRAIDataStore<SAMRAI::geom::CartesianGridGeometry>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    auto gid = AMP::IO::openGroup( fid, name );
    int dim;
    AMP::IO::readHDF5( gid, "dim", dim );
    AMP::IO::closeGroup( gid );
    // Load and set the root database for SAMRAI's RestartDatabase
    loadSAMRAIManager( manager );
    auto db         = SAMRAI::tbox::RestartManager::getManager()->getRootDatabase();
    auto restart_db = db->getDatabase( name );
    return std::make_shared<SAMRAI::geom::CartesianGridGeometry>(
        SAMRAI::tbox::Dimension( dim ), name, restart_db );
}


/********************************************************
 * PatchHierarchy                                        *
 ********************************************************/
template<>
AMP::IO::RestartManager::SAMRAIDataStore<SAMRAI::hier::PatchHierarchy>::SAMRAIDataStore(
    std::shared_ptr<const SAMRAI::hier::PatchHierarchy> data, RestartManager *manager )
{
    this->d_hash = reinterpret_cast<uint64_t>( data.get() );
    this->d_data = data;

    if ( !manager->isRegistered( this->d_hash ) ) {
        // Register the object (this will register it with SAMRAI's restart manager)
        registerSAMRAIObject( manager, data, hash2String( this->d_hash ) );
    }

    // Register all child objects
    auto grid_geometry =
        std::dynamic_pointer_cast<SAMRAI::geom::CartesianGridGeometry>( data->getGridGeometry() );
    AMP_ASSERT( grid_geometry );
    manager->registerObject( grid_geometry );
    manager->registerComm( AMP::AMP_MPI( d_data->getMPI() ) );
}
template<>
void AMP::IO::RestartManager::SAMRAIDataStore<SAMRAI::hier::PatchHierarchy>::write(
    hid_t fid, const std::string &name ) const
{
    auto gid           = AMP::IO::createGroup( fid, name );
    auto grid_geometry = std::dynamic_pointer_cast<SAMRAI::geom::CartesianGridGeometry>(
        this->d_data->getGridGeometry() );
    AMP_ASSERT( grid_geometry );
    auto ptr = reinterpret_cast<std::uintptr_t>( grid_geometry.get() );
    AMP::IO::writeHDF5( gid, "CartesianGridGeometry", static_cast<uint64_t>( ptr ) );
    AMP::IO::writeHDF5( gid, "dim", static_cast<int>( this->d_data->getDim().getValue() ) );
    AMP::IO::writeHDF5( gid, "comm", AMP::AMP_MPI( d_data->getMPI() ).hash() );
    AMP::IO::closeGroup( gid );
}
template<>
std::shared_ptr<SAMRAI::hier::PatchHierarchy>
AMP::IO::RestartManager::SAMRAIDataStore<SAMRAI::hier::PatchHierarchy>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    auto gid = AMP::IO::openGroup( fid, name );

    int dim;
    uint64_t geom_id;
    uint64_t comm_id;
    AMP::IO::readHDF5( gid, "dim", dim );
    AMP::IO::readHDF5( gid, "comm", comm_id );
    AMP::IO::readHDF5( gid, "CartesianGridGeometry", geom_id );
    auto gridGeometry = manager->getSAMRAIData<SAMRAI::geom::CartesianGridGeometry>( geom_id );
    AMP_ASSERT( gridGeometry );

    auto ampComm    = manager->getComm( comm_id );
    auto samraiComm = static_cast<SAMRAI::tbox::SAMRAI_MPI>( ampComm );

    AMP::IO::closeGroup( gid );

    // Load and set the root database for SAMRAI's RestartDatabase
    loadSAMRAIManager( manager );
    auto db         = SAMRAI::tbox::RestartManager::getManager()->getRootDatabase();
    auto restart_db = db->getDatabase( name );

    // Create the patch hierarchy
    auto hier = std::make_shared<SAMRAI::hier::PatchHierarchy>(
        name, gridGeometry, restart_db, samraiComm );
    hier->initializeHierarchy();
    return hier;
}


/********************************************************
 *  Default implementations for SAMRAI data              *
 ********************************************************/
template<class TYPE>
AMP::IO::RestartManager::SAMRAIDataStore<TYPE>::SAMRAIDataStore( hid_t fid,
                                                                 uint64_t hash,
                                                                 AMP::IO::RestartManager *manager )
{
    this->d_hash = hash;
    this->d_data = this->read( fid, hash2String( hash ), manager );
}
template<class TYPE>
AMP::IO::RestartManager::SAMRAIDataStore<TYPE>::SAMRAIDataStore( std::shared_ptr<const TYPE> data,
                                                                 RestartManager *manager )
{
    this->d_hash = reinterpret_cast<uint64_t>( data.get() );
    this->d_data = data;
    // Register the object (this will register it with SAMRAI's restart manager)
    registerSAMRAIObject( manager, data, hash2String( this->d_hash ) );
}
template<class TYPE>
void AMP::IO::RestartManager::SAMRAIDataStore<TYPE>::write( hid_t fid,
                                                            const std::string &name ) const
{
    // The database is written through the restart file
}
template<class TYPE>
std::shared_ptr<TYPE> AMP::IO::RestartManager::SAMRAIDataStore<TYPE>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    // Load and set the root database for SAMRAI's RestartDatabase
    loadSAMRAIManager( manager );
    auto db         = SAMRAI::tbox::RestartManager::getManager()->getRootDatabase();
    auto restart_db = db->getDatabase( name );
    // Now what?
    AMP_ERROR( "Not finished" );
    return nullptr;
}

/********************************************************
 *  Other functions                                      *
 ********************************************************/
template<class TYPE>
std::shared_ptr<TYPE> AMP::IO::RestartManager::getSAMRAIData( uint64_t hash )
{
    auto it = d_data.find( hash );
    if ( it == d_data.end() ) {
        auto obj     = std::make_shared<SAMRAIDataStore<TYPE>>( d_fid, hash, this );
        d_data[hash] = obj;
        it           = d_data.find( hash );
    }
    auto data = std::dynamic_pointer_cast<SAMRAIDataStore<TYPE>>( it->second );
    return data->getData();
}
template<class TYPE>
AMP::IO::RestartManager::DataStoreType<TYPE>::DataStoreType( std::shared_ptr<const TYPE> data,
                                                             RestartManager *manager )
{
    AMP_ERROR( "This should not be called" );
}
template<class TYPE>
void AMP::IO::RestartManager::DataStoreType<TYPE>::write( hid_t, const std::string & ) const
{
    AMP_ERROR( "This should not be called" );
}
template<class TYPE>
std::shared_ptr<TYPE> AMP::IO::RestartManager::DataStoreType<TYPE>::read( hid_t,
                                                                          const std::string &,
                                                                          RestartManager * ) const
{
    AMP_ERROR( "This should not be called" );
}


/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
#define INSTANTIATE( TYPE )                                        \
    template class AMP::IO::RestartManager::SAMRAIDataStore<TYPE>; \
    template std::shared_ptr<TYPE> AMP::IO::RestartManager::getSAMRAIData<TYPE>( uint64_t )
INSTANTIATE( SAMRAI::hier::PatchHierarchy );
INSTANTIATE( SAMRAI::geom::CartesianGridGeometry );
