#include "AMP/vectors/testHelpers/generateVectorFactories.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/testHelpers/StridedVectorSelector.h"
#include "AMP/vectors/testHelpers/VectorFactory.h"

#ifdef AMP_USE_PETSC
    #include "AMP/vectors/petsc/PetscVector.h"
    #include "AMP/vectors/testHelpers/petsc/PetscVectorFactory.h"
#endif

#ifdef AMP_USE_TRILINOS
    #ifdef AMP_USE_TRILINOS_EPETRA
        #include "AMP/vectors/testHelpers/trilinos/epetra/EpetraVectorFactory.h"
        #include "AMP/vectors/trilinos/epetra/EpetraVector.h"
    #endif
    #ifdef AMP_USE_TRILINOS_TPETRA
        #include "AMP/vectors/testHelpers/trilinos/tpetra/TpetraVectorFactory.h"
        #include "AMP/vectors/trilinos/tpetra/TpetraVector.h"
    #endif
    #ifdef AMP_USE_TRILINOS_THYRA
        #include "AMP/vectors/testHelpers/trilinos/thyra/ThyraVectorFactory.h"
    #endif
#endif

#ifdef AMP_USE_SUNDIALS
    #include "AMP/vectors/testHelpers/sundials/SundialsVectorFactory.h"
#endif

#ifdef AMP_USE_OPENMP
    #include "AMP/vectors/operations/OpenMP/VectorOperationsOpenMP.h"
#endif
#ifdef AMP_USE_KOKKOS
    #include "AMP/vectors/operations/kokkos/VectorOperationsKokkos.h"
#endif
#ifdef AMP_USE_DEVICE
    #include "AMP/vectors/operations/device/VectorOperationsDevice.h"
#endif
#include "AMP/utils/Memory.h"


#include <string>
#include <vector>


namespace AMP::LinearAlgebra {


// Trim string
static inline std::string trim( const std::string &s )
{
    constexpr char space[] = { 0x20, 0x0c, 0x0a, 0x0d, 0x09, 0x0b, 0x0 };
    auto i                 = s.find_first_not_of( space );
    auto j                 = s.find_last_not_of( space );
    return s.substr( i, j - i + 1 );
}


// Split input arguments
static inline std::vector<std::string> splitArgs( const std::string &input )
{
    auto pos = input.find_first_of( '<' );
    if ( pos == std::string::npos )
        return std::vector<std::string>();
    std::string tmp = trim( input.substr( pos + 1 ) );
    tmp.resize( tmp.size() - 1 );
    tmp = trim( tmp );
    std::vector<std::string> args;
    while ( !tmp.empty() ) {
        pos       = 0;
        int count = 0;
        while ( pos < tmp.length() ) {
            if ( tmp[pos] == '<' )
                count++;
            if ( tmp[pos] == '>' )
                count--;
            if ( tmp[pos] == ',' && count == 0 )
                break;
            pos++;
        }
        args.push_back( tmp.substr( 0, pos ) );
        if ( pos + 1 < tmp.length() )
            tmp = trim( tmp.substr( pos + 1 ) );
        else
            tmp.clear();
    }
    return args;
}


// Convert strings to arguments of the desired type
int to_int( const std::string &s ) { return atoi( s.c_str() ); }
int to_bool( const std::string &s )
{
    if ( s == "false" )
        return false;
    else if ( s == "true" )
        return true;
    else
        AMP_ERROR( "Unknown value for bool: " + s );
    return false;
}
int count( const std::string &s, char c ) { return std::count( s.begin(), s.end(), c ); }


// Check if a factory is valid based on compiled packages
bool isValid( [[maybe_unused]] const std::string &name )
{
    bool valid = true;
#ifndef AMP_USE_PETSC
    valid = valid && name.find( "Petsc" ) == std::string::npos;
#endif
#ifndef AMP_USE_TRILINOS_EPETRA
    valid = valid && name.find( "Epetra" ) == std::string::npos;
#endif
#ifndef AMP_USE_TRILINOS_TPETRA
    valid = valid && name.find( "Tpetra" ) == std::string::npos;
#endif
#ifndef AMP_USE_TRILINOS_THYRA
    valid = valid && name.find( "Thyra" ) == std::string::npos;
#endif
#ifndef AMP_USE_SUNDIALS
    valid = valid && name.find( "Sundials" ) == std::string::npos;
#endif
#ifndef AMP_USE_OPENMP
    valid = valid && name.find( "openmp" ) == std::string::npos;
#endif
#ifndef AMP_USE_KOKKOS
    valid = valid && name.find( "kokkos" ) == std::string::npos;
#endif
#ifndef AMP_USE_DEVICE
    valid = valid && name.find( "gpu" ) == std::string::npos;
#endif
    return valid;
}


// Remove duplicate and invalid entries perserving the initial order
template<class TYPE>
static inline std::vector<TYPE> cleanList( const std::vector<TYPE> &x )
{
    std::vector<TYPE> y;
    y.reserve( x.size() );
    for ( size_t i = 0; i < x.size(); i++ ) {
        if ( !isValid( x[i] ) )
            continue;
        bool found = false;
        for ( size_t j = 0; j < y.size(); j++ )
            found = found || x[i].compare( y[j] ) == 0;
        if ( !found )
            y.push_back( x[i] );
    }
    return y;
}


// Generate a SimpleVectorFactory
template<typename TYPE, typename VecOps>
std::shared_ptr<VectorFactory>
generateSimpleVectorFactory( const std::string &name, int N, bool global, const std::string &data )
{
    std::shared_ptr<VectorFactory> factory;
    if ( data == "cpu" ) {
        using DATA = AMP::LinearAlgebra::VectorDataDefault<TYPE>;
        factory.reset( new SimpleVectorFactory<TYPE, VecOps, DATA>( N, global, name ) );
    } else if ( data == "gpu" ) {
#ifdef AMP_USE_DEVICE
        using ALLOC = ManagedAllocator<void>;
        using DATA  = AMP::LinearAlgebra::VectorDataDefault<TYPE, ALLOC>;
        factory.reset( new SimpleVectorFactory<TYPE, VecOps, DATA>( N, global, name ) );
#endif
    } else {
        AMP_ERROR( "Unknown VectorData" );
    }
    return factory;
}
template<typename TYPE>
std::shared_ptr<VectorFactory> generateSimpleVectorFactory(
    const std::string &name, int N, bool global, const std::string &ops, const std::string &data )
{
    std::shared_ptr<VectorFactory> factory;
    if ( ops == "default" ) {
        factory =
            generateSimpleVectorFactory<TYPE, AMP::LinearAlgebra::VectorOperationsDefault<TYPE>>(
                name, N, global, data );
    } else if ( ops == "openmp" ) {
#ifdef AMP_USE_OPENMP
        factory =
            generateSimpleVectorFactory<TYPE, AMP::LinearAlgebra::VectorOperationsOpenMP<TYPE>>(
                name, N, global, data );
#endif
    } else if ( ops == "kokkos" ) {
#ifdef AMP_USE_KOKKOS
        factory =
            generateSimpleVectorFactory<TYPE, AMP::LinearAlgebra::VectorOperationsKokkos<TYPE>>(
                name, N, global, data );
#endif
    } else if ( ops == "gpu" ) {
#ifdef AMP_USE_DEVICE
        factory =
            generateSimpleVectorFactory<TYPE, AMP::LinearAlgebra::VectorOperationsDevice<TYPE>>(
                name, N, global, data );
#endif
    } else {
        AMP_ERROR( "Unknown VectorOperations" );
    }
    return factory;
}
std::shared_ptr<VectorFactory> generateSimpleVectorFactory( const std::string &name,
                                                            int N,
                                                            bool global,
                                                            const std::string &type,
                                                            const std::string &ops,
                                                            const std::string &data )
{
    std::shared_ptr<VectorFactory> factory;
    if ( type == "double" ) {
        factory = generateSimpleVectorFactory<double>( name, N, global, ops, data );
    } else if ( type == "float" ) {
        factory = generateSimpleVectorFactory<float>( name, N, global, ops, data );
    } else {
        AMP_ERROR( "Unknown VectorOperations" );
    }
    return factory;
}


std::shared_ptr<VectorFactory> generateVectorFactory( const std::string &name )
{
    AMP_INSIST( isValid( name ), "Factory " + name + " is not valid based on compiled packages" );
    AMP_INSIST( count( name, '<' ) == count( name, '>' ), "Invalid factory: " + name );
    auto pos                      = name.find_first_of( '<' );
    const std::string factoryName = name.substr( 0, pos );
    auto args                     = splitArgs( name );
    std::shared_ptr<VectorFactory> factory;
    if ( factoryName == "SimpleVectorFactory" ) {
        // Create a simple vector
        AMP_ASSERT( args.size() >= 2 );
        if ( args.size() < 3 )
            args.emplace_back( "double" );
        if ( args.size() < 4 )
            args.emplace_back( "default" );
        if ( args.size() < 5 )
            args.emplace_back( "cpu" );
        factory = generateSimpleVectorFactory(
            name, to_int( args[0] ), to_bool( args[1] ), args[2], args[3], args[4] );
    } else if ( factoryName == "ArrayVectorFactory" ) {
        // Create an array vector
        AMP_ASSERT( args.size() >= 3 );
        if ( args.size() == 3 )
            args.emplace_back( "double" );
        if ( args[3] == "double" ) {
            factory.reset( new ArrayVectorFactory<double>(
                to_int( args[0] ), to_int( args[1] ), to_bool( args[2] ) ) );
        } else if ( args[3] == "float" ) {
            factory.reset( new ArrayVectorFactory<float>(
                to_int( args[0] ), to_int( args[1] ), to_bool( args[2] ) ) );
        } else {
            AMP_ERROR( "Unknown type" );
        }
    } else if ( factoryName == "CubeMeshVectorFactory" ) {
        AMP_ASSERT( args.size() == 1 );
        factory.reset( new CubeMeshVectorFactory( to_int( args[0] ) ) );
    } else if ( factoryName == "MultiVectorFactory" ) {
        AMP_ASSERT( args.size() == 4 );
        factory.reset( new MultiVectorFactory( generateVectorFactory( args[0] ),
                                               to_int( args[1] ),
                                               generateVectorFactory( args[2] ),
                                               to_int( args[3] ) ) );
    } else if ( factoryName == "CloneFactory" ) {
        AMP_ASSERT( args.size() == 1 );
        factory.reset( new CloneFactory( generateVectorFactory( args[0] ) ) );
    } else if ( factoryName == "StridedVectorFactory" ) {
        AMP_ASSERT( args.size() == 1 );
        factory.reset( new StridedVectorFactory( generateVectorFactory( args[0] ) ) );
#if defined( AMP_USE_PETSC )
    } else if ( factoryName == "NativePetscVectorFactory" ) {
        factory.reset( new NativePetscVectorFactory() );
#endif
#ifdef AMP_USE_TRILINOS_EPETRA
    } else if ( factoryName == "NativeEpetraFactory" ) {
        AMP_ASSERT( args.size() == 0 );
        factory.reset( new NativeEpetraFactory() );
#endif
#ifdef AMP_USE_TRILINOS_TPETRA
    } else if ( factoryName == "NativeTpetraFactory" ) {
        AMP_ASSERT( args.size() == 0 );
        factory.reset( new NativeTpetraFactory() );
#endif
#ifdef AMP_USE_TRILINOS_THYRA
    } else if ( factoryName == "NativeThyraFactory" ) {
        AMP_ASSERT( args.size() == 0 );
        factory.reset( new NativeThyraFactory() );
    } else if ( factoryName == "ManagedThyraFactory" ) {
        AMP_ASSERT( args.size() == 1 );
        factory.reset( new ManagedThyraFactory( generateVectorFactory( args[0] ) ) );
    } else if ( factoryName == "ManagedNativeThyraFactory" ) {
        AMP_ASSERT( args.size() == 1 );
        factory.reset( new ManagedNativeThyraFactory( generateVectorFactory( args[0] ) ) );
#endif
#ifdef EXT_SUNDIALS
    } else if ( factoryName == "NativeSundialsFactory" ) {
        AMP_ASSERT( args.size() == 0 );
        AMP_ERROR( "Not implemented" );
    } else if ( factoryName == "ManagedSundialsVectorFactory" ) {
        AMP_ASSERT( args.size() == 0 );
        AMP_ERROR( "Not implemented" );
#endif
    } else if ( factoryName == "ViewFactory" ) {
        AMP_ASSERT( args.size() == 2 );
        auto factory2 = generateVectorFactory( args[1] );
        if ( args[0] == "PetscVector" ) {
#ifdef AMP_USE_PETSC
            factory.reset( new ViewFactory<PetscVector>( factory2 ) );
#endif
        } else if ( args[0] == "EpetraVector" ) {
#ifdef AMP_USE_TRILINOS
            factory.reset( new ViewFactory<EpetraVector>( factory2 ) );
#endif
        } else if ( args[0] == "TpetraVector" ) {
#if defined( AMP_USE_TRILINOS ) && defined( AMP_USE_TRILINOS_TPETRA )
            factory.reset( new ViewFactory<TpetraVector>( factory2 ) );
#endif
        } else {
            AMP_ERROR( "Unknown template argument for ViewFactory" );
        }
    } else {
        AMP_ERROR( "Unknown factory: " + name );
    }
    return factory;
}


/********************************************************************
 * Get basic vector factories                                        *
 ********************************************************************/
std::vector<std::string> getSimpleVectorFactories()
{
    std::vector<std::string> list;
    list.emplace_back( "SimpleVectorFactory<15,false,double>" );
    list.emplace_back( "SimpleVectorFactory<45,true,double>" );
    list.emplace_back( "SimpleVectorFactory<15,false,double,openmp,cpu>" );
    list.emplace_back( "SimpleVectorFactory<15,false,double,kokkos,cpu>" );
    // list.push_back( "SimpleVectorFactory<15,false,double,default,gpu>" ); // Requires UVM
    list.emplace_back( "SimpleVectorFactory<15,false,double,gpu,gpu>" );
    list.emplace_back( "SimpleVectorFactory<15,false,float>" );
    list.emplace_back( "SimpleVectorFactory<15,true,float>" );
    list.emplace_back( "SimpleVectorFactory<15,false,float,openmp,cpu>" );
    // list.push_back( "SimpleVectorFactory<15,false,float,default,gpu>" ); // Requires UVM
    list.emplace_back( "SimpleVectorFactory<15,false,float,gpu,gpu>" );
    list = cleanList( list );
    return list;
}
std::vector<std::string> getArrayVectorFactories()
{
    std::vector<std::string> list;
    list.emplace_back( "ArrayVectorFactory<4,10,false,double>" );
    list.emplace_back( "ArrayVectorFactory<4,10,true,double>" );
    list.emplace_back( "ArrayVectorFactory<4,10,false,float>" );
    list.emplace_back( "ArrayVectorFactory<4,10,true,float>" );
    list = cleanList( list );
    return list;
}
std::vector<std::string> getNativeVectorFactories()
{
    std::vector<std::string> list;
    list.emplace_back( "NativePetscVectorFactory" );
    list.emplace_back( "NativeEpetraFactory" );
    list.emplace_back( "NativeTpetraFactory" );
    list.emplace_back( "NativeThyraFactory" );
    list = cleanList( list );
    return list;
}


/********************************************************************
 * Get advanced vector factories                                     *
 ********************************************************************/
std::vector<std::string> getMeshVectorFactories()
{
    std::vector<std::string> list;
    list.emplace_back( "CubeMeshVectorFactory<4>" );
    list.emplace_back( "CubeMeshVectorFactory<16>" );
    list.emplace_back(
        "MultiVectorFactory<CubeMeshVectorFactory<4>,1,SimpleVectorFactory<15,false,double>,1>" );
    return list;
}


/********************************************************************
 * Get advanced vector factories                                     *
 ********************************************************************/
std::vector<std::string> getMultiVectorFactories()
{
    std::vector<std::string> list;
    std::string MVFactory1 = "MultiVectorFactory<NativeEpetraFactory,1,NativePetscVectorFactory,1>";
    std::string MVFactory2 = "MultiVectorFactory<NativeEpetraFactory,3,NativePetscVectorFactory,2>";
    std::string MVFactory3 = "MultiVectorFactory<" + MVFactory1 + ",2," + MVFactory2 + ",2>";
    list.push_back( MVFactory1 );
    list.push_back( MVFactory2 );
    list.push_back( MVFactory3 );
    std::string MVFactory4 = "MultiVectorFactory<NativeTpetraFactory,1,NativePetscVectorFactory,1>";
    std::string MVFactory5 = "MultiVectorFactory<NativeTpetraFactory,3,NativePetscVectorFactory,2>";
    std::string MVFactory6 = "MultiVectorFactory<" + MVFactory1 + ",2," + MVFactory2 + ",2>";
    list.push_back( MVFactory4 );
    list.push_back( MVFactory5 );
    list.push_back( MVFactory6 );
    list = cleanList( list );
    return list;
}
std::vector<std::string> getManagedVectorFactories()
{
    std::vector<std::string> list;
    std::string MVFactory1 = "MultiVectorFactory<NativeEpetraFactory,1,NativePetscVectorFactory,1>";
    std::string MVFactory2 = "MultiVectorFactory<NativeEpetraFactory,3,NativePetscVectorFactory,2>";
    std::string MVFactory3 = "MultiVectorFactory<" + MVFactory1 + ",2," + MVFactory2 + ",2>";
    list.push_back( MVFactory1 );
    list.push_back( MVFactory2 );
    list.push_back( MVFactory3 );
    std::string MVFactory4 = "MultiVectorFactory<NativeTpetraFactory,1,NativePetscVectorFactory,1>";
    std::string MVFactory5 = "MultiVectorFactory<NativeTpetraFactory,3,NativePetscVectorFactory,2>";
    std::string MVFactory6 = "MultiVectorFactory<" + MVFactory1 + ",2," + MVFactory2 + ",2>";
    list.push_back( MVFactory4 );
    list.push_back( MVFactory5 );
    list.push_back( MVFactory6 );
    auto SimpleFactories             = getSimpleVectorFactories();
    std::string ManagedThyraFactory1 = "ManagedThyraFactory<" + SimpleFactories[0] + ">";
    std::string ManagedThyraFactory2 = "ManagedThyraFactory<" + SimpleFactories[1] + ">";
    std::string ManagedNativeThyraFactory1 =
        "ManagedNativeThyraFactory<" + SimpleFactories[0] + ">";
    std::string ManagedNativeThyraFactory2 =
        "ManagedNativeThyraFactory<" + SimpleFactories[1] + ">";
    std::string MNT_MVFactory = "ManagedNativeThyraFactory<" + MVFactory1 + ">";
    list.emplace_back( "NativeThyraFactory" );
    list.push_back( ManagedThyraFactory1 );
    list.push_back( ManagedThyraFactory2 );
    list.push_back( ManagedNativeThyraFactory1 );
    list.push_back( ManagedNativeThyraFactory2 );
    list.push_back( MNT_MVFactory );
    // list.push_back( "StridedVectorFactory<" + SMEVFactory + ">" );
    list = cleanList( list );
    return list;
}
std::vector<std::string> getCloneVectorFactories()
{
    std::vector<std::string> list;
    for ( auto &factory : getNativeVectorFactories() )
        list.push_back( "CloneFactory<" + factory + ">" );
    list.push_back( "CloneFactory<" + getSimpleVectorFactories()[0] + ">" );
    list.push_back( "CloneFactory<" + getArrayVectorFactories()[0] + ">" );
    std::string CloneMVFactory1 =
        "CloneFactory<MultiVectorFactory<" + list[0] + ",1," + list[1] + ",1>>";
    std::string CloneMVFactory2 =
        "CloneFactory<MultiVectorFactory<" + list[0] + ",3," + list[1] + ",2>>";
    std::string CloneMVFactory3 =
        "CloneFactory<MultiVectorFactory<" + CloneMVFactory1 + ",2," + CloneMVFactory2 + ",2>>";
    list.push_back( CloneMVFactory1 );
    list.push_back( CloneMVFactory2 );
    list.push_back( CloneMVFactory3 );
    list = cleanList( list );
    return list;
}

std::vector<std::string> getViewVectorFactories()
{
    std::vector<std::string> list;
    auto SimpleFactories = getSimpleVectorFactories();
    list.push_back( "ViewFactory<PetscVector," + SimpleFactories[0] + ">" );
    std::string ViewSNEVFactory = "ViewFactory<PetscVector,NativeEpetraFactory>";
    std::string ViewSNTVFactory = "ViewFactory<PetscVector,NativeTpetraFactory>";
    std::string ViewSNPVFactory = "ViewFactory<PetscVector,NativePetscVectorFactory>";
    std::string ViewMVFactory1  = "ViewFactory<PetscVector,MultiVectorFactory<" + ViewSNEVFactory +
                                 ",1," + ViewSNPVFactory + ",1>>";
    std::string ViewMVFactory2 = "ViewFactory<PetscVector,MultiVectorFactory<" + ViewSNEVFactory +
                                 ",3," + ViewSNPVFactory + ",2>>";
    std::string ViewMVFactory3 = "ViewFactory<PetscVector,MultiVectorFactory<" + ViewMVFactory1 +
                                 ",2," + ViewMVFactory2 + ",2>>";
    std::string ViewMVFactory4 = "ViewFactory<PetscVector,MultiVectorFactory<" + ViewSNTVFactory +
                                 ",1," + ViewSNPVFactory + ",1>>";
    std::string ViewMVFactory5 = "ViewFactory<PetscVector,MultiVectorFactory<" + ViewSNTVFactory +
                                 ",3," + ViewSNPVFactory + ",2>>";
    std::string ViewMVFactory6 = "ViewFactory<PetscVector,MultiVectorFactory<" + ViewMVFactory4 +
                                 ",2," + ViewMVFactory5 + ",2>>";
    list.push_back( ViewMVFactory1 );
    list.push_back( ViewSNEVFactory );
    list.push_back( ViewSNTVFactory );
    list.push_back( ViewSNPVFactory );
    list.push_back( ViewMVFactory1 );
    list.push_back( ViewMVFactory2 );
    list.push_back( ViewMVFactory3 );
    list.push_back( ViewMVFactory4 );
    list.push_back( ViewMVFactory5 );
    list.push_back( ViewMVFactory6 );
    for ( auto &factory : getManagedVectorFactories() )
        list.push_back( "ViewFactory<PetscVector," + factory + ">" );
    list = cleanList( list );
    return list;
}

std::vector<std::string> getCloneViewVectorFactories()
{
    std::vector<std::string> list;
    for ( auto &view : getViewVectorFactories() )
        list.push_back( "CloneFactory<" + view + ">" );
    std::string CloneViewSNEVFactory = "CloneFactory<ViewFactory<PetscVector,NativeEpetraFactory>>";
    std::string CloneViewSNTVFactory = "CloneFactory<ViewFactory<PetscVector,NativeTpetraFactory>>";
    std::string CloneViewSNPVFactory =
        "CloneFactory<ViewFactory<PetscVector,NativePetscVectorFactory>>";
    std::string CloneViewMVFactory1 = "CloneFactory<ViewFactory<PetscVector,MultiVectorFactory<" +
                                      CloneViewSNEVFactory + ",1," + CloneViewSNPVFactory + ",1>>>";
    std::string CloneViewMVFactory2 = "CloneFactory<ViewFactory<PetscVector,MultiVectorFactory<" +
                                      CloneViewSNEVFactory + ",3," + CloneViewSNPVFactory + ",2>>>";
    std::string CloneViewMVFactory3 = "CloneFactory<ViewFactory<PetscVector,MultiVectorFactory<" +
                                      CloneViewMVFactory1 + ",2," + CloneViewMVFactory2 + ",2>>>";
    std::string CloneViewMVFactory4 = "CloneFactory<ViewFactory<PetscVector,MultiVectorFactory<" +
                                      CloneViewSNTVFactory + ",1," + CloneViewSNPVFactory + ",1>>>";
    std::string CloneViewMVFactory5 = "CloneFactory<ViewFactory<PetscVector,MultiVectorFactory<" +
                                      CloneViewSNTVFactory + ",3," + CloneViewSNPVFactory + ",2>>>";
    std::string CloneViewMVFactory6 = "CloneFactory<ViewFactory<PetscVector,MultiVectorFactory<" +
                                      CloneViewMVFactory4 + ",2," + CloneViewMVFactory5 + ",2>>>";
    list.push_back( CloneViewMVFactory1 );
    list.push_back( CloneViewMVFactory2 );
    list.push_back( CloneViewMVFactory3 );
    list.push_back( CloneViewMVFactory4 );
    list.push_back( CloneViewMVFactory5 );
    list.push_back( CloneViewMVFactory6 );
    list = cleanList( list );
    return list;
}


/********************************************************************
 * Get all vector factories                                          *
 ********************************************************************/
std::vector<std::string> getAllFactories()
{
    std::vector<std::string> list;
    for ( auto &factory : getSimpleVectorFactories() )
        list.push_back( factory );
    for ( auto &factory : getArrayVectorFactories() )
        list.push_back( factory );
    for ( auto &factory : getNativeVectorFactories() )
        list.push_back( factory );
    for ( auto &factory : getMeshVectorFactories() )
        list.push_back( factory );
    for ( auto &factory : getMultiVectorFactories() )
        list.push_back( factory );
    for ( auto &factory : getManagedVectorFactories() )
        list.push_back( factory );
    for ( auto &factory : getCloneVectorFactories() )
        list.push_back( factory );
    for ( auto &factory : getViewVectorFactories() )
        list.push_back( factory );
    for ( auto &factory : getCloneViewVectorFactories() )
        list.push_back( factory );
    list = cleanList( list );
    return list;
}


} // namespace AMP::LinearAlgebra
