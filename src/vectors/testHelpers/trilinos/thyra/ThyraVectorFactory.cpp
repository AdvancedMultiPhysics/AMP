#include "AMP/vectors/testHelpers/trilinos/thyra/ThyraVectorFactory.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/trilinos/thyra/ManagedThyraVector.h"
#include "AMP/vectors/trilinos/thyra/NativeThyraVectorData.h"


// Trilinos includes
DISABLE_WARNINGS
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#ifdef AMP_USE_TRILINOS_BELOS
    // #include "Thyra_SpmdVectorBase_def.hpp"
    #include "BelosMVOPTester.hpp"
    #include "BelosThyraAdapter.hpp"
#endif
#include "Thyra_DefaultSpmdVector_def.hpp"
#ifdef AMP_USE_TRILINOS_THYRAEPETRAADAPTERS
    #include "Thyra_EpetraThyraWrappers.hpp"
#endif
#ifdef AMP_USE_MPI
    #include <Epetra_MpiComm.h>

#else
    #include <Epetra_SerialComm.h>
#endif
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


/****************************************************************
 * NativeThyraFactory                                            *
 ****************************************************************/
AMP::LinearAlgebra::Vector::shared_ptr NativeThyraFactory::getVector() const
{
    AMP_MPI global_comm( AMP_COMM_WORLD );
    int local_size  = 101;
    int global_size = global_comm.sumReduce( local_size );
// Create an epetra vector
#ifdef AMP_USE_MPI
    Epetra_MpiComm comm = global_comm.getCommunicator();
#else
    Epetra_SerialComm comm;
#endif
    Teuchos::RCP<Epetra_Map> epetra_map( new Epetra_Map( global_size, local_size, 0, comm ) );
    Teuchos::RCP<Epetra_Vector> epetra_v( new Epetra_Vector( *epetra_map, true ) );
    // Create a thyra vector from the epetra vector
    auto space   = Thyra::create_VectorSpace( epetra_map );
    auto thyra_v = Thyra::create_Vector( epetra_v, space );
    // Create the NativeThyraVector
    auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "thyra" );
    auto vec = AMP::LinearAlgebra::createVector( thyra_v, local_size, global_comm, var );
    return vec;
}


/****************************************************************
 * ManagedThyraFactory                                           *
 ****************************************************************/
AMP::LinearAlgebra::Vector::shared_ptr ManagedThyraFactory::getVector() const
{
    auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "managed_thyra" );
    // Create an arbitrary vector
    auto vec1 = d_factory->getVector();
    vec1->setVariable( var );
    // Create the managed vector
    auto view = AMP::LinearAlgebra::ThyraVector::view( vec1 );
    auto vec2 = view->getManagedVec();
    vec2->setVariable( var );
    return vec2;
}


/****************************************************************
 * ManagedNativeThyraFactory                                     *
 ****************************************************************/
AMP::LinearAlgebra::Vector::shared_ptr ManagedNativeThyraFactory::getVector() const
{
    // Create an arbitrary vector
    auto vec1 = d_factory->getVector();
    // Create the managed vector
    auto vec2 = std::dynamic_pointer_cast<AMP::LinearAlgebra::ManagedThyraVector>(
        AMP::LinearAlgebra::ThyraVector::view( vec1 ) );
    // Create a native ThyraVector from the managed vector
    auto var  = std::make_shared<AMP::LinearAlgebra::Variable>( "managed_native_thyra" );
    auto vec3 = AMP::LinearAlgebra::createVector(
        vec2->getVec(), vec2->getLocalSize(), vec2->getComm(), var );
    return vec3;
}

#ifdef AMP_USE_TRILINOS_BELOS

Teuchos::RCP<Thyra::VectorBase<double>> getThyraVec( AMP::LinearAlgebra::Vector::shared_ptr v )
{
    auto mv = std::dynamic_pointer_cast<ManagedThyraVector>( v );
    if ( mv )
        return std::dynamic_pointer_cast<ThyraVector>( v )->getVec();

    auto nv_data = std::dynamic_pointer_cast<NativeThyraVectorData>( v->getVectorData() );
    if ( nv_data ) {
        return nv_data->getVec();
    } else {
        AMP_ERROR( "Not a Thyra Vector" );
    }
    return Teuchos::RCP<Thyra::VectorBase<double>>();
}

/****************************************************************
 * testBelosThyraVector                                          *
 ****************************************************************/
void testBelosThyraVector( AMP::UnitTest &ut, const VectorFactory &factory )
{
    auto vector    = factory.getVector();
    using TMVB     = Thyra::MultiVectorBase<double>;
    auto outputmgr = Teuchos::rcp( new Belos::OutputManager<double>() );

    bool pass = Belos::TestMultiVecTraits<double, TMVB>( outputmgr, getThyraVec( vector ) );
    if ( pass )
        ut.passes( "Belos::TestMultiVecTraits of thyra vector" );
    else
        ut.failure( "Belos::TestMultiVecTraits of thyra vector" );
}


#endif
} // namespace AMP::LinearAlgebra
