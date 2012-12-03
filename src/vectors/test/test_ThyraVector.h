#include "test_Vector.h"
#include "discretization/DOF_Manager.h"
#include "vectors/VectorBuilder.h"
#include "vectors/trilinos/ManagedThyraVector.h"
#include "vectors/trilinos/NativeThyraVector.h"
#include "utils/AMP_MPI.h"

// Trilinos includes
#include <Epetra_Vector.h>
#include <Epetra_Map.h>
#include "Thyra_SpmdVectorBase_def.hpp"
#include "Thyra_DefaultSpmdVector_def.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"

#ifdef USE_EXT_MPI
    #include <Epetra_MpiComm.h>
#else
    #include <Epetra_SerialComm.h>
#endif

/// \cond UNDOCUMENTED

namespace AMP {
namespace unit_test {


class  NativeThyraFactory
{
public:
    typedef AMP::LinearAlgebra::ThyraVector                  vector;

    static AMP::LinearAlgebra::Variable::shared_ptr  getVariable() {
        return AMP::LinearAlgebra::Variable::shared_ptr ( new AMP::LinearAlgebra::Variable ( "thyra" ) );
    }

    static AMP::LinearAlgebra::Vector::shared_ptr getVector() {
        AMP_MPI global_comm(AMP_COMM_WORLD);
        int local_size = 101;
        int global_size = global_comm.sumReduce(local_size);
        // Create an epetra vector
        #ifdef USE_EXT_MPI
            Epetra_MpiComm  comm = global_comm.getCommunicator();
        #else
            Epetra_SerialComm  comm;
        #endif
        Teuchos::RCP<Epetra_Map> epetra_map( new Epetra_Map(global_size,local_size,0,comm) );
        Teuchos::RCP<Epetra_Vector> epetra_v( new Epetra_Vector(*epetra_map,true) );
        // Create a thyra vector from the epetra vector
        Teuchos::RCP<const Thyra::VectorSpaceBase<double> > space = Thyra::create_VectorSpace( epetra_map );
        Teuchos::RCP<Thyra::VectorBase<double> > thyra_v = Thyra::create_Vector( epetra_v, space );
        // Create the NativeThyraVector
        boost::shared_ptr<AMP::LinearAlgebra::NativeThyraVectorParameters> params( 
            new AMP::LinearAlgebra::NativeThyraVectorParameters() );
        params->d_InVec = thyra_v;
        params->d_local = local_size;
        params->d_comm = global_comm;
        params->d_var = getVariable();
        boost::shared_ptr<AMP::LinearAlgebra::NativeThyraVector> vec( 
            new AMP::LinearAlgebra::NativeThyraVector( params ) );
        return vec;
    }
};


template <typename FACTORY>
class  ManagedThyraFactory
{
public:
    typedef AMP::LinearAlgebra::ThyraVector                  vector;

    static AMP::LinearAlgebra::Variable::shared_ptr  getVariable() {
        return AMP::LinearAlgebra::Variable::shared_ptr ( new AMP::LinearAlgebra::Variable ( "managed_thyra" ) );
    }

    static AMP::LinearAlgebra::Vector::shared_ptr getVector() {
        // Create an arbitrary vector
        AMP::LinearAlgebra::Vector::shared_ptr vec1 = FACTORY::getVector();
        // Create the managed vector
        AMP::LinearAlgebra::Vector::shared_ptr vec2 = AMP::LinearAlgebra::ThyraVector::view(vec1);
        vec2->setVariable( getVariable() );
        return vec2;
    }
};


template <typename FACTORY>
class  ManagedNativeThyraFactory
{
public:
    typedef AMP::LinearAlgebra::ThyraVector                  vector;

    static AMP::LinearAlgebra::Variable::shared_ptr  getVariable() {
        return AMP::LinearAlgebra::Variable::shared_ptr ( new AMP::LinearAlgebra::Variable ( "managed_native_thyra" ) );
    }

    static AMP::LinearAlgebra::Vector::shared_ptr getVector() {
        // Create an arbitrary vector
        AMP::LinearAlgebra::Vector::shared_ptr vec1 = FACTORY::getVector();
        // Create the managed vector
        boost::shared_ptr<AMP::LinearAlgebra::ManagedThyraVector> vec2 = 
            boost::dynamic_pointer_cast<AMP::LinearAlgebra::ManagedThyraVector>(
            AMP::LinearAlgebra::ThyraVector::view(vec1) );
        // Create a native ThyraVector from the managed vector
        boost::shared_ptr<AMP::LinearAlgebra::NativeThyraVectorParameters> params( 
            new AMP::LinearAlgebra::NativeThyraVectorParameters() );
        params->d_InVec = vec2->getVec();
        params->d_local = vec2->getLocalSize();
        params->d_comm = vec2->getComm();
        params->d_var = getVariable();
        boost::shared_ptr<AMP::LinearAlgebra::NativeThyraVector> vec3( 
            new AMP::LinearAlgebra::NativeThyraVector( params ) );
        return vec3;
    }
};


}
}

/// \endcond