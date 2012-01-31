#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include <iostream>
#include <string>

#include "boost/shared_ptr.hpp"

#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/AMPManager.h"
#include "utils/PIO.h"

#include "ampmesh/Mesh.h"
#include "ampmesh/SiloIO.h"
#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/Vector.h"
#include "vectors/MultiVariable.h"
#include "vectors/VectorBuilder.h"


void myTest(AMP::UnitTest *ut, std::string exeName)
{
    std::string input_file = "input_" + exeName;
    std::string log_file = "output_" + exeName;
    AMP::PIO::logOnlyNodeZero(log_file);

    // Read the input file
    boost::shared_ptr<AMP::InputDatabase>  input_db ( new AMP::InputDatabase ( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile ( input_file , input_db );

    // Get the Mesh database and create the mesh parameters
    boost::shared_ptr<AMP::Database> database = input_db->getDatabase( "Mesh" );
    boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
    params->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));

    // Create the meshes from the input database
    AMP::Mesh::Mesh::shared_ptr mesh = AMP::Mesh::Mesh::buildMesh(params);

    //Construct Variables
    boost::shared_ptr<AMP::LinearAlgebra::Variable> Variable1 (new AMP::LinearAlgebra::Variable("Var1"));
    boost::shared_ptr<AMP::LinearAlgebra::Variable> Variable2 (new AMP::LinearAlgebra::Variable("Var2"));
    boost::shared_ptr<AMP::LinearAlgebra::Variable> Variable3 (new AMP::LinearAlgebra::Variable("Var3"));
    boost::shared_ptr<AMP::LinearAlgebra::Variable> Variable4 (new AMP::LinearAlgebra::Variable("Var4"));

    boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> subVariable (new AMP::LinearAlgebra::MultiVariable("subVar"));
    subVariable->add( Variable1   );
    subVariable->add( Variable2   );

    boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> fullVariable(new AMP::LinearAlgebra::MultiVariable("fullVariable"));
    fullVariable->add( Variable1   );
    fullVariable->add( Variable2   );
    fullVariable->add( Variable3   );
    fullVariable->add( Variable4   );

    // Create the DOF manager
    AMP::Discretization::DOFManager::shared_ptr DOFs = AMP::Discretization::simpleDOFManager::create(mesh,AMP::Mesh::Vertex,1,1);

    // Create the vectors
    AMP::LinearAlgebra::Vector::shared_ptr FullVector = AMP::LinearAlgebra::createVector( DOFs, fullVariable );
    AMP::LinearAlgebra::Vector::shared_ptr SubVector  = FullVector->subsetVectorForVariable( subVariable );
    if ( SubVector.get()!=NULL ) {
        ut->passes("Sub Vector is not NULL");
        if ( FullVector->getGlobalSize() == 2*SubVector->getGlobalSize() )
            ut->passes("Sub Vector is the right size");
        else
            ut->failure("Sub Vector is the right size");
    } else {
        ut->failure("Sub Vector is not NULL");
    }
}


int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.push_back("testMultiVector");
    for (size_t i=0; i<exeNames.size(); i++) {
        myTest(&ut, exeNames[i]);
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}   


