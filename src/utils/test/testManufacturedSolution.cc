/*
 * testManufacturedSolution.cc
 *
 *  Created on: Jul 27, 2010
 *      Author: gad
 */

#include <valarray>
#include <iostream>
#include <exception>

#include "utils/Utilities.h"
#include "utils/AMPManager.h"
#include "utils/AMP_MPI.h"
#include "utils/UnitTest.h"
#include "../ManufacturedSolution.h"
#include "../Database.h"
#include "../MemoryDatabase.h"
#include "boost/shared_ptr.hpp"


void testit(AMP::UnitTest *ut,
		std::string geom, std::string order, std::string bc,
		double x, double y, double z)
{
    boost::shared_ptr<AMP::Database> db(
        boost::dynamic_pointer_cast<AMP::Database>(
            boost::shared_ptr<AMP::MemoryDatabase>(
                new AMP::MemoryDatabase("ManufacturedSolution") ) ) );

    db->putString("Geometry", geom);
    db->putString("Order", order);
    db->putString("BoundaryType", bc);
    db->putDouble("MinX", 4.);
    db->putDouble("MaxX", 14.);
    db->putDouble("MinY", 40.);
    db->putDouble("MaxY", 140.);
    db->putDouble("MinZ", 400.);
    db->putDouble("MaxZ", 1400.);

    AMP::ManufacturedSolution ms(db);
    size_t nc = ms.getNumberOfInputs();
    size_t na = ms.getNumberOfParameters();

    std::valarray<double> out(10), in(nc), param(na);

    for (size_t i=0; i<na; i++) param[i] = i;
    for (size_t i=0; i<nc; i++) in[i] = i+1;
    ms.evaluate(out, x,y,z, in, param);

    std::string msg=geom+" " + order + " " + bc + " basics";
    ut->passes(msg);
}


int main ( int argc , char **argv )
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;
    AMP::AMP_MPI globalComm = AMP::AMP_MPI(AMP_COMM_WORLD);

    if (globalComm.getRank() == 1) {
        try {
	        testit( &ut, "Brick", "Quadratic", "Neumann", 5.,60.,700.);
            testit( &ut, "Brick", "Quadratic", "Dirichlet-1", 5.,60.,700.);
	        testit( &ut, "Brick", "Quadratic", "Dirichlet-2", 5.,60.,700.);
	        testit( &ut, "Brick", "Cubic", "Neumann", 5.,60.,700.);
	        testit( &ut, "Brick", "Cubic", "Dirichlet-1", 5.,60.,700.);
	        testit( &ut, "Brick", "Cubic", "Dirichlet-2", 5.,60.,700.);
	        testit( &ut, "CylindricalRod", "Cubic", "Neumann", 5.,60.,700.);
	        testit( &ut, "CylindricalRod", "Cubic", "Dirichlet-2-z", 5.,60.,700.);
	        testit( &ut, "CylindricalShell", "Quadratic", "Neumann", 5.,60.,700.);
        } catch (std::exception &err) {
            std::cout << "ERROR: While testing " << argv[0] << ", " << err.what() << std::endl;
            ut.failure("Manufactured Solutions");
        } catch( ... ) {
            std::cout << "ERROR: While testing " << argv[0] << ", " << "An unknown exception was thrown" << std::endl;
            ut.failure("Manufactured Solutions");
        }
    } else { 
        ut.expected_failure("Manufactured Solutions only apply to scalar tests.");
    }
    
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}

