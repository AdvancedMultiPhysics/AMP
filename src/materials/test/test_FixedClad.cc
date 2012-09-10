/*
 * test_FixedClad.cc
 *
 *  Created on: Mar 11, 2010
 *      Author: bm, gad
 */

#include "utils/Utilities.h"
#include "utils/UnitTest.h"
#include "utils/AMPManager.h"
#include "materials/Material.h"
#include "materials/VectorProperty.h"
#include "materials/TensorProperty.h"

#include <string>
#include <iostream>
#include <valarray>
using namespace std;

int main ( int argc , char **argv )
{
   AMP::AMPManager::startup(argc, argv);
   AMP::UnitTest ut;

   using namespace AMP::Materials;

   bool good=true;

   try
   {
	   // get material pointer
       Material::shared_ptr mat = AMP::voodoo::Factory<AMP::Materials::Material>::instance().create("FixedClad");
	   PropertyPtr prop = mat->property("ThermalConductivity");

	   // test property accessors
	   string tcname = prop->get_name();
	   string tcsorc = prop->get_source();
	   good = good and tcname == string("FixedClad")+string("_ThermalConductivity");
	   good = good and tcsorc == prop->get_source();
	   std::cout << "thermal conductivity name is " << tcname << "\n";
	   std::cout << "thermal conductivity source is " << tcsorc << "\n";

	   // test material accessors
	   size_t n=10;
	   std::vector<double> &tv=*new std::vector<double>(n);  // temperature vector
	   std::vector<double> &uv=*new std::vector<double>(n);  // concentration
	   std::vector<double> &bv=*new std::vector<double>(n);  // burnup
	   std::vector<double> &prv=*new std::vector<double>(n); // poisson ratio (result)
	   std::vector<double> &tcv=*new std::vector<double>(n); // thermal conductivity (result)
	   for (size_t i=0; i<n; i++) {tv[i]=563.4+i/10.;uv[i]=.05+i/100.,bv[i]=0.+i*100;}  // set input arguments
	   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;  
	   argMap.insert( std::make_pair( "temperance", boost::shared_ptr<std::vector<double> >(&tv) ) );
	   argMap.insert( std::make_pair( "burningman", boost::shared_ptr<std::vector<double> >(&bv) ) );
	   argMap.insert( std::make_pair( "upstart", boost::shared_ptr<std::vector<double> >(&uv) ) );

	   mat->property("PoissonRatio")->evalv(prv, argMap);
	   mat->property("ThermalConductivity")->evalv(tcv, argMap);

	   std::vector<boost::shared_ptr<std::vector<double> > > vfcv(3);
	   for (size_t i=0; i<3; i++) vfcv[i] =
			   boost::shared_ptr<std::vector<double> >(new std::vector<double>(n));

	   boost::shared_ptr<AMP::Materials::VectorProperty<double> > vectorProperty =
			   boost::dynamic_pointer_cast<AMP::Materials::VectorProperty<double> >
	   	   	   	   (mat->property("VectorFickCoefficient"));
	   vectorProperty->set_dimension(3);
	   double vparams[]={1.1, 2.2, 3.3};
	   vectorProperty->set_parameters_and_number(vparams, 3);
	   vectorProperty->evalv(vfcv, argMap);

	   std::vector<std::vector<boost::shared_ptr<std::vector<double> > > > tfcv(3,
			       std::vector<boost::shared_ptr<std::vector<double> > >(3));
	   for (size_t i=0; i<3; i++) for (size_t j=0; j<3; j++) tfcv[i][j] =
			   boost::shared_ptr<std::vector<double> >(new std::vector<double>(n));

	   boost::shared_ptr<AMP::Materials::TensorProperty<double> > tensorProperty =
			   boost::dynamic_pointer_cast<AMP::Materials::TensorProperty<double> >
	   	   	   	   (mat->property("TensorFickCoefficient"));
	   tensorProperty->set_dimensions(std::vector<size_t>(2,3U));
	   double tparams[9]={1.1, 2.2, 3.3, 11., 22., 33., 111., 222., 333.};
	   tensorProperty->set_parameters_and_number(tparams, 9);
	   tensorProperty->evalv(tfcv, argMap);

	   std::vector<double> arg(3);
	   arg[0] = tv[1]; arg[1] = bv[1]; arg[2] = uv[1];
	   tcv[1] = prop->eval( arg );

	   prop->evalv(tcv, argMap);

	   good = good and AMP::Utilities::approx_equal(tcv[1], tcv[n-1]);
	   good = good and AMP::Utilities::approx_equal(tcv[2], tcv[n-1]);
	   valarray<double> params = prop->get_parameters();
	   good = good and AMP::Utilities::approx_equal(tcv[1], params[0]);

	   std::valarray<double> sparams = mat->property("PoissonRatio")->get_parameters();
	   for (size_t i=0; i<n; i++) {
		   good = good and prv[i] == sparams[0];
	   }

	   for (size_t i=0; i<3; i++) {
		   for (size_t j=0; j<n; j++) {
			   double val = (*vfcv[i])[j];
			   good = good and val == vparams[i];
		   }
	   }

	   for (size_t i=0; i<3; i++) {
		   for (size_t j=0; j<3; j++) {
			   for (size_t k=0; k<n; k++) {
				   double val = (*tfcv[i][j])[k];
				   good = good and val == tparams[i*3+j];
			   }
		   }
	   }

	   if (good) ut.passes("basic tests of FixedClad");
	   else ut.failure("basic tests of FixedClad");

	   // test parameter change
	   double param[1]={1.2345};

	   prop->set_parameters(param,1U);
	   double tcs = prop->eval(arg);
	   good = good and AMP::Utilities::approx_equal(tcs, param[0]);

	   mat->property("ThermalConductivity")->set_parameters(param, 1U);
	   mat->property("ThermalConductivity")->evalv(tcv, argMap);
	   good = good and AMP::Utilities::approx_equal(tcv[0], param[0]);

	   if (good) ut.passes("basic tests of parameterized FixedClad");
	   else ut.failure("basic tests of parameterized FixedClad");
   }
   catch( std::exception &err )
   {
     cout << "ERROR: While testing " << argv[0] << err.what() << std::endl;
     ut.failure("ERROR: While testing");
   }
   catch( ... )
   {
     cout << "ERROR: While testing " << argv[0] <<  "An unknown exception was thrown" << endl;
     ut.failure("ERROR: While testing");
   }

   ut.report();

   int num_failed = ut.NumFailGlobal();
   AMP::AMPManager::shutdown();
   return num_failed;
}