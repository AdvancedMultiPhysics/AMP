/*
 * \file materials/test/testWaterLibrary.cc
 * \brief test of WaterLibrary.cc class
 */


#include <string>
#include <iostream>
#include <valarray>
#include <vector>

#include "materials/Material.h"
#include "utils/Utilities.h"
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"


void checkConsistency(double h, double p, double T, bool &allCorrect, bool &allConsistent)
{
   using namespace AMP::Materials;
   boost::shared_ptr<AMP::Materials::Material> mat =
      AMP::voodoo::Factory<AMP::Materials::Material>::instance().create("WaterLibrary"); // get water library
   PropertyPtr temperatureProperty = mat->property("Temperature");// temperature property
   PropertyPtr enthalpyProperty	   = mat->property("Enthalpy");   // enthalpy property

   std::map<std::string, boost::shared_ptr<std::vector<double> > > tempMap;
   tempMap.insert( std::make_pair( "enthalpy", new std::vector<double>(1,h) ) );
   tempMap.insert( std::make_pair( "pressure", new std::vector<double>(1,p) ) );
   std::vector<double> tempOutput(1);
   temperatureProperty->evalv(tempOutput, tempMap);
   // check that answer is correct
   if (!AMP::Utilities::approx_equal(tempOutput[0], T, 0.01)) {
      AMP::pout << "Incorrect value: Calculated T: " << tempOutput[0] << ", Actual T: " << T << std::endl;
      allCorrect = false;
   }
   // check that enthalpy function resturns original enthalpy
   std::map<std::string, boost::shared_ptr<std::vector<double> > > hMap;
   hMap.insert( std::make_pair( "temperature", new std::vector<double>(1,tempOutput[0]) ) );
   hMap.insert( std::make_pair( "pressure",    new std::vector<double>(1,p) ) );
   std::vector<double> hOutput(1);
   enthalpyProperty->evalv(hOutput, hMap);
   if (!AMP::Utilities::approx_equal(hOutput[0], h, 0.01)) allConsistent = false;
}

int main ( int argc , char **argv )
{
  AMP::AMPManager::startup(argc, argv);
  AMP::UnitTest ut;

   using namespace AMP::Materials;

   bool good=true;

   try
   {
	   // test constructors for temperature
	   boost::shared_ptr<AMP::Materials::Material> mat =
		   AMP::voodoo::Factory<AMP::Materials::Material>::instance().create("WaterLibrary"); // get water library
           PropertyPtr temperatureProperty	= mat->property("Temperature");			// temperature property
	   PropertyPtr liquidEnthalpyProperty	= mat->property("SaturatedLiquidEnthalpy");	// saturated liquid enthalpy property
	   PropertyPtr volumeProperty		= mat->property("SpecificVolume");		// specific volume property
	   PropertyPtr conductivityProperty	= mat->property("ThermalConductivity");		// thermal conductivity property
	   PropertyPtr viscosityProperty	= mat->property("DynamicViscosity");		// dynamic viscosity property
	   PropertyPtr enthalpyProperty		= mat->property("Enthalpy");			// enthalpy property

	   // test property accessors for temperature
	   string tcname = temperatureProperty->get_name();
	   string tcsorc = temperatureProperty->get_source();
	   AMP::pout << "\n";
	   good = good && tcname == string("WaterLibrary_Temperature");
	   AMP::pout << "Temperature name is " << tcname << "\n";
	   AMP::pout << "Temperature source is " << tcsorc << "\n";
	   vector<string> args = temperatureProperty->get_arguments();
	   good = good && args[0] == "enthalpy";
	   good = good && args[1] == "pressure";
	   AMP::pout << "Temperature property arguments are " << args[0] << " and " << args[1] <<"\n\n";
	   unsigned int nargs = temperatureProperty->get_number_arguments();
	   good = good && nargs == 2;

	   // test material accessors, all arguments present
	   const size_t n=3; // size of input and output arrays for comparison with known thermodynamic values
	   boost::shared_ptr<std::vector<double> > enthalpyInput(new std::vector<double>(n));	// enthalpy input
	   boost::shared_ptr<std::vector<double> > pressureInput(new std::vector<double>(n));	// pressure input
	   boost::shared_ptr<std::vector<double> > temperatureInput(new std::vector<double>(n));	// temperature input
	   boost::shared_ptr<std::vector<double> > temperatureInputEnthalpy(new std::vector<double>(n));// temperature input for enthalpy function
	   boost::shared_ptr<std::vector<double> > densityInput(new std::vector<double>(n));	// density input
	   boost::shared_ptr<std::vector<double> > temperatureIdenticalInput(new std::vector<double>(n));// temperature input array with identical values
	   boost::shared_ptr<std::vector<double> > enthalpyIdenticalInput(new std::vector<double>(n));// enthalpy input array with identical values
	   boost::shared_ptr<std::vector<double> > pressureIdenticalInput(new std::vector<double>(n));	// pressure input array with identical values
	   vector<double> temperatureOutput(n);		// temperature output
	   vector<double> liquidEnthalpyOutput(n);	// saturated liquid enthalpy output
	   vector<double> volumeOutput(n);		// specific volume output
	   vector<double> conductivityOutput(n);	// thermal conductivity output
	   vector<double> viscosityOutput(n);		// dynamic viscosity output
	   vector<double> enthalpyOutput(n);		// enthalpy output
	   vector<double> temperatureIdenticalOutput(n);		// temperature output array with identical values
	   for (size_t i=0; i<n; i++) 
	   {
		(*temperatureIdenticalInput)[i]=400.0;	// temperature: 400 K
		(*enthalpyIdenticalInput)[i]=500.0e3;	// enthalpy: 500 kJ/kg 
		(*pressureIdenticalInput)[i]=1.0e6;	// pressure: 1 MPa
	   }
	(*enthalpyInput)[0] = 500.0e3;
	(*enthalpyInput)[1] = 1.0e6;
	(*enthalpyInput)[2] = 100.0e3;

	(*pressureInput)[0] = 1.0e6;
	(*pressureInput)[1] = 15.0e6;
	(*pressureInput)[2] = 30.0e3;

	(*temperatureInput)[0] = 400.0;
	(*temperatureInput)[1] = 600.0;
	(*temperatureInput)[2] = 300.0;

	(*temperatureInputEnthalpy)[0] = 392.140;
	(*temperatureInputEnthalpy)[1] = 504.658;
	(*temperatureInputEnthalpy)[2] = 297.004;

	(*densityInput)[0] = 937.871;
	(*densityInput)[1] = 659.388;
	(*densityInput)[2] = 996.526;
	   
	   // Block for temporary variables
	   {
		// argument maps for each property function
		   // temperature
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > temperatureArgMap;
		   temperatureArgMap.insert( std::make_pair( "enthalpy", enthalpyInput ) );
		   temperatureArgMap.insert( std::make_pair( "pressure", pressureInput ) );
		   // saturated liquid enthalpy
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > liquidEnthalpyArgMap;
		   liquidEnthalpyArgMap.insert( std::make_pair( "pressure", pressureInput ) );
		   // specific volume
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > volumeArgMap;
		   volumeArgMap.insert( std::make_pair( "enthalpy", enthalpyInput ) );
		   volumeArgMap.insert( std::make_pair( "pressure", pressureInput ) );
		   // thermal conductivity
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > conductivityArgMap;
		   conductivityArgMap.insert( std::make_pair( "temperature", temperatureInput ) );
		   conductivityArgMap.insert( std::make_pair( "density", densityInput ) );
		   // dynamic viscosity
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > viscosityArgMap;
		   viscosityArgMap.insert( std::make_pair( "temperature", temperatureInput ) );
		   viscosityArgMap.insert( std::make_pair( "density", densityInput ) );
		   // enthalpy
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > enthalpyArgMap;
		   enthalpyArgMap.insert( std::make_pair( "temperature", temperatureInputEnthalpy ) );
		   enthalpyArgMap.insert( std::make_pair( "pressure", pressureInput ) );
		   // temperature identical values case
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > temperatureIdenticalArgMap;
		   temperatureIdenticalArgMap.insert( std::make_pair( "enthalpy", enthalpyIdenticalInput ) );
		   temperatureIdenticalArgMap.insert( std::make_pair( "pressure", pressureIdenticalInput ) );
	
		// evaluate properties
		   // temperature
		   temperatureProperty->evalv(temperatureOutput, temperatureArgMap);
		   //saturated liquid enthalpy
		   liquidEnthalpyProperty->evalv(liquidEnthalpyOutput, liquidEnthalpyArgMap);
		   // specific volume 
		   volumeProperty->evalv(volumeOutput, volumeArgMap);
		   // thermal conductivity 
		   conductivityProperty->evalv(conductivityOutput, conductivityArgMap);
		   // dynamic viscosity 
		   viscosityProperty->evalv(viscosityOutput, viscosityArgMap);
		   // enthalpy
		   enthalpyProperty->evalv(enthalpyOutput, enthalpyArgMap);
		   // temperature identical values case
		   std::vector<double> temperatureOutput_mat(temperatureIdenticalOutput);
		   mat->property("Temperature")->evalv(temperatureOutput_mat, temperatureIdenticalArgMap);
		   temperatureProperty->evalv(temperatureIdenticalOutput, temperatureIdenticalArgMap);

		// known values for testing
		double temperatureKnown[n] = {392.140,504.658,297.004};			// temperature
		double liquidEnthalpyKnown[n] = {762.683e3,1610.15e3,289.229e3};	// saturated liquid enthalpy
		double volumeKnown[n] = {0.00105925,0.00119519,0.00100259};		// specific volume
		double conductivityKnown[n] = {0.684097,0.503998,0.610291};		// thermal conductivity
		double viscosityKnown[n] = {0.000218794,7.72970e-5,0.000853838};	// dynamic viscosity
		double enthalpyKnown[n] = {500.0e3,1.0e6,100.0e3};			// enthalpy

		// test property functions against known values
		for (size_t i=0; i<n; i++)
		{
			AMP::pout << "\nValue test " << i << ":\n=====================\n";
			// temperature
			if( !AMP::Utilities::approx_equal(temperatureOutput[i], temperatureKnown[i], 0.01) )
			{
				ut.failure("The answer is wrong.");
				AMP::pout << "The calculated temperature was " << temperatureOutput[i] << " K and actual is ";
				AMP::pout << temperatureKnown[i] << " K\n";
			}
			else AMP::pout << "temperature value is approximately equal to known value.\n";
			// saturated liquid enthalpy
			if( !AMP::Utilities::approx_equal(liquidEnthalpyOutput[i], liquidEnthalpyKnown[i], 0.01) )
			{
				ut.failure("The answer is wrong.");
				AMP::pout << "The calculated saturated liquid enthalpy was " << liquidEnthalpyOutput[i] << " J/kg and actual is ";
				AMP::pout << liquidEnthalpyKnown[i] << " J/kg\n";
			}
			else AMP::pout << "saturated liquid enthalpy value is approximately equal to known value.\n";
			// specific volume
			if( !AMP::Utilities::approx_equal(volumeOutput[i], volumeKnown[i], 0.01) )
			{
				ut.failure("The answer is wrong.");
				AMP::pout << "The calculated specific volume was " << volumeOutput[i] << " m3/kg and actual is ";
				AMP::pout << volumeKnown[i] << " m3/kg\n";
			}
			else AMP::pout << "specific volume value is approximately equal to known value.\n";
			// thermal conductivity
			if( !AMP::Utilities::approx_equal(conductivityOutput[i], conductivityKnown[i], 0.01) )
			{
				ut.failure("The answer is wrong.");
				AMP::pout << "The calculated thermal conductivity was " << conductivityOutput[i] << " W/m-K and actual is ";
				AMP::pout << conductivityKnown[i] << " W/m-K\n";
			}
			else AMP::pout << "thermal conductivity value is approximately equal to known value.\n";
			// dynamic viscosity
			if( !AMP::Utilities::approx_equal(viscosityOutput[i], viscosityKnown[i], 0.01) )
			{
				ut.failure("The answer is wrong.");
				AMP::pout << "The calculated dynamic viscosity was " << viscosityOutput[i] << " Pa-s and actual is ";
				AMP::pout << viscosityKnown[i] << " Pa-s\n";
			}
			else AMP::pout << "dynamic viscosity value is approximately equal to known value.\n";
			// enthalpy
			if( !AMP::Utilities::approx_equal(enthalpyOutput[i], enthalpyKnown[i], 0.01) )
			{
				ut.failure("The answer is wrong.");
				AMP::pout << "The calculated enthalpy was " << enthalpyOutput[i] << " J/kg and actual is ";
				AMP::pout << enthalpyKnown[i] << " J/kg\n";
			}
			else AMP::pout << "enthalpy value is approximately equal to known value.\n";
		}

		// identical values test: compare temperature values against each other
		for (size_t i=0; i<n; i++)
		{
			if (!AMP::Utilities::approx_equal(temperatureIdenticalOutput[i], temperatureOutput_mat[i]))
			{
				AMP::pout << "Identical values temperature test 1 failed: 1st value: " << temperatureIdenticalOutput[i];
				AMP::pout << " and 2nd value: " << temperatureOutput_mat[i] << "\n";
				good = false;
			}
			if (!AMP::Utilities::approx_equal(temperatureIdenticalOutput[0], temperatureIdenticalOutput[i]))
			{
				AMP::pout << "Identical values temperature test 2 failed: 1st value: " << temperatureIdenticalOutput[0];
				AMP::pout << " and 2nd value: " << temperatureIdenticalOutput[i] << "\n";
				good = false;
			}
		}
	}

	AMP::pout << "\nDefault arguments tests:\n============================\n";
	// set defaults
	   // temperature
	   std::vector<double> temperatureDefaults(2);
	   temperatureDefaults[0] = 200e3;	// enthalpy: 200 kJ/kg
	   temperatureDefaults[1] = 0.5e6;	// pressure: 0.5 MPa
	   temperatureProperty->set_defaults(temperatureDefaults);
	   // saturated liquid enthalpy
	   std::vector<double> liquidEnthalpyDefaults(1);
	   liquidEnthalpyDefaults[0] = 0.5e6;	// pressure: 0.5 MPa
	   liquidEnthalpyProperty->set_defaults(liquidEnthalpyDefaults);
	   // specific volume
	   std::vector<double> volumeDefaults(2);
	   volumeDefaults[0] = 200e3;		// enthalpy: 200 kJ/kg
	   volumeDefaults[1] = 0.5e6;		// pressure: 0.5 MPa
	   volumeProperty->set_defaults(volumeDefaults);
	   // thermal conductivity
	   std::vector<double> conductivityDefaults(2);
	   conductivityDefaults[0] = 350;	// temperature: 350 K
	   conductivityDefaults[1] = 973.919;	// density: 973.919 kg/m3
	   conductivityProperty->set_defaults(conductivityDefaults);
	   // dynamic viscosity
	   std::vector<double> viscosityDefaults(2);
	   viscosityDefaults[0] = 350;		// temperature: 350 K
	   viscosityDefaults[1] = 973.919;	// density: 973.919 kg/m3
	   viscosityProperty->set_defaults(viscosityDefaults);
	   // enthalpy
	   std::vector<double> enthalpyDefaults(2);
	   enthalpyDefaults[0] = 320.835;	// temperature: 320.835
	   enthalpyDefaults[1] = 0.5e6;		// pressure: 0.5 MPa
	   enthalpyProperty->set_defaults(enthalpyDefaults);

	// default testing
	   // temperature, one argument: enthalpy
	   {
		   double knownSolution = 392.224;	// T [K]	@ {0.5 MPa, 500 kJ/kg}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   argMap.insert( std::make_pair( "enthalpy", enthalpyIdenticalInput ) );
		   std::vector<double> temperatureOutput_def(temperatureOutput);
		   temperatureProperty->evalv(temperatureOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(temperatureOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Temperature w/ 1 arg incorrect: ";
			AMP::pout << temperatureOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // temperature, no argument
	   {
		   double knownSolution = 320.835;	// T [K]	@ {0.5 MPa, 200 kJ/kg}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   std::vector<double> temperatureOutput_def(temperatureOutput);
		   temperatureProperty->evalv(temperatureOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(temperatureOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Temperature w/ 0 arg incorrect: ";
			AMP::pout << temperatureOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // saturated liquid enthalpy, no argument
	   {
		   double knownSolution = 640.185e3;	// Hf [J/kg]	@ {0.5 MPa}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   std::vector<double> liquidEnthalpyOutput_def(liquidEnthalpyOutput);
		   liquidEnthalpyProperty->evalv(liquidEnthalpyOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(liquidEnthalpyOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Saturated liquid enthalpy w/ 0 arg incorrect: ";
			AMP::pout << liquidEnthalpyOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // specific volume, one argument: enthalpy
	   {
		   double knownSolution = 0.00105962;	// v [m3/kg]	@ {0.5 MPa, 500 kJ/kg}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   argMap.insert( std::make_pair( "enthalpy", enthalpyIdenticalInput ) );
		   std::vector<double> volumeOutput_def(volumeOutput);
		   volumeProperty->evalv(volumeOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(volumeOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Specific volume w/ 1 arg incorrect: ";
			AMP::pout << volumeOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // specific volume, no argument
	   {
		   double knownSolution = 0.00101083;	// v [m3/kg]	@ {0.5 MPa, 200 kJ/kg}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   std::vector<double> volumeOutput_def(volumeOutput);
		   volumeProperty->evalv(volumeOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(volumeOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Specific volume w/ 0 arg incorrect: ";
			AMP::pout << volumeOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // thermal conductivity, one argument: enthalpy
	   {
		   double knownSolution = 0.731;	// k [W/m-K]	@ {400 K, 973.919 kg/m3}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   argMap.insert( std::make_pair( "temperature", temperatureIdenticalInput ) );
		   std::vector<double> conductivityOutput_def(conductivityOutput);
		   conductivityProperty->evalv(conductivityOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(conductivityOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Thermal conductivity w/ 1 arg incorrect: ";
			AMP::pout << conductivityOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // thermal conductivity, no argument
	   {
		   double knownSolution = 0.668247;	// k [W/m-K]	@ {350 K, 973.919 kg/m3}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   std::vector<double> conductivityOutput_def(conductivityOutput);
		   conductivityProperty->evalv(conductivityOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(conductivityOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Thermal conductivity w/ 0 arg incorrect: ";
			AMP::pout << conductivityOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // dynamic viscosity, one argument: enthalpy
	   {
		   double knownSolution = 0.000239;	// u [Pa-s]	@ {400 K, 973.919 kg/m3}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   argMap.insert( std::make_pair( "temperature", temperatureIdenticalInput ) );
		   std::vector<double> viscosityOutput_def(viscosityOutput);
		   viscosityProperty->evalv(viscosityOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(viscosityOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Dynamic viscosity w/ 1 arg incorrect: ";
			AMP::pout << viscosityOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // dynamic viscosity, no argument
	   {
		   double knownSolution = 0.000368895;	// u [Pa-s]	@ {350 K, 973.919 kg/m3}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   std::vector<double> viscosityOutput_def(viscosityOutput);
		   viscosityProperty->evalv(viscosityOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(viscosityOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Dynamic viscosity w/ 0 arg incorrect: ";
			AMP::pout << viscosityOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // enthalpy, one argument: temperature
	   {
		   double knownSolution = 533.121e3;	// h [J/kg]	@ {400 K, 0.5 MPa}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   argMap.insert( std::make_pair( "temperature", temperatureIdenticalInput ) );
		   std::vector<double> enthalpyOutput_def(enthalpyOutput);
		   enthalpyProperty->evalv(enthalpyOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(enthalpyOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Enthalpy w/ 1 arg incorrect: ";
			AMP::pout << enthalpyOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }
	   // enthalpy, no argument
	   {
		   double knownSolution = 200.0e3;	// h [J/kg]	@ {320.835 K, 0.5 MPa}
		   std::map<std::string, boost::shared_ptr<std::vector<double> > > argMap;
		   std::vector<double> enthalpyOutput_def(enthalpyOutput);
		   enthalpyProperty->evalv(enthalpyOutput_def, argMap);
		if (!AMP::Utilities::approx_equal(enthalpyOutput_def[0], knownSolution, 0.01))
		{
			AMP::pout << "Enthalpy w/ 0 arg incorrect: ";
			AMP::pout << enthalpyOutput_def[0] << " vs " << knownSolution << "\n";
			good = false;
		}
	   }

	   if (good) ut.passes("basic tests of Material");
	   else ut.failure("basic tests of Material");

           // Test if thermodyamic properties are consistent
           bool allCorrect = true;
           bool allConsistent = true;
           
           checkConsistency(430.39e3,15.0e6,373.15,allCorrect,allConsistent);
           checkConsistency(1334.4e3,20.0e6,573.15,allCorrect,allConsistent);
           checkConsistency(176.37e3,10.0e6,313.15,allCorrect,allConsistent);
           checkConsistency(507.19e3,5.0e6,393.15,allCorrect,allConsistent);
           checkConsistency(684.01e3,15.0e6,433.15,allCorrect,allConsistent);

	   if (allCorrect) ut.passes("Thermodynamic property value test");
	   else ut.failure("Thermodynamic property value test");
	   if (allConsistent) ut.passes("Thermodynamic property consistency test");
	   else ut.failure("Thermodynamic property consistency test");
   }
   catch( std::exception &err )
   {
     AMP::pout << "ERROR: While testing " << argv[0] << err.what() << std::endl;
     ut.failure("ERROR: While testing");
   }
   catch( ... )
   {
     AMP::pout << "ERROR: While testing " << argv[0] <<  "An unknown exception was thrown" << endl;
     ut.failure("ERROR: While testing");
   }

   ut.report();

   int num_failed = ut.NumFailGlobal();
   AMP::AMPManager::shutdown();
   return num_failed;
}
