// Generate the desired vector factory
#ifndef included_AMP_test_generateVectorFactory
#define included_AMP_test_generateVectorFactory


#include "AMP/vectors/testHelpers/VectorTests.h"


namespace AMP::LinearAlgebra {


/** \brief Generate the desired vector factory
  * \details  This will generate the desired vector factory
  * \param[in] factory      The vector factory to generate:
        SimpleVectorFactory<15,false,double>
        SimpleVectorFactory<45, true,double>
        SimpleVectorFactory<45,true,double,openmp,gpu>
        ArrayVectorFactory<4,10,false,double>
        ArrayVectorFactory<4,10,true,double>
        SimplePetscNativeFactory
        MultiVectorFactory<SMEVFactory, 1, SNPVFactory, 1> MVFactory1;
        MultiVectorFactory<SMEVFactory, 3, SNPVFactory, 2> MVFactory2;
        MultiVectorFactory<MVFactory1, 2, MVFactory2, 2> MVFactory3;
        MultiVectorFactory<SimpleVectorFactory<15,false>,1,SMEVFactory,1> MVFactory1;
        MultiVectorFactory<SimpleVectorFactory<15,false>,3,SMEVFactory,2> MVFactory2;
        MultiVectorFactory<MVFactory1, 2, MVFactory2, 2> MVFactory3;
  */
std::shared_ptr<VectorFactory> generateVectorFactory( const std::string &factory );


/** \brief Get the valid vector factories
 * \details  This will generate a list of all valid factories to test
 */
std::vector<std::string> getAllFactories();

/** \brief Get the valid vector factories
 * \details  This will generate a list of valid SimpleVector factories to test
 */
std::vector<std::string> getSimpleVectorFactories();

/** \brief Get the valid vector factories
 * \details  This will generate a list of valid ArrayVector factories to test
 */
std::vector<std::string> getArrayVectorFactories();

/** \brief Get the valid vector factories
 * \details  This will generate a list of valid native factories to test
 */
std::vector<std::string> getNativeVectorFactories();

/** \brief Get the valid vector factories
 * \details  This will generate a list of valid mesh vector factories to test
 */
std::vector<std::string> getMeshVectorFactories();

/** \brief Get the valid vector factories
 * \details  This will generate a list of valid managed vector factories to test
 */
std::vector<std::string> getManagedVectorFactories();


} // namespace AMP::LinearAlgebra


#endif
