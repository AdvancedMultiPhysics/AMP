/*
 * apply_tests.h
 *
 *  Created on: May 27, 2010
 *      Author: gad
 */

#ifndef APPLY_TESTS_H_
#define APPLY_TESTS_H_

#include "AMP/operators/Operator.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include <string>

void adjust( AMP::LinearAlgebra::Vector::shared_ptr vec,
             const double shift = 301.,
             const double scale = 1.0 );

void adjust( AMP::LinearAlgebra::Vector::shared_ptr vec,
             const double *shift,
             const double *scale,
             const size_t nshift );

void applyTests( AMP::UnitTest *ut,
                 const std::string &msgPrefix,
                 AMP::shared_ptr<AMP::Operator::Operator> testOperator,
                 AMP::LinearAlgebra::Vector::shared_ptr rhsVec,
                 AMP::LinearAlgebra::Vector::shared_ptr solVec,
                 AMP::LinearAlgebra::Vector::shared_ptr resVec,
                 const double shift = 301.,
                 const double scale = 1.0 );

void applyTests( AMP::UnitTest *ut,
                 const std::string &msgPrefix,
                 AMP::shared_ptr<AMP::Operator::Operator> testOperator,
                 AMP::LinearAlgebra::Vector::shared_ptr rhsVec,
                 AMP::LinearAlgebra::Vector::shared_ptr solVec,
                 AMP::LinearAlgebra::Vector::shared_ptr resVec,
                 const double *shift,
                 const double *scale,
                 const size_t nshift );

#include "applyTests.hpp"

#endif /* APPLY_TESTS_H_ */
