// This file contains so definitions and wrapper functions for Tpetra
#ifndef AMP_TpetraHelpers
#define AMP_TpetraHelpers

#include "AMP/utils/UtilityMacros.h"

DISABLE_WARNINGS
#include "Tpetra_Map_decl.hpp"
#include "Tpetra_Vector_decl.hpp"
ENABLE_WARNINGS

#include <memory>

namespace AMP::LinearAlgebra {

class Vector;


/********************************************************
 * Get an Tpetra vector from an AMP vector               *
 ********************************************************/
std::shared_ptr<Tpetra::Vector<>> getTpetra( std::shared_ptr<Vector> vec );


} // namespace AMP::LinearAlgebra

#endif
