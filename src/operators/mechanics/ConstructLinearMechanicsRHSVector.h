
#ifndef included_AMP_ConstructLinearMechanicsRHSVector
#define included_AMP_ConstructLinearMechanicsRHSVector

#include "materials/Material.h"

#include "utils/shared_ptr.h"

#include "utils/Utilities.h"
#include "utils/Database.h"

/* Libmesh files */
#include "libmesh/enum_order.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/auto_ptr.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/elem.h"
#include "libmesh/fe_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/quadrature.h"

#include <string>
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <vector>

#include "ampmesh/Mesh.h"

void computeTemperatureRhsVector( AMP::Mesh::Mesh::shared_ptr mesh, 
    AMP::shared_ptr<AMP::Database> input_db, 
    AMP::LinearAlgebra::Variable::shared_ptr temperatureVar, 
    AMP::LinearAlgebra::Variable::shared_ptr displacementVar,
    const AMP::shared_ptr<AMP::LinearAlgebra::Vector> &currTemperatureVec, 
    const AMP::shared_ptr<AMP::LinearAlgebra::Vector> &prevTemperatureVec, 
    AMP::LinearAlgebra::Vector::shared_ptr rhsVec);


#endif


