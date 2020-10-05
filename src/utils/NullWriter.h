#ifndef included_AMP_NullWriter
#define included_AMP_NullWriter

#include "AMP/utils/Writer.h"


namespace AMP {
namespace Utilities {


/**
 * \class NullWriter
 * \brief A class used to abstract away reading/writing files for visualization
 * \details  This class provides default routines that do nothing.
 */
class NullWriter : public AMP::Utilities::Writer
{
public:
    //!  Default constructor
    NullWriter() {}

    //!  Default destructor
    virtual ~NullWriter() {}

    // Inherited functions
    std::string getExtension() override { return ""; }
    void readFile( const std::string & ) override{};
    void writeFile( const std::string &, size_t, double = 0 ) override {}
#ifdef USE_AMP_MESH
    void registerMesh( AMP::Mesh::Mesh::shared_ptr,
                       int                 = 1,
                       const std::string & = std::string() ) override
    {
    }
#endif
#ifdef USE_AMP_VECTORS
    void registerVector( AMP::LinearAlgebra::Vector::shared_ptr,
                         AMP::Mesh::Mesh::shared_ptr,
                         AMP::Mesh::GeomType,
                         const std::string & = "" ) override
    {
    }
    void registerVector( AMP::LinearAlgebra::Vector::shared_ptr ) override {}
#endif
#ifdef USE_AMP_MATRICES
    void registerMatrix( AMP::LinearAlgebra::Matrix::shared_ptr ) override {}
#endif
};
} // namespace Utilities
} // namespace AMP

#endif
