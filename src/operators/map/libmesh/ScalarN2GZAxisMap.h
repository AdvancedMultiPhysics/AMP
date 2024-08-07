#ifndef included_AMP_ScalarN2GZAxisMap
#define included_AMP_ScalarN2GZAxisMap

#include "AMP/discretization/createLibmeshElements.h"
#include "AMP/operators/map/Map3to1to3.h"
#include "AMP/operators/map/Map3to1to3Parameters.h"


namespace AMP::Operator {


typedef AMP::Operator::Map3to1to3Parameters ScalarN2GZAxisMapParameters;


/**
 * \class  ScalarN2GZAxisMap
 * \brief  A class used to reduce a 3D problem to 1D, transfer the solution, and map back to 3D
 * \details  This class inherites from Map3to1to3, and performs a reduction from 3D to 1D,
 *    transfers the solution, then maps back to 3D.  It accomplishes this by taking the average
 *    value for each point along the z-axis, creating a 1D function of z, then transfering that
 *    solution.  To behave correctly, the different nodes must be aligned in the z-direction.
 */
class ScalarN2GZAxisMap : public AMP::Operator::Map3to1to3
{
public:
    /** \brief  Returns true if MapType = "ScalarN2GZAxis"
     * \param[in] s  A string extracted from the MapType line in a MeshToMeshMap db
     * \return  True iff s == "ScalarN2GZAxis"
     */
    static bool validMapType( const std::string &s );

    /** \brief  Typedef to identify the parameters class of this operator
     */
    typedef ScalarN2GZAxisMapParameters Parameters;

    //!  The base tag used in communication.
    static const int CommTagBase = 20000;

    /** \brief   Standard constructor
     * \param[in] params  Input parameters
     */
    explicit ScalarN2GZAxisMap( std::shared_ptr<const AMP::Operator::OperatorParameters> params );

    //! Destructor
    virtual ~ScalarN2GZAxisMap();

    //! Return the name of the operator
    std::string type() const override { return "ScalarN2GZAxisMap"; }

protected:
    // Implimented buildMap routine
    virtual std::multimap<double, double> buildMap( AMP::LinearAlgebra::Vector::const_shared_ptr,
                                                    const std::shared_ptr<AMP::Mesh::Mesh>,
                                                    const AMP::Mesh::MeshIterator & ) override;

    // Implimented buildReturn routine
    virtual void buildReturn( AMP::LinearAlgebra::Vector::shared_ptr,
                              const std::shared_ptr<AMP::Mesh::Mesh>,
                              const AMP::Mesh::MeshIterator &,
                              const std::map<double, double> & ) override;

    // Function to return the coordinates of the gauss points
    AMP::LinearAlgebra::Vector::const_shared_ptr getGaussPoints( const AMP::Mesh::MeshIterator & );

    // Internal vector with the z-coordinates of the gauss points
    AMP::LinearAlgebra::Vector::const_shared_ptr d_z_coord1;
    AMP::LinearAlgebra::Vector::const_shared_ptr d_z_coord2;

private:
    Discretization::createLibmeshElements libmeshElements;
};


} // namespace AMP::Operator

#endif
