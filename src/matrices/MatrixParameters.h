#ifndef included_AMP_MatrixParameters
#define included_AMP_MatrixParameters

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/MatrixParametersBase.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/Vector.h"

namespace AMP::LinearAlgebra {


/** \class MatrixParameters
 * \brief  A class used to hold basic parameters for a matrix
 */
class MatrixParameters : public MatrixParametersBase
{
public:
    MatrixParameters() = delete;

    /** \brief Constructor
     * \param[in] dofLeft  The DOFManager for the left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$y\f$ is a left vector )
     * \param[in] dofRight The DOFManager for the right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$x\f$ is a right vector )
     * \param[in] comm     Communicator for the matrix
     * \param[in] getRow   Function that finds finds column indices of a row in each block
     */
    explicit MatrixParameters( std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
                               std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
                               const AMP_MPI &comm,
                               const std::function<std::vector<size_t>( size_t )> getRow = {} );

    /** \brief Constructor
     * \param[in] dofLeft  The DOFManager for the left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$y\f$ is a left vector )
     * \param[in] dofRight The DOFManager for the right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$x\f$ is a right vector )
     * \param[in] comm     Communicator for the matrix
     * \param[in] backend  Acceleration backend for matrix operations
     * \param[in] getRow   Function that finds finds column indices of a row in each block
     */
    explicit MatrixParameters( std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
                               std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
                               const AMP_MPI &comm,
                               AMP::Utilities::Backend backend,
                               const std::function<std::vector<size_t>( size_t )> getRow = {} );


    /** \brief Constructor
     * \param[in] dofLeft  The DOFManager for the left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$y\f$ is a left vector )
     * \param[in] dofRight The DOFManager for the right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$x\f$ is a right vector )
     * \param[in] comm     Communicator for the matrix
     * \param[in] varLeft  Pointer to left variable
     * \param[in] varRight Pointer to right variable
     * \param[in] getRow   Function that finds finds column indices of a row in each block
     */
    explicit MatrixParameters( std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
                               std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
                               const AMP_MPI &comm,
                               std::shared_ptr<Variable> varLeft,
                               std::shared_ptr<Variable> varRight,
                               const std::function<std::vector<size_t>( size_t )> getRow = {} );

    /** \brief Constructor
     * \param[in] dofLeft  The DOFManager for the left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$y\f$ is a left vector )
     * \param[in] dofRight The DOFManager for the right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$x\f$ is a right vector )
     * \param[in] comm     Communicator for the matrix
     * \param[in] varLeft  Pointer to left variable
     * \param[in] varRight Pointer to right variable
     * \param[in] backend  Acceleration backend for matrix operations
     * \param[in] getRow   Function that finds finds column indices of a row in each block
     */
    explicit MatrixParameters( std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
                               std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
                               const AMP_MPI &comm,
                               std::shared_ptr<Variable> varLeft,
                               std::shared_ptr<Variable> varRight,
                               AMP::Utilities::Backend backend,
                               const std::function<std::vector<size_t>( size_t )> getRow = {} );

    /** \brief Constructor
     * \param[in] dofLeft       The DOFManager for the left vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$y\f$ is a left vector )
     * \param[in] dofRight      The DOFManager for the right vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$x\f$ is a right vector )
     * \param[in] comm          Communicator for the matrix
     * \param[in] commListLeft  Communication list for the left vector
     * \param[in] commListRight Communication list for the right vector
     * \param[in] getRow        Function that finds finds column indices of a row in each block
     */
    explicit MatrixParameters( std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
                               std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
                               const AMP_MPI &comm,
                               std::shared_ptr<CommunicationList> commListLeft,
                               std::shared_ptr<CommunicationList> commListRight,
                               const std::function<std::vector<size_t>( size_t )> getRow = {} );

    /** \brief Constructor
     * \param[in] dofLeft       The DOFManager for the left vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$y\f$ is a left vector )
     * \param[in] dofRight      The DOFManager for the right vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$x\f$ is a right vector )
     * \param[in] comm          Communicator for the matrix
     * \param[in] commListLeft  Communication list for the left vector
     * \param[in] commListRight Communication list for the right vector
     * \param[in] backend       Acceleration backend for matrix operations
     * \param[in] getRow        Function that finds finds column indices of a row in each block
     */
    explicit MatrixParameters( std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
                               std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
                               const AMP_MPI &comm,
                               std::shared_ptr<CommunicationList> commListLeft,
                               std::shared_ptr<CommunicationList> commListRight,
                               AMP::Utilities::Backend backend,
                               const std::function<std::vector<size_t>( size_t )> getRow = {} );

    /** \brief Constructor
     * \param[in] dofLeft       The DOFManager for the left vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$y\f$ is a left vector )
     * \param[in] dofRight      The DOFManager for the right vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$x\f$ is a right vector )
     * \param[in] comm          Communicator for the matrix
     * \param[in] varLeft       Pointer to left variable
     * \param[in] varRight      Pointer to right variable
     * \param[in] commListLeft  Communication list for the left vector
     * \param[in] commListRight Communication list for the right vector
     * \param[in] getRow        Function that finds finds column indices of a row in each block
     */
    explicit MatrixParameters( std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
                               std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
                               const AMP_MPI &comm,
                               std::shared_ptr<Variable> varLeft,
                               std::shared_ptr<Variable> varRight,
                               std::shared_ptr<CommunicationList> commListLeft,
                               std::shared_ptr<CommunicationList> commListRight,
                               const std::function<std::vector<size_t>( size_t )> getRow = {} );

    /** \brief Constructor
     * \param[in] dofLeft       The DOFManager for the left vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$y\f$ is a left vector )
     * \param[in] dofRight      The DOFManager for the right vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$x\f$ is a right vector )
     * \param[in] comm          Communicator for the matrix
     * \param[in] varLeft       Pointer to left variable
     * \param[in] varRight      Pointer to right variable
     * \param[in] commListLeft  Communication list for the left vector
     * \param[in] commListRight Communication list for the right vector
     * \param[in] backend       Acceleration backend for matrix operations
     * \param[in] getRow        Function that finds finds column indices of a row in each block
     */
    explicit MatrixParameters( std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
                               std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
                               const AMP_MPI &comm,
                               std::shared_ptr<Variable> varLeft,
                               std::shared_ptr<Variable> varRight,
                               std::shared_ptr<CommunicationList> commListLeft,
                               std::shared_ptr<CommunicationList> commListRight,
                               AMP::Utilities::Backend backend,
                               const std::function<std::vector<size_t>( size_t )> getRow = {} );

    //! Destructor
    virtual ~MatrixParameters() = default;

    //! Return the local number of rows
    size_t getLocalNumberOfRows() const;

    //! Return the local number of columns
    size_t getLocalNumberOfColumns() const;

    //! Return the global number of rows
    size_t getGlobalNumberOfRows() const;

    //! Return the global number of columns
    size_t getGlobalNumberOfColumns() const;

    //! Get the bound function that generates column IDs for each row
    const std::function<std::vector<size_t>( size_t )> &getRowFunction() const { return d_getRow; }

    //!  Get the DOFManager for the left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$y\f$ is a
    //!  left vector )
    std::shared_ptr<AMP::Discretization::DOFManager> getLeftDOFManager();

    //!  Get the DOFManager for the right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$x\f$ is a
    //!  right vector )
    std::shared_ptr<AMP::Discretization::DOFManager> getRightDOFManager();

    //!  Get the communication list for the left vector
    std::shared_ptr<CommunicationList> getLeftCommList();

    //!  Get the communication list for the right vector
    std::shared_ptr<CommunicationList> getRightCommList();

    std::string type() const override { return "MatrixParameters"; }

public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    void writeRestart( int64_t fid ) const override;

    /**
     * \brief    Read restart data from file
     * \param fid       File identifier to write
     * \param manager   Restart manager
     */
    MatrixParameters( int64_t fid, AMP::IO::RestartManager *manager );

protected:
    // Generate communication lists internally from the dof managers if not provided
    void generateCommLists();

    // The DOFManager for the left vector ( may be null )
    std::shared_ptr<AMP::Discretization::DOFManager> d_DOFManagerLeft;

    // The DOFManager for the right vector ( may be null )
    std::shared_ptr<AMP::Discretization::DOFManager> d_DOFManagerRight;

    //!  The communication list of a left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$y\f$ is a
    //!  left vector )
    std::shared_ptr<CommunicationList> d_CommListLeft;

    //!  The communication list of a right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$x\f$ is a
    //!  right vector )
    std::shared_ptr<CommunicationList> d_CommListRight;

    //! Function that generates column ids for each row of the matrix
    std::function<std::vector<size_t>( size_t )> d_getRow;
};
} // namespace AMP::LinearAlgebra

#endif
