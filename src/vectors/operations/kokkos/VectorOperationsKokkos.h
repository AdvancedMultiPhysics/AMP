#ifndef included_AMP_VectorOperationsKokkos
#define included_AMP_VectorOperationsKokkos

#include "AMP/AMP_TPLs.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"

#ifdef AMP_USE_KOKKOS

    #include "Kokkos_Core.hpp"

namespace AMP::LinearAlgebra {

/**
 * \brief Vector operations using Kokkos that can be run on host or device
 * \details VectorOperationsKokkos implements all VectorOperations and
 *          holds execution spaces for both host and device (if available).
 *          Computation is done in the space where the given data lives.
 */
template<typename TYPE = double>
class VectorOperationsKokkos : public VectorOperations
{
public:
    // type aliases for execution and view spaces
    using ExecSpaceHost = Kokkos::DefaultHostExecutionSpace;
    using ViewSpaceHost = typename ExecSpaceHost::memory_space;
    #ifdef AMP_USE_DEVICE
    using ExecSpaceDevice  = Kokkos::DefaultExecutionSpace;
    using ViewSpaceDevice  = typename ExecSpaceDevice::memory_space;
    using ViewSpaceManaged = Kokkos::SharedSpace;
    #endif

    // Constructor
    VectorOperationsKokkos() { d_default_ops = std::make_shared<VectorOperationsDefault<TYPE>>(); }

    //! Destructor
    virtual ~VectorOperationsKokkos() = default;

    //! Clone the operations
    virtual std::shared_ptr<VectorOperations> cloneOperations() const override;

public:
    //  functions that operate on VectorData
    std::string VectorOpName() const override { return "VectorOperationsKokkos"; }
    void zero( VectorData &z ) override;
    void setToScalar( const Scalar &alpha, VectorData &z ) override;
    void setRandomValues( VectorData &x ) override;
    void copy( const VectorData &x, VectorData &z ) override;
    void copyCast( const VectorData &x, VectorData &z ) override;
    void scale( const Scalar &alpha, const VectorData &x, VectorData &y ) override;
    void scale( const Scalar &alpha, VectorData &x ) override;
    void add( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void subtract( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void multiply( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void divide( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void reciprocal( const VectorData &x, VectorData &y ) override;
    void linearSum( const Scalar &alpha,
                    const VectorData &x,
                    const Scalar &beta,
                    const VectorData &y,
                    VectorData &z ) override;
    void
    axpy( const Scalar &alpha, const VectorData &x, const VectorData &y, VectorData &z ) override;
    void
    axpby( const Scalar &alpha, const Scalar &beta, const VectorData &x, VectorData &y ) override;
    void abs( const VectorData &x, VectorData &z ) override;
    void addScalar( const VectorData &x, const Scalar &alpha_in, VectorData &y ) override;

    void setMax( const Scalar &val, VectorData &x ) override;
    void setMin( const Scalar &val, VectorData &x ) override;

    Scalar localMin( const VectorData &x ) const override;
    Scalar localMax( const VectorData &x ) const override;
    Scalar localSum( const VectorData &x ) const override;
    Scalar localL1Norm( const VectorData &x ) const override;
    Scalar localL2Norm( const VectorData &x ) const override;
    Scalar localMaxNorm( const VectorData &x ) const override;
    Scalar localDot( const VectorData &x, const VectorData &y ) const override;
    Scalar localMinQuotient( const VectorData &x, const VectorData &y ) const override;
    Scalar localWrmsNorm( const VectorData &x, const VectorData &y ) const override;
    Scalar localWrmsNormMask( const VectorData &x,
                              const VectorData &mask,
                              const VectorData &y ) const override;
    bool localEquals( const VectorData &x,
                      const VectorData &y,
                      const Scalar &tol = 1e-6 ) const override;

protected:
    ExecSpaceHost d_exec_host;
    #ifdef AMP_USE_DEVICE
    ExecSpaceDevice d_exec_device;
    #endif
    std::shared_ptr<VectorOperationsDefault<TYPE>> d_default_ops;
};

#endif
} // namespace AMP::LinearAlgebra

#endif
