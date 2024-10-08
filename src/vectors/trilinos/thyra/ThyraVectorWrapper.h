#ifndef included_AMP_ThyraVectorWrapper
#define included_AMP_ThyraVectorWrapper

// AMP includes
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/trilinos/thyra/ThyraVector.h"


// Trilinos includes
DISABLE_WARNINGS
#include "Thyra_VectorDefaultBase.hpp"
#include <Teuchos_Comm.hpp>
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


/** \class ThyraVectorWrapper
 * \brief  Wrapper for an AMP vector in Thyra
 * \details  This allows us to safely wrap an AMP vector
 *   in a thyra vector for use within Trilinos.
 */
class ThyraVectorWrapper : public Thyra::VectorDefaultBase<double>
{
public:
    // Default constructor
    explicit ThyraVectorWrapper( const std::vector<AMP::LinearAlgebra::Vector::shared_ptr> &vecs );

    //! Destructor
    virtual ~ThyraVectorWrapper();

    //! Copy constructor
    ThyraVectorWrapper( const ThyraVectorWrapper & ) = delete;

    //! Assignment operator
    ThyraVectorWrapper &operator=( const ThyraVectorWrapper & ) = delete;

    //! Get the underlying AMP vector
    Vector::shared_ptr getVec( int i ) { return d_vecs[i]; }

    //! Get the underlying AMP vector
    Vector::const_shared_ptr getVec( int i ) const { return d_vecs[i]; }

    //! Get the number of duplicate vectors stored
    size_t numVecs() const { return d_vecs.size(); }

    //! Get the number of rows
    size_t numRows() const;

    //! Get the number of columns
    size_t numColumns() const;

    //! Get the DOF Manager for the vector (all vectors must share compatible DOFManagers)
    std::shared_ptr<const AMP::Discretization::DOFManager> getDOFManager() const
    {
        return d_vecs[0]->getDOFManager();
    }

    // Functions derived from Thyra::LinearOpBase
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double>> range() const override;
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double>> domain() const override;

    // Functions derived from Thyra::MultiVectorBase
    virtual Teuchos::RCP<Thyra::MultiVectorBase<double>> clone_mv() const override;

    // Functions derived from Thyra::VectorBase
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double>> space() const override;
    virtual Teuchos::RCP<Thyra::VectorBase<double>> clone_v() const override;

protected:
    // Protected constructor for view of subset of columns
    explicit ThyraVectorWrapper( const std::vector<AMP::LinearAlgebra::Vector::shared_ptr> &vecs,
                                 const std::vector<size_t> &cols,
                                 size_t N_cols );
    void initialize( const std::vector<AMP::LinearAlgebra::Vector::shared_ptr> &vecs,
                     const std::vector<size_t> &cols,
                     size_t N_cols );

    // Functions derived from Thyra::LinearOpBase
    virtual bool opSupportedImpl( Thyra::EOpTransp M_trans ) const override;
    virtual void applyImpl( const Thyra::EOpTransp M_trans,
                            const Thyra::MultiVectorBase<double> &X,
                            const Teuchos::Ptr<Thyra::MultiVectorBase<double>> &Y,
                            const double alpha,
                            const double beta ) const override;

    void assignImpl( double alpha ) override;

    // Functions derived from Thyra::MultiVectorBase
    virtual Teuchos::RCP<Thyra::VectorBase<double>> nonconstColImpl( Teuchos::Ordinal j ) override;
    virtual Teuchos::RCP<const Thyra::MultiVectorBase<double>>
    contigSubViewImpl( const Teuchos::Range1D &colRng ) const override;
    virtual Teuchos::RCP<Thyra::MultiVectorBase<double>>
    nonconstContigSubViewImpl( const Teuchos::Range1D &colRng ) override;
    virtual Teuchos::RCP<const Thyra::MultiVectorBase<double>>
    nonContigSubViewImpl( const Teuchos::ArrayView<const int> &cols ) const override;
    virtual Teuchos::RCP<Thyra::MultiVectorBase<double>>
    nonconstNonContigSubViewImpl( const Teuchos::ArrayView<const int> &cols ) override;
    virtual void mvMultiReductApplyOpImpl(
        const RTOpPack::RTOpT<double> &primary_op,
        const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<double>>>
            &multi_vecs,
        const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::MultiVectorBase<double>>>
            &targ_multi_vecs,
        const Teuchos::ArrayView<const Teuchos::Ptr<RTOpPack::ReductTarget>> &reduct_objs,
        const Teuchos::Ordinal primary_global_offset ) const override;
    virtual void mvSingleReductApplyOpImpl(
        const RTOpPack::RTOpT<double> &primary_op,
        const RTOpPack::RTOpT<double> &secondary_op,
        const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<double>>>
            &multi_vecs,
        const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::MultiVectorBase<double>>>
            &targ_multi_vecs,
        const Teuchos::Ptr<RTOpPack::ReductTarget> &reduct_obj,
        const Teuchos::Ordinal primary_global_offset ) const override;
    virtual void acquireDetachedMultiVectorViewImpl(
        const Teuchos::Range1D &rowRng,
        const Teuchos::Range1D &colRng,
        RTOpPack::ConstSubMultiVectorView<double> *sub_mv ) const override;
    virtual void releaseDetachedMultiVectorViewImpl(
        RTOpPack::ConstSubMultiVectorView<double> *sub_mv ) const override;
    virtual void acquireNonconstDetachedMultiVectorViewImpl(
        const Teuchos::Range1D &rowRng,
        const Teuchos::Range1D &colRng,
        RTOpPack::SubMultiVectorView<double> *sub_mv ) override;
    virtual void commitNonconstDetachedMultiVectorViewImpl(
        RTOpPack::SubMultiVectorView<double> *sub_mv ) override;

    // Functions derived from Thyra::VectorBase
    virtual void applyOpImpl(
        const RTOpPack::RTOpT<double> &op,
        const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::VectorBase<double>>> &vecs,
        const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::VectorBase<double>>> &targ_vecs,
        const Teuchos::Ptr<RTOpPack::ReductTarget> &reduct_obj,
        const Teuchos::Ordinal global_offset ) const override;
    virtual void
    acquireDetachedVectorViewImpl( const Teuchos::Range1D &rng,
                                   RTOpPack::ConstSubVectorView<double> *sub_vec ) const override;
    virtual void
    releaseDetachedVectorViewImpl( RTOpPack::ConstSubVectorView<double> *sub_vec ) const override;
    virtual void
    acquireNonconstDetachedVectorViewImpl( const Teuchos::Range1D &rng,
                                           RTOpPack::SubVectorView<double> *sub_vec ) override;
    virtual void
    commitNonconstDetachedVectorViewImpl( RTOpPack::SubVectorView<double> *sub_vec ) override;
    virtual void setSubVectorImpl( const RTOpPack::SparseSubVectorT<double> &sub_vec ) override;

    // Internal data
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> d_vecs;
    std::vector<size_t> d_cols;
    size_t d_N_cols = 0;

private:
    // Private constructor
    ThyraVectorWrapper() = default;

    // Comm
    Teuchos::Comm<RTOpPack::index_type> *d_comm = nullptr;

    // Get shared_ptr to *this
    std::shared_ptr<const ThyraVectorWrapper> shared_from_this() const;

    friend class ThyraVector;
};
} // namespace AMP::LinearAlgebra

#endif
