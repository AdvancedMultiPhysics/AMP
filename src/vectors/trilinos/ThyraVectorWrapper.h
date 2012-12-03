#ifndef included_AMP_ThyraVectorWrapper
#define included_AMP_ThyraVectorWrapper

// AMP includes
#include "vectors/Vector.h"


// Trilinos includes
#include "Thyra_VectorDefaultBase_decl.hpp"
#include "Thyra_VectorBase.hpp"
#include "Teuchos_ArrayViewDecl.hpp"
#include "RTOpPack_RTOpT_decl.hpp"
#include <Teuchos_Comm.hpp>


namespace AMP {
namespace LinearAlgebra {


/** \class ThyraVectorWrapper
  * \brief  Wrapper for an AMP vector in Thyra
  * \details  This allows us to safely wrap an AMP vector
  *   in a thyra vector for use within Trilinos. 
  */
class ThyraVectorWrapper : public Thyra::VectorBase<double>
{
public:

    // Default constructor
    ThyraVectorWrapper( AMP::LinearAlgebra::Vector::shared_ptr vec );

    //! Destructor
    virtual ~ThyraVectorWrapper();

    //! Get the underlying AMP vector
    Vector::shared_ptr getVec() { return d_vec; }

    //! Get the underlying AMP vector
    Vector::const_shared_ptr getVec() const { return d_vec; }

    // Functions derived from Thyra::LinearOpBase
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double> > range() const;
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double> > domain() const;

    // Functions derived from Thyra::MultiVectorBase
    virtual Teuchos::RCP<Thyra::MultiVectorBase<double> > clone_mv() const;

    // Functions derived from Thyra::VectorBase
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double> > space() const;
    virtual Teuchos::RCP<Thyra::VectorBase<double> > clone_v() const;

protected:

    // Functions derived from Thyra::LinearOpBase
    virtual bool opSupportedImpl(Thyra::EOpTransp M_trans) const;
    virtual void applyImpl(const Thyra::EOpTransp M_trans, const Thyra::MultiVectorBase<double> &X, 
        const Teuchos::Ptr<Thyra::MultiVectorBase<double> > &Y, const double alpha, const double beta) const;

    // Functions derived from Thyra::MultiVectorBase
    virtual Teuchos::RCP<Thyra::VectorBase<double> > nonconstColImpl(Teuchos::Ordinal j);
    virtual Teuchos::RCP<const Thyra::MultiVectorBase<double> > contigSubViewImpl(const Teuchos::Range1D &colRng) const;
    virtual Teuchos::RCP<Thyra::MultiVectorBase<double> > nonconstContigSubViewImpl(const Teuchos::Range1D &colRng);
    virtual Teuchos::RCP<const Thyra::MultiVectorBase<double> > nonContigSubViewImpl(const Teuchos::ArrayView< const int > &cols) const;
    virtual Teuchos::RCP<Thyra::MultiVectorBase<double> > nonconstNonContigSubViewImpl(const Teuchos::ArrayView< const int > &cols);
    virtual void mvMultiReductApplyOpImpl( const RTOpPack::RTOpT<double> &primary_op, 
        const Teuchos::ArrayView< const Teuchos::Ptr< const Thyra::MultiVectorBase<double> > > &multi_vecs, 
        const Teuchos::ArrayView< const Teuchos::Ptr< Thyra::MultiVectorBase<double> > > &targ_multi_vecs, 
        const Teuchos::ArrayView< const Teuchos::Ptr< RTOpPack::ReductTarget > > &reduct_objs, 
        const Teuchos::Ordinal primary_global_offset ) const;
    virtual void mvSingleReductApplyOpImpl( const RTOpPack::RTOpT<double> &primary_op, 
        const RTOpPack::RTOpT<double> &secondary_op, 
        const Teuchos::ArrayView< const Teuchos::Ptr< const Thyra::MultiVectorBase<double> > > &multi_vecs, 
        const Teuchos::ArrayView< const Teuchos::Ptr< Thyra::MultiVectorBase<double> > > &targ_multi_vecs, 
        const Teuchos::Ptr< RTOpPack::ReductTarget > &reduct_obj, 
        const Teuchos::Ordinal primary_global_offset ) const;
    virtual void acquireDetachedMultiVectorViewImpl( const Teuchos::Range1D &rowRng, const Teuchos::Range1D &colRng, 
        RTOpPack::ConstSubMultiVectorView<double> *sub_mv ) const;
    virtual void releaseDetachedMultiVectorViewImpl(RTOpPack::ConstSubMultiVectorView<double> *sub_mv) const;
    virtual void acquireNonconstDetachedMultiVectorViewImpl(const Teuchos::Range1D &rowRng, const Teuchos::Range1D &colRng, 
        RTOpPack::SubMultiVectorView<double> *sub_mv );
    virtual void commitNonconstDetachedMultiVectorViewImpl( RTOpPack::SubMultiVectorView<double> *sub_mv );

    // Functions derived from Thyra::VectorBase
    virtual void applyOpImpl( const RTOpPack::RTOpT<double> &op, 
        const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::VectorBase<double> > > &vecs, 
        const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::VectorBase<double> > > &targ_vecs, 
        const Teuchos::Ptr<RTOpPack::ReductTarget> &reduct_obj, 
        const Teuchos::Ordinal global_offset) const;
    virtual void acquireDetachedVectorViewImpl(const Teuchos::Range1D &rng, RTOpPack::ConstSubVectorView<double> *sub_vec) const;
    virtual void releaseDetachedVectorViewImpl(RTOpPack::ConstSubVectorView<double> *sub_vec) const;
    virtual void acquireNonconstDetachedVectorViewImpl(const Teuchos::Range1D &rng, RTOpPack::SubVectorView<double> *sub_vec);
    virtual void commitNonconstDetachedVectorViewImpl(RTOpPack::SubVectorView<double> *sub_vec);
    virtual void setSubVectorImpl(const RTOpPack::SparseSubVectorT<double> &sub_vec);

    // Internal data
    AMP::LinearAlgebra::Vector::shared_ptr d_vec;

private:

    // Private constructor
    ThyraVectorWrapper() {}

    // Comm
    Teuchos::Comm<RTOpPack::index_type> *d_comm;

};


}
}

#endif
