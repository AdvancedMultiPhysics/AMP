#ifndef included_AMP_ThyraVectorSpaceWrapper
#define included_AMP_ThyraVectorSpaceWrapper

// Trilinos includes
#include "Thyra_VectorSpaceBase_decl.hpp"
#include "Thyra_VectorDefaultBase_decl.hpp"

// AMP includes
#include "discretization/DOF_Manager.h"


namespace AMP {
namespace LinearAlgebra {


/** \class ThyraVectorSpaceWrapper
  * \brief  Wrapper a VectorSpace in AMP
  * \details  This function is used to allow us to safely wrap an AMP vector
  *   in a thyra vector for use within Trilinos. 
  */
class ThyraVectorSpaceWrapper : public Thyra::VectorSpaceBase<double>
{
public:

    //! Default constuctor
    ThyraVectorSpaceWrapper( AMP::Discretization::DOFManager::shared_ptr dofs );

    //! Destructor
    virtual ~ThyraVectorSpaceWrapper();

    // Virtual functions inherited from VectorSpaceBase
    virtual Teuchos::Ordinal dim() const;
    virtual bool isCompatible(const Thyra::VectorSpaceBase<double> &vecSpc) const;
    virtual Teuchos::RCP<const Thyra::VectorSpaceFactoryBase<double> > smallVecSpcFcty() const;
    virtual double scalarProd(const Thyra::VectorBase<double> &x, const Thyra::VectorBase<double> &y) const;

protected:

    // Virtual functions inherited from VectorSpaceBase
    virtual Teuchos::RCP<Thyra::VectorBase<double> >  createMember() const;
    virtual Teuchos::RCP<Thyra::MultiVectorBase <double> >  createMembers(int numMembers) const;
    virtual Teuchos::RCP<Thyra::VectorBase<double> >  createMemberView(const RTOpPack::SubVectorView<double> &raw_v) const;
    virtual Teuchos::RCP<const Thyra::VectorBase <double> >  createMemberView(const RTOpPack::ConstSubVectorView<double> &raw_v) const;
    virtual Teuchos::RCP<Thyra::MultiVectorBase  <double> >  createMembersView(const RTOpPack::SubMultiVectorView<double> &raw_mv) const;
    virtual Teuchos::RCP<const Thyra::MultiVectorBase<double> >  createMembersView(const RTOpPack::ConstSubMultiVectorView<double> &raw_mv) const;
    virtual void scalarProdsImpl(const Thyra::MultiVectorBase<double> &X, const Thyra::MultiVectorBase<double> &Y, const Teuchos::ArrayView<double> &scalarProds) const;

    // Local data
    AMP::Discretization::DOFManager::shared_ptr d_dofs;

private:

    // Private constuctor
    ThyraVectorSpaceWrapper( ) {}

};


}
}

#endif

