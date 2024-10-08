#ifndef included_AMP_ThyraVectorFactor
#define included_AMP_ThyraVectorFactor


#include "AMP/AMP_TPLs.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/testHelpers/VectorTests.h"


/// \cond UNDOCUMENTED


namespace AMP::LinearAlgebra {


class NativeThyraFactory : public VectorFactory
{
public:
    NativeThyraFactory() {}
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override;
    std::string name() const override { return "NativeThyraFactory"; }
};


class ManagedThyraFactory : public VectorFactory
{
public:
    explicit ManagedThyraFactory( std::shared_ptr<VectorFactory> factory ) : d_factory( factory ) {}
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override;
    std::string name() const override { return "ManagedThyraFactory<" + d_factory->name() + ">"; }

private:
    ManagedThyraFactory();
    std::shared_ptr<VectorFactory> d_factory;
};


class ManagedNativeThyraFactory : public VectorFactory
{
public:
    explicit ManagedNativeThyraFactory( std::shared_ptr<VectorFactory> factory )
        : d_factory( factory )
    {
    }
    virtual AMP::LinearAlgebra::Vector::shared_ptr getVector() const override;
    virtual std::string name() const override
    {
        return "ManagedNativeThyraFactory<" + d_factory->name() + ">";
    }

private:
    ManagedNativeThyraFactory();
    std::shared_ptr<VectorFactory> d_factory;
};

#ifdef AMP_USE_TRILINOS_BELOS
void testBelosThyraVector( AMP::UnitTest &utils, const VectorFactory &factory );
#endif

} // namespace AMP::LinearAlgebra

/// \endcond

#endif
