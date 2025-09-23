#ifndef included_AMP_AMG_UTIL
#define included_AMP_AMG_UTIL

#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/OperatorParameters.h"

namespace AMP::Solver::AMG {

constexpr inline std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();
template<std::size_t E>
struct extent_storage {
    explicit extent_storage( std::size_t ) {}
    [[nodiscard]] constexpr std::size_t value() const { return E; }
};
template<>
struct extent_storage<dynamic_extent> {
    [[nodiscard]] constexpr std::size_t value() const { return e; }
    std::size_t e;
};


template<class T, std::size_t Extent = dynamic_extent>
struct span {
    using element_type     = T;
    using value_type       = typename std::remove_cv_t<T>;
    using size_type        = std::size_t;
    using difference_type  = std::ptrdiff_t;
    using pointer          = T *;
    using const_pointer    = const T *;
    using reference        = T &;
    using const_reference  = const T &;
    using iterator         = T *;
    using reverse_iterator = std::reverse_iterator<iterator>;

    static constexpr std::size_t extent = Extent;


    template<std::size_t E = Extent,
             class         = typename std::enable_if_t<E == dynamic_extent || E == 0>>
    span() : b( nullptr ), ext{ 0 }
    {
    }

    constexpr span( pointer p, size_type s ) : b( p ), ext{ s } {}

    constexpr iterator begin() const noexcept { return b; }

    constexpr iterator end() const noexcept { return b + size(); }

    constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator( end() ); }

    constexpr reverse_iterator rend() const noexcept { return reverse_iterator( begin() ); }

    constexpr reference front() const { return *b; }

    constexpr reference back() const { return *( b + ( size() - 1 ) ); }

    constexpr reference operator[]( size_type idx ) const { return begin()[idx]; }

    constexpr pointer data() const noexcept { return b; }

    [[nodiscard]] constexpr size_type size() const noexcept { return ext.value(); }

    [[nodiscard]] constexpr size_type size_bytes() const noexcept { return sizeof( T ) * size(); }

    [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }

    template<size_type Count>
    constexpr span<T, Count> first() const noexcept
    {
        return { b, Count };
    }

    constexpr span<T, dynamic_extent> first( size_type count ) const noexcept
    {
        return { data(), count };
    }

    template<size_type Count>
    constexpr span<T, Count> last() const noexcept
    {
        return { data() + ( size() - Count ), Count };
    }

    constexpr span<T, dynamic_extent> last( size_type count ) const noexcept
    {
        return { data() + ( size() - count ), count };
    }

    constexpr span<T, dynamic_extent> subspan( size_t offset,
                                               size_t count = dynamic_extent ) const noexcept
    {
        if ( count == dynamic_extent )
            count = size() - offset;
        AMP_INSIST( offset + count <= size(), "span: invalid subset" );
        return { data() + offset, count };
    }

protected:
    pointer b;
    extent_storage<Extent> ext;
};

template<class A, class T>
using rebind_alloc = typename std::allocator_traits<A>::template rebind_alloc<T>;

template<class Config, class ColID>
struct seq_csr {
    using allocator_type = typename Config::allocator_type;
    using csr_policy     = Config;
    using col_idx_t      = ColID;
    using lidx_t         = typename Config::lidx_t;
    using scalar_t       = typename Config::scalar_t;
    template<class T>
    using vector_type = std::vector<T, rebind_alloc<allocator_type, T>>;

    vector_type<lidx_t> rowptr;
    vector_type<col_idx_t> colind;
    vector_type<scalar_t> values;
};

template<class Config, class ColID = typename Config::lidx_t>
struct par_csr {
    using allocator_type = typename Config::allocator_type;
    using csr_policy     = Config;
    using lidx_t         = typename Config::lidx_t;
    using scalar_t       = typename Config::scalar_t;
    using seq_type       = seq_csr<Config, ColID>;

    par_csr() : d_diag{ std::make_shared<seq_type>() }, d_offd{ std::make_shared<seq_type>() } {}

    seq_type &diag() { return *d_diag; }
    seq_type &offd() { return *d_offd; }

    const seq_type &diag() const { return *d_diag; }
    const seq_type &offd() const { return *d_offd; }

    bool has_offd() const { return offd().rowptr.size() > 0; }

protected:
    std::shared_ptr<seq_type> d_diag, d_offd;
};


template<class Config>
struct coarse_matrix {
    using gidx_t   = typename Config::gidx_t;
    using parcsr_t = par_csr<Config, gidx_t>;

    AMP_MPI comm;
    std::array<gidx_t, 2> diag_extents;
    std::shared_ptr<LinearAlgebra::Variable> right_var, left_var;
    parcsr_t store;
};

template<class C>
struct coarse_operator_parameters : Operator::OperatorParameters {
    coarse_operator_parameters() : AMP::Operator::OperatorParameters( nullptr ) {}

    coarse_matrix<C> matrix;
};

template<class Config>
struct coarse_operator : AMP::Operator::LinearOperator {
    using parcsr_t = par_csr<Config, typename Config::gidx_t>;
    explicit coarse_operator( std::shared_ptr<AMP::Operator::OperatorParameters> params )
        : AMP::Operator::LinearOperator( params )
    {
        auto cop = std::dynamic_pointer_cast<coarse_operator_parameters<Config>>( params );
        AMP_DEBUG_ASSERT( cop );
        storage = cop->matrix.store;
        setMatrix( create_matrix( cop->matrix.comm,
                                  cop->matrix.diag_extents,
                                  cop->matrix.left_var,
                                  cop->matrix.right_var ) );
        setVariables( cop->matrix.left_var, cop->matrix.right_var );
    }

    std::shared_ptr<LinearAlgebra::CSRMatrix<Config>>
    create_matrix( const AMP_MPI &comm,
                   const std::array<typename Config::gidx_t, 2> diag_extents,
                   std::shared_ptr<LinearAlgebra::Variable> left_var,
                   std::shared_ptr<LinearAlgebra::Variable> right_var )
    {
        using seq_type   = typename parcsr_t::seq_type;
        auto make_params = []( seq_type &in ) ->
            typename LinearAlgebra::RawCSRMatrixParameters<Config>::RawCSRLocalMatrixParameters {
                return { in.rowptr.data(), in.colind.data(), in.values.data() };
            };

        auto [diag_params, offd_params] = [&]( auto &...smats ) {
            return std::tuple( make_params( smats )... );
        }( storage.diag(), storage.offd() );

        auto params =
            std::make_shared<LinearAlgebra::RawCSRMatrixParameters<Config>>( diag_extents[0],
                                                                             diag_extents[1],
                                                                             diag_extents[0],
                                                                             diag_extents[1],
                                                                             diag_params,
                                                                             offd_params,
                                                                             comm,
                                                                             left_var,
                                                                             right_var );

        return std::make_shared<LinearAlgebra::CSRMatrix<Config>>( params );
    }

protected:
    parcsr_t storage;
};


template<class Mat>
struct csr_view {
};

template<class Config>
struct csr_view<LinearAlgebra::CSRMatrix<Config>> {
    using allocator_type = typename Config::allocator_type;
    using csr_policy     = Config;
    using csr_data_type  = LinearAlgebra::CSRMatrixData<Config>;
    using value_type     = LinearAlgebra::CSRMatrix<Config>;
    using reference      = const value_type &;
    using pointer        = const value_type *;

    using mask_t   = typename csr_data_type::mask_t;
    using lidx_t   = typename csr_policy::lidx_t;
    using gidx_t   = typename csr_policy::gidx_t;
    using scalar_t = typename csr_policy::scalar_t;

    explicit csr_view( reference p ) : ptr( &p ) {}

    auto diag() const { return csr_ptrs( *( data().getDiagMatrix() ) ); }

    auto offd() const { return csr_ptrs( *( data().getOffdMatrix() ) ); }

    [[nodiscard]] bool has_offd() const { return data().hasOffDiag(); }

    [[nodiscard]] size_t numLocalRows() const { return ptr->numLocalRows(); }

    [[nodiscard]] size_t numGlobalRows() const { return ptr->numGlobalRows(); }

    [[nodiscard]] size_t numGhosts() const { return data().getOffdMatrix()->numUniqueColumns(); }

    void getGhostValues( const LinearAlgebra::Vector &vec, scalar_t *dst ) const
    {
        auto &offd_mat = *( data().getOffdMatrix() );
        if constexpr ( std::is_same_v<size_t, gidx_t> ) {
            size_t *colmap = offd_mat.getColumnMap();
            vec.getGhostValuesByGlobalID( numGhosts(), colmap, dst );
        } else {
            std::vector<size_t> colmap;
            offd_mat.getColumnMap( colmap );
            vec.getGhostValuesByGlobalID( numGhosts(), colmap.data(), dst );
        }
    }

    const auto &data() const
    {
        auto data = std::dynamic_pointer_cast<const csr_data_type>( ptr->getMatrixData() );
        AMP_DEBUG_ASSERT( data );
        return *data;
    }

protected:
    template<class T>
    auto csr_ptrs( const T &local_data ) const
    {
        auto nrows                            = local_data.numLocalRows();
        auto nnz                              = local_data.numberOfNonZeros();
        auto [rowptr, ignore, colind, values] = local_data.getDataFields();
        return std::make_tuple( span<const lidx_t>{ rowptr, static_cast<size_t>( nrows + 1 ) },
                                span<const lidx_t>{ colind, static_cast<size_t>( nnz ) },
                                span<const scalar_t>{ values, static_cast<size_t>( nnz ) } );
    }

    pointer ptr;
};
template<class Config>
csr_view( const LinearAlgebra::CSRMatrix<Config> & ) -> csr_view<LinearAlgebra::CSRMatrix<Config>>;


template<class Config, class ColID>
struct csr_view<par_csr<Config, ColID>> {
    using allocator_type = typename Config::allocator_type;
    using csr_policy     = Config;
    using csr_data_type  = LinearAlgebra::CSRMatrixData<Config>;
    using mask_t         = typename csr_data_type::mask_t;
    using lidx_t         = typename Config::lidx_t;
    using scalar_t       = typename Config::scalar_t;
    using value_type     = par_csr<Config>;

    csr_view( value_type val ) : data{ val } {}

    auto diag() const { return csr_ptrs( data.diag() ); }

    auto offd() const { return csr_ptrs( data.offd() ); }

    [[nodiscard]] bool has_offd() const { return data.has_offd(); }

    [[nodiscard]] size_t numLocalRows() const { return data.diag().rowptr.size() - 1; }

private:
    auto csr_ptrs( const typename value_type::seq_type &v ) const
    {
        return std::make_tuple( span<const lidx_t>( v.rowptr.data(), v.rowptr.size() ),
                                span<const ColID>( v.colind.data(), v.colind.size() ),
                                span<const scalar_t>( v.values.data(), v.values.size() ) );
    }

    value_type data;
};
template<class Config, class ColID>
csr_view( par_csr<Config, ColID> ) -> csr_view<par_csr<Config, ColID>>;

} // namespace AMP::Solver::AMG

#endif
