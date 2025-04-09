#ifndef included_AMP_AMG_STRENGTH
#define included_AMP_AMG_STRENGTH

#include <vector>
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/solvers/amg/Util.h"

namespace AMP::Solver::AMG {

template<class Mat>
struct Strength {
	explicit Strength( csr_view<Mat> A );
	using lidx_t = typename csr_view<Mat>::lidx_t;
	using scalar_t = typename csr_view<Mat>::scalar_t;

	constexpr auto diag_row(lidx_t r) {
		return d_diag.row(r);
	}

	constexpr auto diag_row(lidx_t r) const {
		return d_diag.row(r);
	}

	constexpr auto offd_row(lidx_t r) {
		return d_offd.row(r);
	}

	constexpr auto offd_row(lidx_t r) const {
		return d_offd.row(r);
	}

	constexpr lidx_t numLocalRows() const {
		return d_diag.rowptr.size() - 1;
	}

	bool is_strong( lidx_t i, lidx_t j ) const {
		auto search = [=](auto ptrs) {
			auto [rowptr, colind, values] = ptrs;
			for (lidx_t off = rowptr[i]; off < rowptr[i + 1]; ++off) {
				if (j == colind[off] && values[off]) return true;
			}
			return false;
		};

		return search(diag()) || search(offd());
	}

	template<class F>
	void do_strong( lidx_t r, F && f ) const {
		auto loop = [=](auto ptrs) {
			auto [rowptr, colind, values] = ptrs;
			for (lidx_t off = rowptr[r]; off < rowptr[r + 1]; ++off) {
				if (values[off]) f(colind[off]);
			}
		};
		loop(diag());
		// loop(offd());
	}

	template<class F>
	void do_strong_val( lidx_t r, F && f ) const {
		auto loop = [=](auto ptrs, auto mat_values) {
			auto [rowptr, colind, values] = ptrs;
			for (lidx_t off = rowptr[r]; off < rowptr[r + 1]; ++off) {
				if (values[off]) f(colind[off], mat_values[off]);
			}
		};
		loop(diag(), d_diag.mat_values);
		// loop(offd());
	}

private:
	struct storage {
		using alloc_t = typename std::allocator_traits<typename csr_view<Mat>::allocator_type>::template rebind_alloc<bool>;
		using value_type = std::vector<bool, alloc_t>;

		value_type         values;
		span<const lidx_t> rowptr;
		span<const lidx_t> colind;
		span<const scalar_t> mat_values;

		struct reference {
			std::vector<bool, alloc_t> * ptr;
			lidx_t offset;
			constexpr typename std::vector<bool, alloc_t>::reference operator[](lidx_t i) {
				return (*ptr)[offset + i];
			}
			constexpr typename std::vector<bool, alloc_t>::const_reference operator[](lidx_t i) const {
				return (*ptr)[offset + i];
			}
		};
		constexpr reference row(lidx_t r) {
			return {&values, rowptr[r]};
		}

		constexpr reference row(lidx_t r) const {
			return {&values, rowptr[r]};
		}
	} d_diag, d_offd;
public:
	using rep_type = std::tuple<span<const lidx_t>, span<const lidx_t>, const typename storage::value_type &>;
	auto diag() const {
		return rep_type{d_diag.rowptr, d_diag.colind, d_diag.values};
	}

	auto offd() const {
		return rep_type{d_offd.rowptr, d_offd.colind, d_offd.values};
	}
};

enum class norm { abs, min };
template<norm norm_type>
struct classical_strength;

template<class StrengthPolicy, class Mat>
Strength<Mat>
compute_soc( csr_view<Mat> A, float threshold );

}

#endif
