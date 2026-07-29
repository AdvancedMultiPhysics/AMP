#ifndef included_AMP_Flags
#define included_AMP_Flags

#include <type_traits>

namespace AMP::Utilities {

/**
 * \brief Store and query a set of enum-valued bit flags.
 *
 * Flags wraps an enum type whose enumerators represent bit masks, typically
 * powers of two.  The stored value is the bitwise OR of the raised flags.
 *
 * \tparam FlagSet enum type used to name the individual flags.
 */
template<class FlagSet>
struct Flags {
    static_assert( std::is_enum_v<FlagSet>, "Flags requires an enum type" );

    //! Unsigned integer type used to store the bit mask.
    using utype = std::make_unsigned_t<std::underlying_type_t<FlagSet>>;

    //! Construct an empty flag set.
    constexpr Flags() noexcept = default;

    //! Construct from a raw bit mask.
    constexpr explicit Flags( utype f ) noexcept : d_flags( f ) {}

    //! Construct with one or more flags already raised.
    template<class First,
             class... Rest,
             std::enable_if_t<std::is_same_v<FlagSet, std::decay_t<First>> &&
                                  ( std::is_same_v<FlagSet, std::decay_t<Rest>> && ... ),
                              int> = 0>
    constexpr Flags( First first, Rest... rest ) noexcept : d_flags( mask( first, rest... ) )
    {
    }

    //! Raise all given flags.
    template<class... Fs>
    constexpr void raise( Fs... fs ) noexcept
    {
        d_flags |= mask( fs... );
    }

    //! Lower all given flags.
    template<class... Fs>
    constexpr void lower( Fs... fs ) noexcept
    {
        d_flags &= ~mask( fs... );
    }

    //! Toggle all given flags.
    template<class... Fs>
    constexpr void toggle( Fs... fs ) noexcept
    {
        d_flags ^= mask( fs... );
    }

    //! Lower every flag.
    constexpr void reset() noexcept { d_flags = 0; }

    //! Return true if every given flag is raised.
    template<class... Fs>
    constexpr bool raised( Fs... fs ) const noexcept
    {
        auto m = mask( fs... );
        return ( d_flags & m ) == m;
    }

    //! Return true if at least one given flag is raised.
    template<class... Fs>
    constexpr bool any( Fs... fs ) const noexcept
    {
        auto m = mask( fs... );
        return ( d_flags & m ) != 0;
    }

    //! Return true if none of the given flags are raised.
    template<class... Fs>
    constexpr bool none( Fs... fs ) const noexcept
    {
        return ( d_flags & mask( fs... ) ) == 0;
    }

    //! Return true if no flags are raised.
    constexpr bool empty() const noexcept { return d_flags == 0; }

    //! Return the raw bit mask.
    constexpr utype value() const noexcept { return d_flags; }

    //! Return true if any flag is raised.
    constexpr explicit operator bool() const noexcept { return d_flags != 0; }

    //! Convert to the raw bit mask.
    constexpr explicit operator utype() const noexcept { return d_flags; }

    //! Raise a single flag.
    constexpr Flags &operator|=( FlagSet f ) noexcept
    {
        raise( f );
        return *this;
    }

    //! Return a copy with a single flag raised.
    constexpr Flags operator|( FlagSet f ) const noexcept
    {
        Flags ret = *this;
        ret |= f;
        return ret;
    }

    //! Check if two flag sets contain the same raw bit mask.
    constexpr bool operator==( const Flags &other ) const noexcept
    {
        return d_flags == other.d_flags;
    }

    //! Check if two flag sets contain different raw bit masks.
    constexpr bool operator!=( const Flags &other ) const noexcept
    {
        return d_flags != other.d_flags;
    }

private:
    utype d_flags = 0;

    template<class... Fs>
    static constexpr utype mask( Fs... fs ) noexcept
    {
        static_assert( ( std::is_same_v<FlagSet, std::decay_t<Fs>> && ... ),
                       "Flags functions require values of the enum type" );

        return static_cast<utype>( ( utype{ 0 } | ... | static_cast<utype>( fs ) ) );
    }
};


} // namespace AMP::Utilities
#endif
