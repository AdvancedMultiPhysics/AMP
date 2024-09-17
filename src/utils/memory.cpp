#include "AMP/utils/memory.h"


namespace AMP::Utilities {


/****************************************************************************
 *  Get pointer location                                                     *
 ****************************************************************************/
MemoryType getMemoryType( const void *ptr )
{
    NULL_USE( ptr );
    auto type = MemoryType::unregistered;
#if defined( AMP_USE_CUDA ) || defined( USE_CUDA )
    type = getCudaMemoryType( ptr );
    if ( type != MemoryType::unregistered )
        return type;
#endif
#if defined( AMP_USE_HIP ) || defined( USE_HIP )
    type = getHipMemoryType( ptr );
    if ( type != MemoryType::unregistered )
        return type;
#endif
    return type;
}
std::string getString( MemoryType type )
{
    if ( type == MemoryType::unregistered )
        return "unregistered";
    else if ( type == MemoryType::host )
        return "host";
    else if ( type == MemoryType::device )
        return "device";
    else if ( type == MemoryType::managed )
        return "managed";
    else
        AMP_ERROR( "Unknown pointer type" );
}


} // namespace AMP::Utilities
