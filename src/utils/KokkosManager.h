#ifndef included_AMP_KokkosManager
#define included_AMP_KokkosManager

#include "AMP/utils/AMPManager.h"

#include <string>
#include <vector>

namespace AMP::Utilities {


void initializeKokkos( int &argc, char *argv[], const AMPManagerProperties & );
void finalizeKokkos();
bool KokkosEnabled();
bool isKokkosInitialized();
bool KokkosInitializedOpenMP();

} // namespace AMP::Utilities

#endif
