#include <Tpetra_Details_FixedHashTable.hpp>
#include <Tpetra_Details_FixedHashTable_decl.hpp>
#include <Tpetra_Details_FixedHashTable_def.hpp>
#include <Tpetra_Details_Transfer_decl.hpp>
#include <Tpetra_Details_Transfer_def.hpp>
#include <Tpetra_DirectoryImpl_decl.hpp>
#include <Tpetra_DirectoryImpl_def.hpp>
#include <Tpetra_Directory_def.hpp>
#include <Tpetra_DistObject_decl.hpp>
#include <Tpetra_DistObject_def.hpp>
#include <Tpetra_ImportExportData_decl.hpp>
#include <Tpetra_ImportExportData_def.hpp>
#include <Tpetra_Map_def.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_MultiVector_def.hpp>
#include <Tpetra_Vector_decl.hpp>
#include <Tpetra_Vector_def.hpp>

#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"


#define TPETRA_INST_ALL( ST, LO, GO )                          \
    template class Tpetra::Vector<ST, LO, GO, Tpetra_NT>;      \
    template class Tpetra::MultiVector<ST, LO, GO, Tpetra_NT>; \
    template class Tpetra::DistObject<ST, LO, GO, Tpetra_NT>;

#define TPETRA_INST_ORD( LO, GO )                                      \
    TPETRA_INST_ALL( double, LO, GO )                                  \
    TPETRA_INST_ALL( float, LO, GO )                                   \
    template class Tpetra::Map<LO, GO, Tpetra_NT>;                     \
    template class Tpetra::Details::Transfer<LO, GO, Tpetra_NT>;       \
    template class Tpetra::Directory<LO, GO, Tpetra_NT>;               \
    template class Tpetra::Details::FixedHashTable<LO, GO, Tpetra_NT>; \
    template class Tpetra::ImportExportData<LO, GO, Tpetra_NT>;

#define TPETRA_INST_LO( LO )       \
    TPETRA_INST_ORD( LO, int32_t ) \
    TPETRA_INST_ORD( LO, int64_t )

TPETRA_INST_LO( int32_t )
