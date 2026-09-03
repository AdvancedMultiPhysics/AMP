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
#include <Tpetra_Map_def.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_MultiVector_def.hpp>
#include <Tpetra_Vector_decl.hpp>
#include <Tpetra_Vector_def.hpp>


template class Tpetra::Vector<double, int32_t, int32_t, Tpetra::Vector<>::node_type>;
template class Tpetra::MultiVector<double, int32_t, int32_t, Tpetra::Vector<>::node_type>;
template class Tpetra::Map<int, int, Tpetra::Vector<>::node_type>;

template class Tpetra::Vector<double, int32_t, int64_t, Tpetra::Vector<>::node_type>;
template class Tpetra::MultiVector<double, int32_t, int64_t, Tpetra::Vector<>::node_type>;

template class Tpetra::Vector<float, int32_t, int64_t, Tpetra::Vector<>::node_type>;
template class Tpetra::MultiVector<float, int32_t, int64_t, Tpetra::Vector<>::node_type>;

template class Tpetra::Map<int, long, Tpetra::Vector<>::node_type>;
template class Tpetra::Details::FixedHashTable<long, int, Tpetra::Vector<>::node_type>;
