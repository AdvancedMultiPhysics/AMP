#include "AMP/IO/AsciiWriter.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <vector>


namespace AMP::IO {


/************************************************************
 * Constructor/Destructor                                    *
 ************************************************************/
AsciiWriter::AsciiWriter() : AMP::IO::Writer() {}
AsciiWriter::~AsciiWriter() = default;


/************************************************************
 * Some basic functions                                      *
 ************************************************************/
Writer::WriterProperties AsciiWriter::getProperties() const
{
    WriterProperties properties;
    properties.type           = "Ascii";
    properties.extension      = "ascii";
    properties.registerVector = true;
    properties.registerMatrix = true;
    properties.enabled        = true;
    return properties;
}


/************************************************************
 * Function to read a silo file                              *
 ************************************************************/
void AsciiWriter::readFile( const std::string & )
{
    AMP_ERROR( "readFile is not implemented yet" );
}


/************************************************************
 * Function to write an ascii file                           *
 ************************************************************/
template<class ID, class TYPE>
std::set<ID> getKeys( const std::map<ID, TYPE> &local_map, const AMP_MPI &comm )
{
    std::set<ID> ids;
    for ( auto it = local_map.begin(); it != local_map.end(); ++it )
        ids.insert( it->first );
    comm.setGather( ids );
    return ids;
}
void AsciiWriter::writeFile( const std::string &fname_in, size_t iteration_count, double )
{
    PROFILE( "writeFile" );
    // Open the file for writing
    FILE *fid = nullptr;
    if ( d_comm.getRank() == 0 ) {
        auto pwd = IO::path( fname_in );
        IO::recursiveMkdir( pwd );
        auto fname = fname_in + "_" + std::to_string( iteration_count ) + "." + getExtension();
        fid        = fopen( fname.c_str(), "w" );
        AMP_ASSERT( fid );
    }
    // Get the ids for the vectors and save the data
    auto vec_ids = getKeys( d_vectors, d_comm );
    for ( const auto &vec_id : vec_ids ) {
        // Send the data to rank 0
        d_comm.barrier();
        std::shared_ptr<AMP::LinearAlgebra::Vector> src_vec;
        if ( d_vectors.find( vec_id ) != d_vectors.end() )
            src_vec = d_vectors[vec_id].vec;
        auto dst_vec = sendVecToRoot( src_vec, d_comm );
        // Write the data
        if ( d_comm.getRank() == 0 ) {
            fprintf( fid,
                     "Vector: \"%s\" %i\n",
                     dst_vec->getName().c_str(),
                     static_cast<int>( dst_vec->getGlobalSize() ) );
            for ( size_t i = 0; i < dst_vec->getGlobalSize(); i++ )
                fprintf( fid, "   %0.14e\n", dst_vec->getValueByGlobalID( i ) );
            fprintf( fid, "\n\n" );
        }
    }
    // Get the ids for the matricies and save the data
    auto mat_ids = getKeys( d_matrices, d_comm );
    for ( const auto &mat_id : mat_ids ) {
        // Send the header data to rank 0
        d_comm.barrier();
        std::string name;
        size_t size[2] = { 0, 0 };
        std::shared_ptr<AMP::LinearAlgebra::Matrix> mat;
        if ( d_matrices.find( mat_id ) != d_matrices.end() ) {
            mat     = d_matrices[mat_id].mat;
            name    = d_matrices[mat_id].name;
            size[0] = mat->createOutputVector()->getGlobalSize();
            size[1] = mat->createInputVector()->getGlobalSize();
        }
        name    = d_comm.bcast( name, 0 );
        size[0] = d_comm.bcast( size[0], 0 );
        size[1] = d_comm.bcast( size[1], 0 );
        // Write the data
        if ( d_comm.getRank() == 0 ) {
            fprintf( fid,
                     "Matrix: \"%s\" %i %i\n",
                     name.c_str(),
                     static_cast<int>( size[0] ),
                     static_cast<int>( size[1] ) );
        }
        std::vector<size_t> col;
        std::vector<double> data;
        for ( int row = 0; row < static_cast<int>( size[0] ); row++ ) {
            // Get and print the current row
            sendRowToRoot( mat, d_comm, row, col, data );
            if ( d_comm.getRank() == 0 ) {
                for ( size_t i = 0; i < col.size(); i++ )
                    fprintf( fid, "   %4i %4u  %0.14e\n", row, (unsigned int) col[i], data[i] );
            }
        }
        if ( d_comm.getRank() == 0 ) {
            fprintf( fid, "\n\n" );
        }
    }
    // Close the file
    if ( fid )
        fclose( fid );
}


/************************************************************
 * Function to copy a vector to rank 0                       *
 ************************************************************/
std::shared_ptr<const AMP::LinearAlgebra::Vector>
AsciiWriter::sendVecToRoot( std::shared_ptr<const AMP::LinearAlgebra::Vector> src_vec,
                            const AMP_MPI &comm )
{

    int rank      = comm.getRank();
    int ownerRank = src_vec ? comm.getRank() : 1e9;
    ownerRank     = comm.minReduce( ownerRank );
    // Broadcast the local vector size and name to all processors for simplicity
    std::string name;
    if ( rank == ownerRank )
        name = src_vec->getName();
    name              = comm.bcast( name, ownerRank );
    size_t local_size = 0;
    if ( src_vec )
        local_size = src_vec->getLocalSize();
    std::vector<size_t> size( comm.getSize(), 0 );
    comm.allGather( local_size, &size[0] );
    size_t global_size = 0;
    for ( int i = 0; i < comm.getSize(); i++ )
        global_size += size[i];
    // If we are not rank 0 and do not have a copy of the vector we are done
    if ( !src_vec && rank != 0 )
        return nullptr;
    // Send the local data to rank 0
    std::vector<AMP_MPI::Request> requests;
    std::vector<double> local_data( local_size, 0 );
    if ( local_size > 0 ) {
        for ( size_t i = 0; i < local_size; i++ )
            local_data[i] = src_vec->getValueByLocalID( i );
        requests.push_back( comm.Isend( &local_data[0], local_size, 0, 123 ) );
    }
    // Rank 0 needs to create the vector and recv all data
    std::shared_ptr<AMP::LinearAlgebra::Vector> dst_vec;
    if ( rank == 0 ) {
        auto var = std::make_shared<AMP::LinearAlgebra::Variable>( name );
        dst_vec  = AMP::LinearAlgebra::createSimpleVector<double>(
            global_size, var, AMP_MPI( AMP_COMM_SELF ) );
        AMP_ASSERT( dst_vec->numberOfDataBlocks() == 1 );
        auto *ptr = dst_vec->getRawDataBlock<double>( 0 );
        size_t i  = 0;
        for ( int j = 0; j < comm.getSize(); j++ ) {
            if ( size[j] > 0 ) {
                requests.push_back( comm.Irecv( &ptr[i], size[j], j, 123 ) );
                i += size[j];
            }
        }
    }
    if ( !requests.empty() )
        comm.waitAll( requests.size(), &requests[0] );
    return dst_vec;
}


/************************************************************
 * Function to copy a row to rank 0                          *
 ************************************************************/
void AsciiWriter::sendRowToRoot( std::shared_ptr<const AMP::LinearAlgebra::Matrix> mat,
                                 const AMP_MPI &comm,
                                 int row,
                                 std::vector<size_t> &cols,
                                 std::vector<double> &data )
{
    int rank = comm.getRank();
    cols.clear();
    data.clear();
    // Determine who "owns" the row
    int own_rank = 0;
    if ( mat ) {
        auto DOF = mat->getLeftDOFManager();
        if ( row >= (int) DOF->beginDOF() && row < (int) DOF->endDOF() )
            own_rank = rank;
    }
    own_rank = comm.maxReduce( own_rank );
    // Send the data
    std::vector<AMP_MPI::Request> requests;
    if ( own_rank == rank ) {
        mat->getRowByGlobalID( row, cols, data );
        if ( rank == 0 )
            return;
        size_t size = cols.size();
        requests.push_back( comm.Isend<size_t>( &size, 1, 0, 124 ) );
        if ( size > 0 ) {
            requests.push_back( comm.Isend<size_t>( &cols[0], size, 0, 125 ) );
            requests.push_back( comm.Isend<double>( &data[0], size, 0, 126 ) );
        }
    }
    // Recv the data
    if ( rank == 0 ) {
        size_t size = 0;
        int length  = 1;
        comm.recv<size_t>( &size, length, own_rank, false, 124 );
        cols.resize( size, 0 );
        data.resize( size, 0 );
        if ( size > 0 ) {
            requests.push_back( comm.Irecv<size_t>( &cols[0], size, own_rank, 125 ) );
            requests.push_back( comm.Irecv<double>( &data[0], size, own_rank, 126 ) );
        }
    }
    if ( !requests.empty() )
        comm.waitAll( requests.size(), &requests[0] );
}


} // namespace AMP::IO
