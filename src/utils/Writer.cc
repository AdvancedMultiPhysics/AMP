#include "AMP/utils/Writer.h"
#include "AMP/utils/Utilities.h"

#include "AMP/utils/AsciiWriter.h"
#include "AMP/utils/NullWriter.h"
#ifdef USE_AMP_MESH
#include "AMP/ampmesh/SiloIO.h"
#endif


namespace AMP {
namespace Utilities {


/************************************************************
 * Builder                                                   *
 ************************************************************/
std::shared_ptr<AMP::Utilities::Writer> Writer::buildWriter( const std::string &type )
{
    std::shared_ptr<AMP::Utilities::Writer> writer;
    if ( type == "None" || type == "none" || type == "NONE" ) {
        writer.reset( new AMP::Utilities::NullWriter() );
    } else if ( type == "Silo" || type == "silo" || type == "SILO" ) {
#if defined( USE_AMP_MESH ) && defined( USE_EXT_SILO )
        writer.reset( new AMP::Mesh::SiloIO() );
#else
        writer.reset( new AMP::Utilities::NullWriter() );
#endif
    } else if ( type == "Ascii" || type == "ascii" || type == "ASCII" ) {
        writer.reset( new AMP::Utilities::AsciiWriter() );
    } else {
        AMP_ERROR( "Unknown writer" );
    }
    return writer;
}
std::shared_ptr<AMP::Utilities::Writer> Writer::buildWriter( std::shared_ptr<AMP::Database> db )
{
    std::string type                               = db->getString( "Name" );
    std::shared_ptr<AMP::Utilities::Writer> writer = Writer::buildWriter( type );
    if ( db->keyExists( "Decomposition" ) )
        writer->setDecomposition( db->getScalar<int>( "Decomposition" ) );
    return writer;
}


/************************************************************
 * Constructor/Destructor                                    *
 ************************************************************/
Writer::Writer() : d_comm( AMP_COMM_WORLD ) { d_decomposition = 2; }
Writer::~Writer() = default;


/************************************************************
 * Some basic functions                                      *
 ************************************************************/
void Writer::setDecomposition( int d )
{
    AMP_INSIST( d == 1 || d == 2, "decomposition must be 1 or 2" );
    d_decomposition = d;
}
void Writer::createDirectories( const std::string &filename )
{
    size_t i = filename.rfind( '/' );
    if ( i != std::string::npos && d_comm.getRank() == 0 )
        AMP::Utilities::recursiveMkdir(
            filename.substr( 0, i ), ( S_IRUSR | S_IWUSR | S_IXUSR ), false );
    d_comm.barrier();
}


} // namespace Utilities
} // namespace AMP
