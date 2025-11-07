#include "AMP/IO/NullWriter.h"


namespace AMP::IO {


/************************************************************
 * Some basic functions                                      *
 ************************************************************/
Writer::WriterProperties NullWriter::getProperties() const
{
    WriterProperties properties;
    properties.type                   = "Null";
    properties.extension              = "";
    properties.registerMesh           = true;
    properties.registerVector         = true;
    properties.registerVectorWithMesh = true;
    properties.registerMatrix         = true;
    properties.enabled                = true;
    properties.isNull                 = true;
    return properties;
}


} // namespace AMP::IO
