#ifndef included_AMP_NullWriter
#define included_AMP_NullWriter

#include "AMP/IO/Writer.h"


namespace AMP::IO {


/**
 * \class NullWriter
 * \brief A class used to abstract away reading/writing files for visualization
 * \details  This class provides default routines that do nothing.
 */
class NullWriter : public AMP::IO::Writer
{
public:
    //!  Default constructor
    NullWriter( const WriterParameters &params = WriterParameters() );

    //!  Default destructor
    virtual ~NullWriter() = default;

    // Inherited functions
    WriterProperties getProperties() const override;
    void readFile( const std::string & ) override{};
    void writeFile( const std::string &, size_t, double = 0 ) override {}
};

} // namespace AMP::IO

#endif
