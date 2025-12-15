#ifndef included_AMP_loadBalanceSimulator
#define included_AMP_loadBalanceSimulator

#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/Database.h"


namespace AMP::Mesh {


/**
 * \class loadBalanceSimulator
 * \brief Class used to contain simulated mesh load
 * \details  This structure provides info that can be used to simulate loading
 *   a mesh, and checking the resulting load balance
 */
class loadBalanceSimulator
{
public:
    /**
     * \brief    Default constructor
     * \details  This will simulate creating a new load balance
     * \param params  Input parameters for the mesh that will be used
     */
    loadBalanceSimulator( std::shared_ptr<const AMP::Database> params );

    /**
     * \brief    Default constructor
     * \details  This will simulate creating a new load balance
     * \param cost      The cost of the mesh (# of elements)
     * \param maxProc   The maximum number of processors (0: no limit)
     * \param name      The mesh name
     */
    loadBalanceSimulator( double cost, int maxProc = 0, const std::string &name = "" );

    /**
     * \brief    Default constructor
     * \details  This will simulate creating a new load balance
     * \param meshes    The list of meshes
     * \param method    The load balance method
     * \param name      The mesh name
     */
    loadBalanceSimulator( const std::vector<loadBalanceSimulator> &meshes,
                          int method              = 2,
                          const std::string &name = "" );

    //! Empty constructor
    loadBalanceSimulator();

    //! Function to get the current ranks in the load balance
    size_t getMethod() const { return d_method; }

    //! Return the number of assigned ranks
    inline int nRanks() const { return d_end - d_begin; }

    //! Function to get the ranks for the mesh
    std::vector<int> getRanks() const;

    //! Function to get the ranks for the ith submesh
    inline std::vector<int> getRanks( int i ) const { return d_submeshes[i].getRanks(); }

    //! Function to change the ranks
    inline int maxProcs() const { return d_max_procs; }

    //! Function to change the ranks
    void setProcs( int N_ranks );

    //! Return the submeshes
    inline const auto &getSubmeshes() const { return d_submeshes; }

    //! Return the number of base meshes
    int getMeshCount() const;

    //! Function to get the total cost
    double getCost() const { return d_cost; }

    //! Get the cost per rank ( # of elements * relative weight )
    std::vector<double> getRankCost() const;

    /**
     * \brief    Print the mesh hierarchy
     * \details  This function will print the load balance and mesh hierarchy
     * \param detail    The details on what to print (bit mask)
     *                  0: Auto determine the level of info to print
     *                  1: Print summary info only
     *                  3: Print summary + rank info
     *                  5: Print summary + mesh info
     *                  6: Print summary + rank + mesh info
     * \param indent    Number of spaces to indent the printing
     */
    void print( uint8_t detail = 0, uint8_t indent = 0 );


private:                                           // Internal data
    std::string d_name;                            // Name of mesh
    double d_cost;                                 // Total cost of the mesh
    double d_maxCostRank;                          // Current maximum cost of any rank
    int d_max_procs;                               // Maximum number of allowed ranks
    int d_method;                                  // Load balance method (see MultiMesh)
    bool d_allEqual;                               // Is the cost of all meshes equal
    int d_begin;                                   // Begin assigned rank
    int d_end;                                     // End assigned rank
    std::vector<loadBalanceSimulator> d_submeshes; // list of sub-meshes


private: // Private functions
    void setRanks( int begin, int end );
    inline void setRanks( int rank ) { setRanks( rank, rank + 1 ); }
    void addRankCost( std::vector<double> &cost ) const;
    void addRank();
    void addRanks( int N );
    void copyRanks( const loadBalanceSimulator &x, int offset );
    void loadBalance( int, std::vector<int> &N );
};


} // namespace AMP::Mesh

#endif
