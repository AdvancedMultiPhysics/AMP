
#ifndef EDGE_T
#define EDGE_T

#include <vector>

class edge_t
{
public:
    edge_t( double const *A, double const *B, double const *ABC );
    void set_support_points( double const *A, double const *B );
    void set_containing_plane( double const *ABC );
    double const *get_support_point_ptr( unsigned int i ) const;
    double const *get_normal();
    double const *get_direction();
    double const *get_center();
    bool above_point( double const *point, double tolerance = 1.0e-12 );
    int project_point( double const *point_in_containing_plane,
                       double *projection,
                       double tolerance = 1.0e-12 );

private:
    double const *support_points_ptr[2];
    double const *containing_plane_ptr;
    std::vector<double> normal;
    std::vector<double> direction;
    std::vector<double> center;
    std::vector<double> tmp;
    bool normal_updated;
    bool direction_updated;
    bool center_updated;

    void compute_normal();
    void compute_direction();
    void compute_center();
    double compute_distance_to_containing_line( double const *point_in_containing_plane );
};

#endif // EDGE_T
