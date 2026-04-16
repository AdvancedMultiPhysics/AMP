#ifndef included_AMP_Polynomial
#define included_AMP_Polynomial

#include <complex>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <tuple>
#include <vector>


namespace AMP {


/*! \class Polynomial
    \brief A polynomial class

    This class provides polynomial capabilities.  It allows for the creation and operations
    on polynomials of the form:  f(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
    All polynomials and polynomial operations assume real numbers.
*/
class Polynomial final
{
public:
    //! Empty constructor
    Polynomial();

    //! Construct a constant polynomial
    Polynomial( double a0 );

    //! Construct a polynomial of order n (a has n+1 coefficients)
    Polynomial( int n, const double *a );

    //! Construct a polynomial of order n (a has n+1 coefficients)
    Polynomial( std::vector<double> a );

    //! Function to construct a polynomial given it's roots
    static Polynomial createFromRoots( int n, const double *roots );

    //! Function to construct a polynomial given it's roots
    static Polynomial createFromRoots( const std::vector<double> &roots );

    //! Return the order of the polynomial
    inline int order() const { return d_p.size() - 1; }

    //! Set the ith coefficient in the polynomial
    void setCoeff( int i, double a );

    //! Get the ith coefficient in the polynomial
    inline double getCoeff( int i ) const { return d_p[i]; }

    //! Get the ith coefficient in the polynomial
    inline const double &operator[]( const int i ) const { return d_p[i]; }

    //! Print the polynomial to the screen
    void print() const;

    //! Convert the polynomial to a string
    std::string getPolynomial() const;

    //! Check if two polynomials are equal
    inline bool operator==( const Polynomial &rhs ) const { return d_p == rhs.d_p; }

    //! Check if two polynomials are not equal
    inline bool operator!=( const Polynomial &rhs ) const { return d_p != rhs.d_p; }

    //! Add two polynomials
    friend Polynomial operator+( const Polynomial &a, const Polynomial &b );

    //! Subtract two polynomials
    friend Polynomial operator-( const Polynomial &a, const Polynomial &b );

    //! Multiply two polynomials
    friend Polynomial operator*( const Polynomial &a, const Polynomial &b );

    //! Divide two polynomials (this returns the quotient and remainder as a std::tuple)
    friend std::tuple<Polynomial, Polynomial> operator/( const Polynomial &a, const Polynomial &b );

    //! Compute the derivative of the polynomial
    Polynomial derivative() const;

    //! Evaluate the polynomial at x
    double eval( double x ) const noexcept;

    //! Evaluate the polynomial at complex point x
    std::complex<double> eval( const std::complex<double> &x ) const noexcept;

    //! Evaluate the derivative of the polynomial at x
    double evalDerivative( double x ) const;

    //! Evaluate the derivative of the polynomial at complex point x
    std::complex<double> evalDerivative( const std::complex<double> &x ) const;

    /*!
     * \brief    Function to return a real root in the given interval
     * \details  This function will return a real root in the given
     *    interval.  If multiple roots exist, then any one root may be returned.
     *    Note: This function uses a variation of the bisection method, and will
     *    throw an error if there is not an odd number of root in the intervals.
     *    This can be checked by checking the sign of the polynomial at the boundaries
     *    and ensuring that they have different signs.
     */
    double rootInterval( double lb, double ub, double tol = 1e-8 ) const;

    /*!
     * \brief    Function to return the roots
     * \details  This function will return all the roots of the polynomial
     *   Note: since some roots may be complex, all roots will be returned
     *   as complex numbers.
     *   We return the roots sorted accoring to their real value and
     *   then their imaginary value, or unsorted.
     */
    std::vector<std::complex<double>> roots() const;


    /*!
     * \brief    Fit a polynomial to the function
     * \details  This function will fit a polynomial the form
     *   \f$ p_0 + p_1 x + ... + p_n ^n \f$ to the function y = f(x) .
     *   The fitting will be based on a least squares minimization.
     * @param[in] n             Polynomial order
     * @param[in] x             Points to fit
     * @param[in] y             f(x)
     */
    static Polynomial fit( int n, const std::vector<double> &x, const std::vector<double> &y );


    /*!
     * \brief    Fit a polynomial to the function
     * \details  This function will fit a polynomial the form
     *   \f$ p_0 + p_1 x + ... + p_n ^n \f$ to the function f(x) in [lb,ub].
     *   The fitting will be based on a least squares minimization.
     * @param[in] n             Polynomial order
     * @param[in] fun           Function to fit: y = f(x)
     * @param[in] lb            Lower bound of x
     * @param[in] ub            Upper bound of x
     * @param[in] N             Number of points to fit
     */
    static Polynomial
    fit( int n, std::function<double( double )> fun, double lb, double ub, int N );


    /*!
     * \brief    Estimate the error for a polynomial fit
     * \details  This function will estimate the error in fitting a polynomial
     *     to a function f(x) in [lb,ub]
     * @param[in] fun           Function to fit: y = f(x)
     * @param[in] lb            Lower bound of x
     * @param[in] ub            Upper bound of x
     */
    double error( std::function<double( double )> fun, double lb, double ub ) const;

private:
    std::vector<double> d_p;
};


} // namespace AMP

#endif
