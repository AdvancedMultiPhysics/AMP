#include <algorithm>

namespace AMP {
namespace Materials {

// ====================================================================================
// FUNCTIONS
// ====================================================================================

/**
 * \brief	   Evaluate a vector of results
 * \tparam	  VTYPE	   A vector type container.
 *
 *			  Template argument:
 *			  VTYPE   A vector type that must support:
 *					  VTYPE::iterator
 *					  VTYPE::const_iterator
 *					  VTYPE::iterator begin()
 *					  VTYPE::iterator end()
 *			  Constraints:
 *					  type of *VTYPE::iterator is a Number.
 *					  type of *VTYPE::const_iterator is a const Number.
 * \param[out]  r	  vector of results
 * \param[in]   args  input arguments corresponding to d_sequence
 *					  Must be in the correct order: T, u, burn
 */
template <class Number>
template <class INPUT_VTYPE, class RETURN_VTYPE>
void Property<Number>::evalvActual(RETURN_VTYPE& r, const std::map< std::string, boost::shared_ptr<INPUT_VTYPE> >& args)
{
  std::vector<Number> eval_args( d_n_arguments );		// list of arguments for each input type
  typename RETURN_VTYPE::iterator r_iter;				// Iterator for the results vector

  // First we make sure that all of the vectors have something in them
  AMP_ASSERT( r.begin() != r.end() );

  // Make a vector of iterators - one for each d_arguments
  std::vector<typename INPUT_VTYPE::iterator> parameter_iter;
  std::vector<size_t> parameter_indices;
  std::vector< typename std::map<std::string, boost::shared_ptr<INPUT_VTYPE> >::const_iterator > parameter_map_iter;

  // Walk through d_arguments and set the iterator at the beginning of the map vector to which it corresponds
  for ( size_t i=0; i<d_arguments.size(); ++i ) {
	  typename std::map<std::string, boost::shared_ptr<INPUT_VTYPE> >::const_iterator mapIter;
	  mapIter = args.find(d_arguments[i]);
	  if( mapIter==args.end() )
	  {
		  eval_args[i] = d_defaults[i];
	  } else {
		  parameter_iter.push_back(mapIter->second->begin());
		  parameter_indices.push_back(i);
		  parameter_map_iter.push_back(mapIter);
	  }
  }
  const size_t npresent = parameter_iter.size();

  for (r_iter = r.begin(); r_iter != r.end(); ++r_iter )
  {
	  // Loop through the list of actually present parameter iterators and assign their values to the vector being sent to eval
	  // Check that parameter iterators have not gone off the end - meaning result and input sizes do not match
	  if (d_n_arguments > 0) {
		  for ( size_t ipresent =0; ipresent<npresent; ipresent++) {
			  AMP_INSIST(parameter_iter[ipresent] != parameter_map_iter[ipresent]->second->end(),
				     std::string("size mismatch between results and arguments - too few argument values for argument ")+
				     d_arguments[parameter_indices[ipresent]]+std::string("\n") );
			  eval_args[parameter_indices[ipresent]] = *(parameter_iter[ipresent]);
		  }
	  }
	  *r_iter = eval(eval_args);

	  // update the parameter iterators
	  for (size_t i=0; i<npresent; i++) {
		  parameter_iter[i]++;
	  }
  }
  // Make sure the input value iterators all got to the end.
  if (d_n_arguments > 0) {
	  for ( size_t ipresent =0; ipresent<npresent; ipresent++) {
		  AMP_INSIST(parameter_iter[ipresent] == parameter_map_iter[ipresent]->second->end(),
				  "size mismatch between results and arguments - too few results\n");
	  }
  }
}

/**
 * \brief	   Evaluate a vector of vectors of results
 * \tparam	  VTYPE	   A vector type container.
 *
 *			  Template argument:
 *			  VTYPE   A vector type that must support:
 *					  VTYPE::iterator
 *					  VTYPE::const_iterator
 *					  VTYPE::iterator begin()
 *					  VTYPE::iterator end()
 *			  Constraints:
 *					  type of *VTYPE::iterator is a Number.
 *					  type of *VTYPE::const_iterator is a const Number.
 * \param[out]  r	  vector of vectors of results
 * \param[in]   args  input arguments corresponding to d_sequence
 *					  Must be in the correct order: T, u, burn
 */
template <class Number>
template <class INPUT_VTYPE, class RETURN_VTYPE>
void Property<Number>::evalvActualVector(std::vector<boost::shared_ptr<RETURN_VTYPE> >& r,
		const std::map< std::string, boost::shared_ptr<INPUT_VTYPE> >& args)
{
  size_t rdim0 = r.size();										// number of results vectors to return
  std::vector<Number> eval_args( d_n_arguments );				// list of arguments for each input type
  std::vector<typename RETURN_VTYPE::iterator> r_iter(rdim0);	// Iterator for the results vector

  // First we make sure that all of the vectors have something in them
  for (size_t i=0; i<rdim0; i++) {
	  AMP_ASSERT( r[i]->begin() != r[i]->end() );
  }

  // Make a vector of iterators - one for each d_arguments
  std::vector<typename INPUT_VTYPE::iterator> parameter_iter;
  std::vector<size_t> parameter_indices;
  std::vector< typename std::map<std::string, boost::shared_ptr<INPUT_VTYPE> >::const_iterator > parameter_map_iter;

  // Walk through d_arguments and set the iterator at the beginning of the map vector to which it corresponds
  for ( size_t i=0; i<d_arguments.size(); ++i ) {
	  typename std::map<std::string, boost::shared_ptr<INPUT_VTYPE> >::const_iterator mapIter;
	  mapIter = args.find(d_arguments[i]);
	  if( mapIter==args.end() )
	  {
		  eval_args[i] = d_defaults[i];
	  } else {
		  parameter_iter.push_back(mapIter->second->begin());
		  parameter_indices.push_back(i);
		  parameter_map_iter.push_back(mapIter);
	  }
  }
  const size_t npresent = parameter_iter.size();

  for (size_t i=0; i<rdim0; i++) {r_iter[i] = r[i]->begin();}
  bool goAgain = true;
  while (goAgain)
  {
	  // Loop through the list of actually present parameter iterators and assign their values to the vector being sent to eval
	  // Check that parameter iterators have not gone off the end - meaning result and input sizes do not match
	  if (d_n_arguments > 0) {
		  for ( size_t ipresent =0; ipresent<npresent; ipresent++) {
			  AMP_INSIST(parameter_iter[ipresent] != parameter_map_iter[ipresent]->second->end(),
				     std::string("size mismatch between results and arguments - too few argument values for argument ")+
				     d_arguments[parameter_indices[ipresent]]+std::string("\n") );
			  eval_args[parameter_indices[ipresent]] = *(parameter_iter[ipresent]);
		  }
	  }
	  std::vector<Number> evalResult = evalVector(eval_args);
	  for (size_t i=0; i<rdim0; i++) {*(r_iter[i]) = evalResult[i];}

	  // update the parameter iterators
	  for (size_t i=0; i<npresent; i++) {
		  parameter_iter[i]++;
	  }

	  // update result iterators;
	  for (size_t i=0; i<rdim0; i++) {++r_iter[i];}

	  // check if any of result iterators reached end
	  bool alldone = true;
	  for (size_t i=0; i<rdim0; i++) {
		  if (r_iter[i] == r[i]->end()) goAgain = false;
		  if (not goAgain) alldone = alldone and r_iter[i] == r[i]->end();
	  }
	  if (not goAgain) AMP_INSIST(alldone, "vector result vectors have unequal sizes");
  }
  // Make sure the input value iterators all got to the end.
  if (d_n_arguments > 0) {
	  for ( size_t ipresent =0; ipresent<npresent; ipresent++) {
		  AMP_INSIST(parameter_iter[ipresent] == parameter_map_iter[ipresent]->second->end(),
				  "size mismatch between results and arguments - too few results\n");
	  }
  }
}

/**
 * \brief	   Evaluate a tensor of vectors of results
 * \tparam	  VTYPE	   A vector type container.
 *
 *			  Template argument:
 *			  VTYPE   A vector type that must support:
 *					  VTYPE::iterator
 *					  VTYPE::const_iterator
 *					  VTYPE::iterator begin()
 *					  VTYPE::iterator end()
 *			  Constraints:
 *					  type of *VTYPE::iterator is a Number.
 *					  type of *VTYPE::const_iterator is a const Number.
 * \param[out]  r	  vector of vectors of results
 * \param[in]   args  input arguments corresponding to d_sequence
 *					  Must be in the correct order: T, u, burn
 */
template <class Number>
template <class INPUT_VTYPE, class RETURN_VTYPE>
void Property<Number>::evalvActualTensor(std::vector<std::vector<boost::shared_ptr<RETURN_VTYPE> > >& r,
		const std::map< std::string, boost::shared_ptr<INPUT_VTYPE> >& args)
{
  size_t rdim0 = r.size();										// dimension 0 of results tensor to return
  size_t rdim1 = r[0].size();									// dimension 1 of results tensor to return
  std::vector<Number> eval_args( d_n_arguments );				// list of arguments for each input type
  std::vector<std::vector<typename RETURN_VTYPE::iterator> >
  	  r_iter(rdim0, std::vector<typename RETURN_VTYPE::iterator>(rdim1));	// Iterator for the results tensor

  // Check rows of tensor are same size
  for (size_t i=1; i<rdim0; i++) {AMP_ASSERT( r[i].size() == rdim1);}

  // First we make sure that all of the vectors have something in them
  for (size_t i=0; i<rdim0; i++) for (size_t j=0; j<rdim1; j++) {
	  AMP_ASSERT( r[i][j]->begin() != r[i][j]->end() );
  }

  // Make a vector of iterators - one for each d_arguments
  std::vector<typename INPUT_VTYPE::iterator> parameter_iter;
  std::vector<size_t> parameter_indices;
  std::vector< typename std::map<std::string, boost::shared_ptr<INPUT_VTYPE> >::const_iterator > parameter_map_iter;

  // Walk through d_arguments and set the iterator at the beginning of the map vector to which it corresponds
  for ( size_t i=0; i<d_arguments.size(); ++i ) {
	  typename std::map<std::string, boost::shared_ptr<INPUT_VTYPE> >::const_iterator mapIter;
	  mapIter = args.find(d_arguments[i]);
	  if( mapIter==args.end() )
	  {
		  eval_args[i] = d_defaults[i];
	  } else {
		  parameter_iter.push_back(mapIter->second->begin());
		  parameter_indices.push_back(i);
		  parameter_map_iter.push_back(mapIter);
	  }
  }
  const size_t npresent = parameter_iter.size();

  for (size_t i=0; i<rdim0; i++) for (size_t j=0; j<rdim1; j++) {r_iter[i][j] = r[i][j]->begin();}
  bool goAgain = true;
  while (goAgain)
  {
	  // Loop through the list of actually present parameter iterators and assign their values to the vector being sent to eval
	  // Check that parameter iterators have not gone off the end - meaning result and input sizes do not match
	  if (d_n_arguments > 0) {
		  for ( size_t ipresent =0; ipresent<npresent; ipresent++) {
			  AMP_INSIST(parameter_iter[ipresent] != parameter_map_iter[ipresent]->second->end(),
				     std::string("size mismatch between results and arguments - too few argument values for argument ")+
				     d_arguments[parameter_indices[ipresent]]+std::string("\n") );
			  eval_args[parameter_indices[ipresent]] = *(parameter_iter[ipresent]);
		  }
	  }
	  std::vector<std::vector<Number> > evalResult = evalTensor(eval_args);
	  for (size_t i=0; i<rdim0; i++) for (size_t j=0; j<rdim1; j++) {*(r_iter[i][j]) = evalResult[i][j];}

	  // update the parameter iterators
	  for (size_t i=0; i<npresent; i++) {
		  parameter_iter[i]++;
	  }

	  // update result iterators;
	  for (size_t i=0; i<rdim0; i++)  for (size_t j=0; j<rdim1; j++) {++r_iter[i][j];}

	  // check if any of result iterators reached end
	  bool alldone = true;
	  for (size_t i=0; i<rdim0; i++) for (size_t j=0; j<rdim1; j++)  {
		  if (r_iter[i][j] == r[i][j]->end()) goAgain = false;
		  if (not goAgain) alldone = alldone and r_iter[i][j] == r[i][j]->end();
	  }
	  if (not goAgain) AMP_INSIST(alldone, "tensor result vectors have unequal sizes");
  }
  // Make sure the input value iterators all got to the end.
  if (d_n_arguments > 0) {
	  for ( size_t ipresent =0; ipresent<npresent; ipresent++) {
		  AMP_INSIST(parameter_iter[ipresent] == parameter_map_iter[ipresent]->second->end(),
				  "size mismatch between results and arguments - too few results\n");
	  }
  }
}

template<class Number>
Number Property<Number>::eval( std::vector<Number>& ) {
	AMP_INSIST(false, "function is not implemented for this property");
	Number x = 0.;
	return x;
}

template<class Number>
std::vector<Number> Property<Number>::evalVector( std::vector<Number>& )
{
	AMP_INSIST(false, "function is not implemented for this property");
	std::vector<Number> r(0);
	return r;
}

template<class Number>
std::vector<std::vector<Number> > Property<Number>::evalTensor( std::vector<Number>& )
{
	AMP_INSIST(false, "function is not implemented for this property");
	std::vector<std::vector<Number> > r(0);
	return r;
}

template<class Number>
void Property<Number>::evalv(std::vector<Number>& r,
const std::map< std::string, boost::shared_ptr<std::vector<Number> > >& args)
{
	AMP_ASSERT(in_range(args));
	evalvActual(r, args);
}

template<class Number>
void Property<Number>::evalv(std::vector< boost::shared_ptr< std::vector<Number> > >& r,
const std::map< std::string, boost::shared_ptr<std::vector<Number> > >& args)
{
	AMP_ASSERT(in_range(args));
	evalvActualVector(r, args);
}

template<class Number>
void Property<Number>::evalv(std::vector< std::vector< boost::shared_ptr<std::vector<Number> > > >& r,
const std::map< std::string, boost::shared_ptr<std::vector<Number> > >& args)
{
	AMP_ASSERT(in_range(args));
	evalvActualTensor(r, args);
}

/// determine if a set of values are all within range or not
template<class Number>
template <class INPUT_VTYPE>
bool Property<Number>::in_range(const std::string &argname, const INPUT_VTYPE &values)
{
        if(!is_argument(argname)) return true;
	std::vector<Number> range = get_arg_range(argname);
	bool result = true;
    for (typename INPUT_VTYPE::const_iterator i=values.begin(); i!=values.end(); ++i)
		result = result and *i >= range[0] and *i <= range[1];
	return result;
}

/// determine if a set of sets of values are all within range or not
template<class Number>
template<class INPUT_VTYPE>
bool Property<Number>::in_range(
		const std::map<std::string, boost::shared_ptr<INPUT_VTYPE> > &values) {
	bool result = true;
	for (typename std::map<std::string, boost::shared_ptr<INPUT_VTYPE> >::const_iterator
			j = values.begin(); j != values.end(); j++) {
		if (is_argument(j->first)) {
			std::vector<Number> range = get_arg_range(j->first);
			for (typename INPUT_VTYPE::const_iterator i = j->second->begin(); i != j->second->end(); ++i)
			{
				Number datum = *i;
				result = result and datum >= range[0] and datum <= range[1];
			}
		}
	}
	return result;
}

} // Materials namespace
} // AMP namespace

