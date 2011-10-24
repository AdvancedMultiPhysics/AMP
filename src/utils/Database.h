//
// File:	$URL: file:///usr/casc/samrai/repository/AMP/tags/v-2-4-4/source/toolbox/database/Database.h $
// Package:	AMP toolbox
// Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
// Revision:	$LastChangedRevision: 2620 $
// Modified:	$LastChangedDate: 2008-11-19 14:24:28 -0800 (Wed, 19 Nov 2008) $
// Description:	An abstract base class for the AMP database objects
//

#ifndef included_Database
#define included_Database

#include <iostream>
#include <vector>
#include "DatabaseBox.h"
#include <complex>
#include "PIO.h"
#include <string>
#include "boost/shared_ptr.hpp"

#include "Castable.h"

namespace AMP {
   

/**
 * @brief Class Database is an abstract base class for the input, restart,
 * and visualization databases.  
 *
 * AMP databases store (key,value) pairs in a hierarchical
 * database.  Each value may be another database or a boolean, box,
 * character, double complex, double, float, integer, or string.
 * DatabaseBoxes are stored using the toolbox box structure.
 *
 * Data is entered into the database through methods of the general form
 * putTYPE(key, TYPE) or putTYPEArray(key, TYPE array), where TYPE is the
 * type of value created.  If the specified key already exists in the
 * database, then the existing key is silently deleted.
 *
 * Data is extracted from the database through methods of the general form
 * TYPE = getTYPE(key), where TYPE is the type of value to be returned
 * from the database.  There are two general lookup methods.  In the first,
 * a default value is provided (for scalars only).  If the specified key is
 * not found in the database, then the specified default is returned.  In
 * the second form, no default is provided, and the database exists with
 * an error message and program exits if the key is not found.  The array
 * version of getTYPE() works in a similar fashion.
 */

class Database : public Castable
{
public:
   
      typedef boost::shared_ptr<Database> shared_ptr;

   /**
    * Enumerated type indicating what type of values is stored in
    * a database entry.  Returned from getType() method.
    * 
    * Note: The AMP_ prefix is needed since some poorly written
    *       packages do "#define CHAR" etc.
    */
   enum DataType { AMP_INVALID, 
		   AMP_DATABASE,
		   AMP_BOOL,
		   AMP_CHAR,
		   AMP_INT,
		   AMP_COMPLEX,
		   AMP_DOUBLE,
		   AMP_FLOAT,
		   AMP_STRING,
		   AMP_BOX};

   /**
    * The constructor for the database base class does nothing interesting.
    */
   Database();

   /**
    * The virtual destructor for the database base class does nothing
    * interesting.
    */
   virtual ~Database();

   /**
    * Create a new database file.
    *
    * Returns true if successful.
    *
    * @param name name of database. Normally a filename.
    */
   virtual bool create(const std::string& name) = 0;


   /**
    * Open an existing database file.
    *
    * Returns true if successful.
    *
    * @param name name of database. Normally a filename.
    */
   virtual bool open(const std::string& name) = 0;


   /**
    * Close the database.
    *
    * Returns true if successful.
    *
    * If the database is currently open then close it.  This should
    * flush all data to the file (if the database is on disk).
    */
   virtual bool close() = 0;

   /**
    * Return true if the specified key exists in the database and false
    * otherwise.
    *
    * @param key Key name to lookup.
    */
   virtual bool keyExists(const std::string& key) = 0;

   /**
    * Return all keys in the database.
    */
   virtual std::vector<std::string> getAllKeys() = 0;


   /**
    * @brief Return the type of data associated with the key.
    *
    * If the key does not exist, then INVALID is returned
    *
    * @param key Key name in database.
    */
   virtual enum DataType getArrayType(const std::string& key) = 0;

   /**
    * @brief Return the size of the array associated with the key.  
    *
    * If the key does not exist, then zero is returned.  If the key is
    * a database then zero is returned.
    *
    * @param key Key name in database.
    */
   virtual int getArraySize(const std::string& key) = 0;

   /**
    * Return whether the specified key represents a database entry.  If
    * the key does not exist, then false is returned.
    *
    * @param key Key name in database.
    */
   virtual bool isDatabase(const std::string& key) = 0;

   /**
    * Create a new database with the specified key name.  If the key already
    * exists in the database, then the old key record is deleted and the new
    * one is silently created in its place.
    *
    * @param key Key name in database.
    */
   virtual boost::shared_ptr<Database> putDatabase(const std::string& key) = 0;

   /**
    * Get the database with the specified key name.  If the specified
    * key does not exist in the database or it is not a database, then
    * an error message is printed and the program exits.
    *
    * @param key Key name in database.
    */
   virtual boost::shared_ptr<Database> getDatabase(const std::string& key) = 0;

   /**
    * Return whether the specified key represents a boolean entry.  If
    * the key does not exist, then false is returned.
    *
    * @param key Key name in database.
    */
   virtual bool isBool(const std::string& key) = 0;

   /**
    * Create a boolean scalar entry in the database with the specified
    * key name.  If thoe key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key Key name in database.
    * @param data Value to put into database.
    */
   virtual void putBool(const std::string& key, const bool& data);

   /**
    * Create a boolean array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key  Key name in database.
    * @param data Array with data to put into database.
    */
   virtual void putBoolArray(const std::string& key,
       const std::vector<unsigned char>& data);

   /**
    * Create a boolean array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void putBoolArray(const std::string& key,
       const unsigned char* const data, const int nelements) = 0;

   /**
    * Get a boolean entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not a
    * boolean scalar, then an error message is printed and the program
    * exits.
    *
    * @param key Key name in database.
    */
   virtual bool getBool(const std::string& key);

   /**
    * Get a boolean entry in the database with the specified key name.
    * If the specified key does not exist in the database, then the default
    * value is returned.  If the key exists but is not a boolean scalar,
    * then an error message is printed and the program exits.
    */
   virtual bool getBoolWithDefault(const std::string& key,
       const bool& defaultvalue);

   /**
    * Get a boolean entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a boolean array, then an error message is printed and
    * the program exits.
    *
    * @param key Key name in database.
    */
   virtual std::vector<unsigned char> getBoolArray(const std::string& key) = 0;

   /**
    * Get a boolean entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a boolean array, then an error message is printed and
    * the program exits.  The specified number of elements must match
    * exactly the number of elements in the array in the database.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void getBoolArray(const std::string& key,
       bool* data, const int nelements);

   /**
    * Return whether the specified key represents a box entry.  If
    * the key does not exist, then false is returned.
    *
    * @param key Key name in database.
    */
   virtual bool isDatabaseBox(const std::string& key) = 0;

   /**
    * Create a box scalar entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key  Key name in database.
    * @param data Data to put into database.
    */
   virtual void putDatabaseBox(const std::string& key, const DatabaseBox& data);

   /**
    * Create a box array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key  Key name in database.
    * @param data Array with data to put into database.
    */
   virtual void putDatabaseBoxArray(const std::string& key,
       const std::vector<DatabaseBox>& data);

   /**
    * Create a box array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void putDatabaseBoxArray(const std::string& key,
       const DatabaseBox* const data, const int nelements) = 0;

   /**
    * Get a box entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not a
    * box scalar, then an error message is printed and the program
    * exits.
    *
    * @param key Key name in database.
    */
   virtual DatabaseBox getDatabaseBox(const std::string& key);

   /**
    * Get a box entry in the database with the specified key name.
    * If the specified key does not exist in the database, then the default
    * value is returned.  If the key exists but is not a box scalar,
    * then an error message is printed and the program exits.
    *
    * @param key          Key name in database.
    * @param defaultvalue Default value to return if not found.
    */
   virtual DatabaseBox getDatabaseBoxWithDefault(const std::string& key,
       const DatabaseBox& defaultvalue);

   /**
    * Get a box entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a box array, then an error message is printed and
    * the program exits.
    *
    * @param key Key name in database.
    */
   virtual std::vector<DatabaseBox> getDatabaseBoxArray(const std::string& key) = 0;

   /**
    * Get a box entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a box array, then an error message is printed and
    * the program exits.  The specified number of elements must match
    * exactly the number of elements in the array in the database.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void getDatabaseBoxArray(const std::string& key,
       DatabaseBox* data, const int nelements);

   /**
    * Return whether the specified key represents a character entry.  If
    * the key does not exist, then false is returned.
    *
    * @param key Key name in database.
    */
   virtual bool isChar(const std::string& key) = 0;

   /**
    * Create a character scalar entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key Key name in database.
    * @param data Value to put into database.
    */
   virtual void putChar(const std::string& key, const char& data);

   /**
    * Create a character array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key Key name in database.
    *
    * @param key  Key name in database.
    * @param data Array with data to put into database.
    */
   virtual void putCharArray(const std::string& key,
       const std::vector<char>& data);

   /**
    * Create a character array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void putCharArray(const std::string& key,
       const char* const data, const int nelements) = 0;

   /**
    * Get a character entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not an
    * character scalar, then an error message is printed and the program
    * exits.
    *
    * @param key Key name in database.
    */
   virtual char getChar(const std::string& key);

   /**
    * Get a character entry in the database with the specified key name.
    * If the specified key does not exist in the database, then the default
    * value is returned.  If the key exists but is not a character scalar,
    * then an error message is printed and the program exits.
    *
    * @param key          Key name in database.
    * @param defaultvalue Default value to return if not found.
    */
   virtual char getCharWithDefault(const std::string& key,
       const char& defaultvalue);

   /**
    * Get a character entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a character array, then an error message is printed and
    * the program exits.
    *
    * @param key Key name in database.
    */
   virtual std::vector<char> getCharArray(const std::string& key) = 0;

   /**
    * Get a character entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a character array, then an error message is printed and
    * the program exits.  The specified number of elements must match
    * exactly the number of elements in the array in the database.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void getCharArray(const std::string& key,
       char* data, const int nelements);

   /**
    * Return whether the specified key represents a complex entry.  If
    * the key does not exist, then false is returned.
    *
    * @param key Key name in database.
    */
   virtual bool isComplex(const std::string& key) = 0;

   /**
    * Create a complex scalar entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key Key name in database.
    * @param data Value to put into database.
    */
   virtual void putComplex(const std::string& key, const std::complex<double>& data);

   /**
    * Create a complex array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key  Key name in database.
    * @param data Array with data to put into database.
    */
   virtual void putComplexArray(
				const std::string& key, const std::vector<std::complex<double> >& data);

   /**
    * Create a complex array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void putComplexArray(
      const std::string& key, const std::complex<double>* const data, const int nelements) = 0;

   /**
    * Get a complex entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not a
    * complex scalar, then an error message is printed and the program
    * exits.
    *
    * @param key Key name in database.
    */
   virtual std::complex<double> getComplex(const std::string& key);

   /**
    * Get a complex entry in the database with the specified key name.
    * If the specified key does not exist in the database, then the default
    * value is returned.  If the key exists but is not a complex scalar,
    * then an error message is printed and the program exits.
    *
    * @param key          Key name in database.
    * @param defaultvalue Default value to return if not found.
    */
   virtual std::complex<double> getComplexWithDefault(
      const std::string& key, const std::complex<double>& defaultvalue);

   /**
    * Get a complex entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a complex array, then an error message is printed and
    * the program exits.
    *
    * @param key Key name in database.
    */
   virtual std::vector<std::complex<double> > getComplexArray(const std::string& key) = 0;

   /**
    * Get a complex entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a complex array, then an error message is printed and
    * the program exits.  The specified number of elements must match
    * exactly the number of elements in the array in the database.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void getComplexArray(
      const std::string& key, std::complex<double>* data, const int nelements);

   /**
    * Return whether the specified key represents a double entry.  If
    * the key does not exist, then false is returned.
    *
    * @param key Key name in database.
    */
   virtual bool isDouble(const std::string& key) = 0;

   /**
    * Create a double scalar entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key Key name in database.
    * @param data Value to put into database.
    */
   virtual void putDouble(const std::string& key, const double& data);

   /**
    * Create a double array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key  Key name in database.
    * @param data Array with data to put into database.
    */
   virtual void putDoubleArray(
			       const std::string& key, const std::vector<double>& data);

   /**
    * Create a double array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void putDoubleArray(
      const std::string& key, const double* const data, const int nelements) = 0;

   /**
    * Get a double entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not a
    * double scalar, then an error message is printed and the program
    * exits.
    *
    * @param key Key name in database.
    */
   virtual double getDouble(const std::string& key);

   /**
    * Get a double entry in the database with the specified key name.
    * If the specified key does not exist in the database, then the default
    * value is returned.  If the key exists but is not a double scalar,
    * then an error message is printed and the program exits.
    *
    * @param key          Key name in database.
    * @param defaultvalue Default value to return if not found.
    */
   virtual double getDoubleWithDefault(
      const std::string& key, const double& defaultvalue);

   /**
    * Get a double entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a double array, then an error message is printed and
    * the program exits.
    *
    * @param key Key name in database.
    */
   virtual std::vector<double> getDoubleArray(const std::string& key) = 0;

   /**
    * Get a double entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a double array, then an error message is printed and
    * the program exits.  The specified number of elements must match
    * exactly the number of elements in the array in the database.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void getDoubleArray(
      const std::string& key, double* data, const int nelements);

   /**
    * Return whether the specified key represents a float entry.  If
    * the key does not exist, then false is returned.
    *
    * @param key Key name in database.
    */
   virtual bool isFloat(const std::string& key) = 0;

   /**
    * Create a float scalar entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key Key name in database.
    * @param data Value to put into database.
    */
   virtual void putFloat(const std::string& key, const float& data);

   /**
    * Create a float array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key  Key name in database.
    * @param data Array with data to put into database.
    */
   virtual void putFloatArray(const std::string& key, 
                              const std::vector<float>& data);

   /**
    * Create a float array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void putFloatArray(
      const std::string& key, const float* const data, const int nelements) = 0;

   /**
    * Get a float entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not a
    * float scalar, then an error message is printed and the program
    * exits.
    *
    * @param key Key name in database.
    */
   virtual float getFloat(const std::string& key);

   /**
    * Get a float entry in the database with the specified key name.
    * If the specified key does not exist in the database, then the default
    * value is returned.  If the key exists but is not a float scalar,
    * then an error message is printed and the program exits.
    *
    * @param key          Key name in database.
    * @param defaultvalue Default value to return if not found.
    */
   virtual float getFloatWithDefault(
      const std::string& key, const float& defaultvalue);

   /**
    * Get a float entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a float array, then an error message is printed and
    * the program exits.
    *
    * @param key Key name in database.
    */
   virtual std::vector<float> getFloatArray(const std::string& key) = 0;

   /**
    * Get a float entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a float array, then an error message is printed and
    * the program exits.  The specified number of elements must match
    * exactly the number of elements in the array in the database.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void getFloatArray(
      const std::string& key, float* data, const int nelements);

   /**
    * Return whether the specified key represents an integer entry.  If
    * the key does not exist, then false is returned.
    *
    * @param key Key name in database.
    */
   virtual bool isInteger(const std::string& key) = 0;

   /**
    * Create an integer scalar entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key Key name in database.
    * @param data Value to put into database.
    */
   virtual void putInteger(const std::string& key, const int& data);

   /**
    * Create an integer array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key  Key name in database.
    * @param data Array with data to put into database.
    */
   virtual void putIntegerArray(const std::string& key, 
                                const std::vector<int>& data);

   /**
    * Create an integer array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void putIntegerArray(
      const std::string& key, const int* const data, const int nelements) = 0;

   /**
    * Get an integer entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not an
    * integer scalar, then an error message is printed and the program
    * exits.
    *
    * @param key Key name in database.
    */
   virtual int getInteger(const std::string& key);

   /**
    * Get an integer entry in the database with the specified key name.
    * If the specified key does not exist in the database, then the default
    * value is returned.  If the key exists but is not an integer scalar,
    * then an error message is printed and the program exits.
    *
    * @param key          Key name in database.
    * @param defaultvalue Default value to return if not found.
    */
   virtual int getIntegerWithDefault(
      const std::string& key, const int& defaultvalue);

   /**
    * Get an integer entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not an integer array, then an error message is printed and
    * the program exits.
    *
    * @param key Key name in database.
    */
   virtual std::vector<int> getIntegerArray(const std::string& key) = 0;

   /**
    * Get an integer entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not an integer array, then an error message is printed and
    * the program exits.  The specified number of elements must match
    * exactly the number of elements in the array in the database.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void getIntegerArray(
      const std::string& key, int* data, const int nelements);

   /**
    * Return whether the specified key represents a std::string entry.  If
    * the key does not exist, then false is returned.
    *
    * @param key Key name in database.
    */
   virtual bool isString(const std::string& key) = 0;

   /**
    * Create a string scalar entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key Key name in database.
    * @param data Value to put into database.
    */
   virtual void putString(const std::string& key, const std::string& data);

   /**
    * Create a string array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key  Key name in database.
    * @param data Array with data to put into database.
    */
   virtual void putStringArray(const std::string& key, 
                               const std::vector<std::string>& data);

   /**
    * Create a string array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void putStringArray(
      const std::string& key, const std::string* const data, const int nelements) = 0;

   /**
    * Get a string entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not an
    * string scalar, then an error message is printed and the program
    * exits.
    *
    * @param key Key name in database.
    */
   virtual std::string getString(const std::string& key);

   /**
    * Get a string entry in the database with the specified key name.
    * If the specified key does not exist in the database, then the default
    * value is returned.  If the key exists but is not a string scalar,
    * then an error message is printed and the program exits.
    *
    * @param key          Key name in database.
    * @param defaultvalue Default value to return if not found.
    */
   virtual std::string getStringWithDefault(
      const std::string& key, const std::string& defaultvalue);

   /**
    * Get a string entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a string array, then an error message is printed and
    * the program exits.
    *
    * @param key Key name in database.
    */
   virtual std::vector<std::string> getStringArray(const std::string& key) = 0;

   /**
    * Get a string entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a string array, then an error message is printed and
    * the program exits.  The specified number of elements must match
    * exactly the number of elements in the array in the database.
    *
    * @param key       Key name in database.
    * @param data      Array with data to put into database.
    * @param nelements Number of elements to write from array.
    */
   virtual void getStringArray(
      const std::string& key, std::string* data, const int nelements);

   /**
    * Get a bool entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not an
    * bool scalar, then an error message is printed and the program
    * exits.
    *
    * @param key    Key name in database.
    * @param scalar Returns scalar that was read.
    */
   void getScalar(const std::string& key, bool& scalar);

   /**
    * Get a bool entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a bool array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param scalar Value to put into database.
    */
   void putScalar(const std::string& key, const bool scalar);


   /**
    * Get a bool entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a bool array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param array  Returns array that was read.
    */
   void getArray(const std::string& key, std::vector<unsigned char>& array);

   /**
    * Create an bool array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key    Key name in database.
    * @param array  Array to put into database.
    */
   void putArray(const std::string& key, const std::vector<unsigned char> array);

   /**
    * Get a char entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not an
    * char scalar, then an error message is printed and the program
    * exits.
    *
    * @param key    Key name in database.
    * @param scalar Returns scalar that was read.
    */
   void getScalar(const std::string& key, char& scalar);

   /**
    * Get a char entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a char array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param scalar Value to put into database.
    */
   void putScalar(const std::string& key, const char scalar);


   /**
    * Get a char entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a char array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param array  Returns array that was read.
    */
   void getArray(const std::string& key, std::vector<char>& array);

   /**
    * Create an char array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key    Key name in database.
    * @param array  Array to put into database.
    */
   void putArray(const std::string& key, const std::vector<char> array);


   /**
    * Get a complex entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not an
    * complex scalar, then an error message is printed and the program
    * exits.
    *
    * @param key    Key name in database.
    * @param scalar Returns scalar that was read.
    */
   void getScalar(const std::string& key, std::complex<double>& scalar);

   /**
    * Get a complex entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a complex array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param scalar Value to put into database.
    */
   void putScalar(const std::string& key, const std::complex<double> scalar);


   /**
    * Get a complex entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a complex array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param array  Returns array that was read.
    */
   void getArray(const std::string& key, std::vector<std::complex<double> >& array);

   /**
    * Create an complex array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key    Key name in database.
    * @param array  Array to put into database.
    */
   void putArray(const std::string& key, const std::vector<std::complex<double> > array);


   /**
    * Get a float entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not an
    * float scalar, then an error message is printed and the program
    * exits.
    *
    * @param key    Key name in database.
    * @param scalar Returns scalar that was read.
    */
   void getScalar(const std::string& key, float& scalar);

   /**
    * Get a float entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a float array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param scalar Value to put into database.
    */
   void putScalar(const std::string& key, const float scalar);


   /**
    * Get a float entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a float array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param array  Returns array that was read.
    */
   void getArray(const std::string& key, std::vector<float>& array);

   /**
    * Create an float array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key    Key name in database.
    * @param array  Array to put into database.
    */
   void putArray(const std::string& key, const std::vector<float> array);


   /**
    * Get a double entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not an
    * double scalar, then an error message is printed and the program
    * exits.
    *
    * @param key    Key name in database.
    * @param scalar Returns scalar that was read.
    */
   void getScalar(const std::string& key, double& scalar);

   /**
    * Get a double entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a double array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param scalar Value to put into database.
    */
   void putScalar(const std::string& key, const double scalar);


   /**
    * Get a double entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a double array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param array  Returns array that was read.
    */
   void getArray(const std::string& key, std::vector<double>& array);

   /**
    * Create an double array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key    Key name in database.
    * @param array  Array to put into database.
    */
   void putArray(const std::string& key, const std::vector<double> array);

   /**
    * Get a integer entry in the database with the specified key name.
    * If the specified key does not exist in the database or is not an
    * integer scalar, then an error message is printed and the program
    * exits.
    *
    * @param key    Key name in database.
    * @param scalar Returns scalar that was read.
    */
   void getScalar(const std::string& key, int& scalar);

   /**
    * Get a integer entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a integer array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param scalar Value to put into database.
    */
   void putScalar(const std::string& key, const int scalar);


   /**
    * Get a integer entry from the database with the specified key
    * name.  If the specified key does not exist in the database or
    * is not a integer array, then an error message is printed and
    * the program exits.
    *
    * @param key    Key name in database.
    * @param array  Returns array that was read.
    */
   void getArray(const std::string& key, std::vector<int>& array);

   /**
    * Create an integer array entry in the database with the specified
    * key name.  If the key already exists in the database, then the old
    * key record is deleted and the new one is silently created in its place.
    *
    * @param key    Key name in database.
    * @param array  Array to put into database.
    */
   void putArray(const std::string& key, const std::vector<int> array);


   /**
    * @brief Returns the name of this database.  
    *
    * The name for the root of the database is the name supplied when creating it.
    * Names for nested databases are the keyname of the database.
    * 
    */
   virtual std::string getName() = 0;

   /**
    * Print the current database to the specified output stream.  If
    * no output stream is specified, then data is written to stream pout.
    * 
    * @param os Output stream.
    */
   virtual void printClassData(std::ostream& os = pout) = 0;

};

}

#ifndef DEBUG_NO_INLINE
#include "Database.I"
#endif
#endif



