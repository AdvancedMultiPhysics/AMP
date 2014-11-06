# Create a shared_ptr.h file in the include directory that contains 
#    a shared_ptr class (hopefully typedef to a compiler basic)
# Arguements:
#    INSTALL_DIR - Directory to install shared_ptr.h
#    NAMESPACE - Namespace to contain the shared_ptr class (may be empty)
INCLUDE( CheckCXXSourceCompiles )
FUNCTION( CONFIGURE_SHARED_PTR INSTALL_DIR NAMESPACE )
    # Determine if we want to use the timer utility
    CHECK_CXX_SOURCE_COMPILES(
	    "   #include <memory>
            namespace ${NAMESPACE} { using using std::shared_ptr; }
	        int main() {
	            ${NAMESPACE}::shared_ptr<int> ptr;
	            return 0;
	        }
	    "
	    MEMORY_SHARED_PTR )
    CHECK_CXX_SOURCE_COMPILES(
	    "   #include <memory>
            namespace ${NAMESPACE} { using std::tr1::shared_ptr; }
	        int main() {
	            ${NAMESPACE}::shared_ptr<int> ptr;
	            return 0;
	        }
	    "
	    MEMORY_TR1_SHARED_PTR )
    CHECK_CXX_SOURCE_COMPILES(
	    "   #include <tr1/memory>
            namespace  ${NAMESPACE} { using std::tr1::shared_ptr; }
	        int main() {
	            ${NAMESPACE}::shared_ptr<int> ptr;
	            return 0;
	        }
	    "
	    TR1_MEMORY_TR1_SHARED_PTR )
    SET( CMAKE_REQUIRED_INCLUDES "${BOOST_INCLUDE}" )
    CHECK_CXX_SOURCE_COMPILES(
	    "   #include \"boost/shared_ptr.hpp\"
            namespace  ${NAMESPACE} { using boost::shared_ptr; }
	        int main() {
	            ${NAMESPACE}::shared_ptr<int> ptr;
	            return 0;
	        }
	    "
	    BOOST_SHARED_PTR )
	IF ( BOOST_SHARED_PTR )
        FILE(WRITE  "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "#include \"boost/shared_ptr.hpp\"\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "#include \"boost/weak_ptr.hpp\"\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "#include \"boost/enable_shared_from_this.hpp\"\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "namespace ${NAMESPACE} {\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using boost::shared_ptr; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using boost::dynamic_pointer_cast; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using boost::const_pointer_cast; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using boost::weak_ptr; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using boost::enable_shared_from_this; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "}\n")
	ELSEIF ( MEMORY_SHARED_PTR )
        FILE(WRITE  "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "#include <memory>\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "namespace ${NAMESPACE} {\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::shared_ptr; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::dynamic_pointer_cast; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::const_pointer_cast; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::weak_ptr; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::enable_shared_from_this; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "}\n")
	ELSEIF ( MEMORY_TR1_SHARED_PTR )
        FILE(WRITE  "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "#include <memory>\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "namespace ${NAMESPACE} {\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::shared_ptr; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::dynamic_pointer_cast; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::const_pointer_cast; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::weak_ptr; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::enable_shared_from_this; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "}\n")
	ELSEIF ( TR1_MEMORY_TR1_SHARED_PTR )
        FILE(WRITE  "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "#include <tr1/memory>\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "namespace ${NAMESPACE} {\n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::shared_ptr; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::dynamic_pointer_cast; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::const_pointer_cast; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::weak_ptr; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "    using std::tr1::enable_shared_from_this; \n")
        FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "}\n")
    ELSE() 
        MESSAGE(FATAL_ERROR "No valid shared_ptr found" )
    ENDIF()
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "${CMAKE_CURRENT_BINARY_DIR}/tmp/shared_ptr.h" "${INSTALL_DIR}/shared_ptr.h" )
ENDFUNCTION()

