# Makefile (for formatting CMake files)

.PHONY: format

# Find all cmake files we want to format
CMAKE_FORMAT := $(shell find . -type f \( -name '*.cmake' -o -name 'CMakeLists.txt' \) )
REFORMAT = $(CMAKE_FORMAT:=.format)

ifndef VERBOSE
.SILENT:
endif

# Format target
cmake_format:
	@# Guard: ensure cmake-format exists before doing any work
	@command -v cmake-format >/dev/null 2>&1 || { \
	  echo "Error: 'cmake-format' not found in PATH."; \
	  exit 1; \
	}
%.format: % cmake_format
	cmake-format -i --enable-markup --literal-comment-pattern='^#' $<
	sed -E -i \
	    -e '/^[^#]/s/\(([^[:space:]])/(\ \1/g' \
	    -e '/^[^#]/s/([^[:space:]])\)/\1\ )/g' \
	    -e '/^[^#]/s/\( \)/\(\)/g' \
	    -e '/^[^#]/s/\$\( MAKE \)/\$\(MAKE\)/g' \
	    -e '/^[^#]/s/MACRO \(/MACRO\(/g' \
	    -e '/^[^#]/s/FUNCTION \(/FUNCTION\(/g' \
	    -e '/^[^#]/s/FOREACH \(/FOREACH\(/g' \
	    -e '/^[^#]/s/ELSE \(/ELSE\(/g' \
	    -e '/^[^#]/s/ENDIF \(/ENDIF\(/g' \
	    -e '/^[^#]/s/\^\( /\^\(/g' \
	    -e '/^[^#]/s/ \)\$$\"/\)\$$\"/g' \
        -e '/^[^#]/s/\( \\r\?\\n \)/\(\\r\?\\n\)/g' \
	    $<

format: $(REFORMAT)
	@echo "Formatting CMake files"; 


