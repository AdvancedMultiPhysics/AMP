# Makefile
# Format all CMake files and (optionally) add spaces just inside parentheses.
# NOTE: The sed step is aggressive and may change things you don’t want.
#       Consider limiting the regex or specific directories/files if needed.

.PHONY: format

format:
	@# Guard: ensure cmake-format exists before doing any work
	@command -v cmake-format >/dev/null 2>&1 || { \
	  echo "Error: 'cmake-format' not found in PATH."; \
	  exit 1; \
	}
	@echo "Running cmake-format on all CMake files"
	@find . -type f \( -name '*.cmake' -o -name 'CMakeLists.txt' \) -exec cmake-format -i {} +
	@echo "Adding spaces inside parentheses with sed"
	@if [ "$$(uname)" = "Darwin" ]; then \
	  find . -type f \( -name '*.cmake' -o -name 'CMakeLists.txt' \) -exec sed -E -i '' \
	    -e 's/\(([^[:space:]])/(\ \1/g' \
	    -e 's/([^[:space:]])\)/\1\ )/g' {} + ; \
	else \
	  find . -type f \( -name '*.cmake' -o -name 'CMakeLists.txt' \) -exec sed -E -i \
	    -e 's/\(([^[:space:]])/(\ \1/g' \
	    -e 's/([^[:space:]])\)/\1\ )/g' \
	    -e 's/\( \)/\(\)/g' \
	    -e 's/\$\( MAKE \)/\$\(MAKE\)/g' \
	    -e 's/MACRO \(/MACRO\(/g' \
	    -e 's/FUNCTION \(/FUNCTION\(/g' \
	    -e 's/FOREACH \(/FOREACH\(/g' \
	    -e 's/ENDIF \(/ENDIF\(/g' \
	    {} + ; \
	fi
