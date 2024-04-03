OS := $(shell uname | tr A-Z a-z)
ifeq ($(OS),linux)
	CPP := g++
	SOURCES = ${shell find src/ -type f -regextype egrep -regex ".*\.(cpp|cu|c)$$"}
endif
ifeq ($(OS),darwin)
	CPPMAJVER = $(shell brew list --versions | fgrep gcc | cut -d ' ' -f 2 | cut -d '.' -f1)
	CPP = g++-${CPPMAJVER}
	CPPFLAGS += -I/opt/homebrew/include/c++/${CPPMAJVER} 
		ifeq ($(shell uname -p),arm)
			CPPFLAGS += -I/opt/homebrew/include/c++/${CPPMAJVER}/aarch64-apple-darwin23
		endif
	SOURCES = ${shell find -E src -type f -regex ".*\.(cpp|cu|c)$$"}
endif
ifeq ($(OS),cygwin)

endif

build/libtensorc.so: $(SOURCES)
	python3 scripts/ewise_binary_arith.py
	${CPP} ${CPPFLAGS} -std=c++20 --shared -fPIC -I ./include -L ./build -o build/libtensorc.so $(SOURCES)

build: build/libtensorc.so

copy_dart: build/libtensorc.so
	cp build/libtensorc.so ../../gpuc_dart/lib/asset/$(OS)/

.PHONY: build build/libtensorc.so copy_dart