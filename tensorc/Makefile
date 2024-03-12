OS := $(shell uname | tr a-z A-Z)
ifeq ($(OS),linux)
SOURCES = ${shell find src/ -type f -regextype egrep -regex ".*\.(cpp|cu|c)$$"}
endif
ifeq ($(OS),darwin)
SOURCES = ${shell find -E src -type f -regex ".*\.(cpp|cu|c)$$"}
endif
ifeq ($(OS),cygwin)

endif

build/libtensorc.so: $(SOURCES)
	g++ -c -std=c++20 -fPIC -I/usr/local/include -o build/tensorc.o $(SOURCES)
	g++ -shared -o build/libtensorc.so build/tensorc.o

build: build/libtensorc.so

copy_dart: build/libtensorc.so
	cp build/libtensorc.so ../../gpuc_dart/lib/asset/$(OS)/

.PHONY: build build/libtensorc.so copy_dart