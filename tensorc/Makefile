OS := $(shell uname)
ifeq ($(OS),Linux)
SOURCES = ${shell find src/ -type f -regextype egrep -regex ".*\.(cpp|cu|c)$$"}
endif
ifeq ($(OS),Darwin)
SOURCES = ${shell find -E src -type f -regex ".*\.(cpp|cu|c)$$"}
endif
ifeq ($(OS),CYGWIN)

endif

build/libtensorc.so: $(SOURCES)
	g++ -c -std=c++20 -fPIC -I/usr/local/include -o build/tensorc.o $(SOURCES)
	g++ -shared -o build/libtensorc.so build/tensorc.o

build: build/libtensorc.so

.PHONY: build build/libtensorc.so