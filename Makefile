OS := $(shell uname | tr A-Z a-z)

ifeq ($(OS),linux)
only_copy:
	cd tensorcpu && make only_copy
	cd tensorcuda && make only_copy

copy_all_dart:
	cd tensorcpu && make copy_dart
	cd tensorcuda && make copy_dart

build:
	cd tensorcpu && make build
	cd tensorcuda && make build
endif

ifeq ($(OS),darwin)
copy_all_dart:
	cd tensorcpu && make copy_dart

build:
	cd tensorcpu && make build
endif

.PHONY: copy_all_dart build
