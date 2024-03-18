OS := $(shell uname | tr A-Z a-z)

ifeq ($(OS),linux)
copy_all_dart:
	cd tensorc && make copy_dart
	cd tensorcuda && make copy_dart

build:
	cd tensorc && make build
	cd tensorcuda && make build
endif

ifeq ($(OS),darwin)
copy_all_dart:
	cd tensorc && make copy_dart

build:
	cd tensorc && make build
endif

.PHONY: copy_all_dart build
