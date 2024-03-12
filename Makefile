OS := $(shell uname | tr A-Z a-z)

ifeq ($(OS),linux)
copy_all_dart:
	cd tensorc && make copy_dart
	cd tensorcuda && make copy_dart
endif

ifeq ($(OS),darwin)
copy_all_dart:
	cd tensorc && make copy_dart
endif

