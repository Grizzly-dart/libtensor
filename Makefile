OS := $(shell uname | tr a-z A-Z)

copy_all_dart:
	cd tensorc && make copy_dart
ifeq($(OS),linux)
	cd tensorcuda && make copy_dart
endif