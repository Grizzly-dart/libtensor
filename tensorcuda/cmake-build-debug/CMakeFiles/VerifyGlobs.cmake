# CMAKE generated file: DO NOT EDIT!
# Generated by CMake Version 3.27
cmake_policy(SET CMP0009 NEW)

# MYSRC at CMakeLists.txt:12 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/*.c")
set(OLD_GLOB
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/cmake-build-debug/CMakeFiles/cmake.verify_globs")
endif()

# MYSRC at CMakeLists.txt:12 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/*.cpp")
set(OLD_GLOB
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/cmake-build-debug/CMakeFiles/cmake.verify_globs")
endif()

# MYSRC at CMakeLists.txt:12 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/*.cu")
set(OLD_GLOB
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/cuda_interface.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/ewise/activation/elu.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/ewise/activation/sigmoid.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/ewise/activation/softplus.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/ewise/ewise_arith.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/ewise/ewise_arith_v2.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/ewise/ewise_trignometry.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/ewise/ewise_unary.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/matrix/addmatmul.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/matrix/addmatmulT.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/matrix/matmul.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/matrix/matmulT.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/matrix/pick_rows.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/matrix/tranpose.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/nn2d/conv2d.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/nn2d/maxpool2d.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/reducers.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/rwise/rwise_arith.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/stat/mean.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/stat/sum.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/stat/variance.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/stat2d/mean2d.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/stat2d/moment2d.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/stat2d/normalize2d.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/stat2d/sum2d.cu"
  "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/src/stat2d/variance2d.cu"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/tejag/projects/dart/tensorc/libtensor/tensorcuda/cmake-build-debug/CMakeFiles/cmake.verify_globs")
endif()
