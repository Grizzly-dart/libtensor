import typing
from common import types

f: typing.TextIO

def genCast():
  str = """
const char* tcuCast(tcuStream& stream, void* out, void* inp,
    uint64_t n, dtype outType, dtype inpType) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  if(inpType == outType) {
    if(out == inp) {
      return nullptr;
    }
    auto err = cudaMemcpyAsync(out, inp, n * sizeof(inpType % 16), cudaMemcpyDeviceToDevice, stream.stream);
    if (err != cudaSuccess) {
      return cudaGetErrorString(err);
    }
    return nullptr;
  }

  cudaError_t err;
"""
  
  for o in types:
    if o.name != "double":
      str += " else if (outType == %s) {\n" % o.short
    else:
      str += "  if (outType == %s) {\n" % o.short
    first = True
    for inp in types:
      if inp.name == o.name: continue
      if not first:
        str += " else if (inpType == %s) {" % inp.short
      else:
        str += "    if (inpType == %s) {" % inp.short
        first = False
      str += """
      err = cudaLaunchKernelEx(
        &config, cast<%s, %s>, (%s *)out,
        (%s *)inp, n
      );
    }""" % (o.name, inp.name, o.name, inp.name)
    str += """ else {
      return "Unsupported input type";
    }
  }"""
    
  str += """ else {
    return "Unsupported output type";
  }

  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}
"""
  return str

f = open("src/ewise/ewise_unary_gen.inc", "w")

f.write(
"""#include <cuda_runtime.h>

#include <cstdint>
#include <tensorcuda.hpp>
#include <reducers.hpp>
#include <string>
""")

f.write("""
template <typename O, typename I>
__global__ void cast(O *out, I *inp, uint64_t n);
        """)

f.write(genCast())
f.close()