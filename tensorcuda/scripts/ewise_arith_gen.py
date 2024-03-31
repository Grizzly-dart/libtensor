f = open("src/ewise/ewise_arith_gen.inc", "w")

class Type:
  def __init__(self, name, short):
    self.name = name
    self.short = short

types = [
  Type(name="double", short="f64"),
  Type(name="float", short="f32"),
  Type(name="int64_t", short="i64"),
  Type(name="int32_t", short="i32"),
  Type(name="int16_t", short="i16"),
  Type(name="int8_t", short="i8"),
  Type(name="uint64_t", short="u64"),
  Type(name="uint32_t", short="u32"),
  Type(name="uint16_t", short="u16"),
  Type(name="uint8_t", short="u8"),
]

ops = ["plus", "minus", "mul", "div", "pow"]

def genSigs(op: str):
  f.write("""
template<typename O, typename I1, typename I2>
__global__ void %s(O* out, I1* inp1, I2* inp2, I2 scalar, uint64_t n, uint8_t flipScalar);
""" % op)
  

def gen(op: str):
  str = """

const char* libtcCuda%s(libtcCudaStream& stream, void* out, void* inp1, void* inp2, void* scalar,
    uint64_t n, uint8_t flipScalar, dtype outType, dtype inp1Type, dtype inp2Type) {
  if((scalar == nullptr) == (inp2 == nullptr)) {
    return "Confusing input";
  }

  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  cudaError_t err;
""" % (op.capitalize())
  
  for o in types:
    if o.name != "double":
      str += " else if (outType == %s) {\n" % o.short
    else:
      str += "  if (outType == %s) {\n" % o.short
    for i1 in types:
      if i1.name != "double":
        str += " else if (inp1Type == %s) {\n" % i1.short
      else:
        str += "    if (inp1Type == %s) {\n" % i1.short
      for i2 in types:
        if i2.name != "double":
          str += " else if (inp2Type == %s) {\n" % i2.short
        else:
          str += "      if (inp2Type == %s) {\n" % i2.short
        str += "        %s s = scalar == nullptr ? 0 : *(%s *)scalar;" % (i2.name, i2.name)
        str += """
        err = cudaLaunchKernelEx(
          &config, %s<%s, %s, %s>, (%s *)out,
          (%s *)inp1, (%s *)inp2, s, n, flipScalar
        );
      }""" % (op, o.name, i1.name, i2.name, o.name, i1.name, i2.name)
      str += """ else {
        return "Unsupported input type";
      }
    }"""
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

def genScalar(op: str):
  str = """
const char* libtcCuda%sScalar(libtcCudaStream& stream, void* out, void* inp1, void* inp2, uint64_t n, dtype outType, dtype inp1Type, dtype inp2Type) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

""" % (op.capitalize())
  
  for o in types:
    if o.name != "double":
      str += " else if (outType == %s) {\n" % o.short
    else:
      str += "  if (outType == %s) {\n" % o.short
    for i1 in types:
      if i1.name != "double":
        str += " else if (inp1Type == %s) {\n" % i1.short
      else:
        str += "    if (inp1Type == %s) {\n" % i1.short
      for i2 in types:
        if i2.name != "double":
          str += " else if (inp2Type == %s) {" % i2.short
        else:
          str += "      if (inp2Type == %s) {" % i2.short
        str += """
        err = cudaLaunchKernelEx(
          &config, %sScalar<%s, %s, %s>, (%s *)out,
          (%s *)inp1, *(%s *)inp2, n
        );
      }""" % (op, o.name, i1.name, i2.name, o.name, i1.name, i2.name)
      str += """ else {
        return "Unsupported input type";
      }
    }"""
    str += """ else {
      return "Unsupported input type";
    }
  }"""
    
  str += """
  } else {
    return "Unsupported output type";
  }

  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}
"""
  return str

f.write(
"""#include <cuda_runtime.h>

#include <cstdint>
#include <libgpuc_cuda.hpp>
#include <reducers.hpp>
#include <string>
""")

#for op in ops:
#  f.write(genScalar(op = op))


for op in ops:
  genSigs(op = op)

for op in ops:
  if op.endswith("Lhs"): continue
  f.write(gen(op = op))

