f = open("src/elementwise_gen.inc", "w")

class Type:
  def __init__(self, name, short):
    self.name = name
    self.short = short

def genElementwise2(op: str, o: Type, i1: Type, i2: Type):
  return """
const char* libtcCuda%s2_%s_%s_%s(libtcCudaStream& stream, void* out, const void* in1, const void* in2, uint64_t n) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  auto err = cudaLaunchKernelEx(&config, %s2<%s, %s, %s>, (%s*)out, (%s*)in1, (%s*)in2, n);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

""" % (op.capitalize(), o.short, i1.short, i2.short, op, o.name, i1.name, i2.name, o.name, i1.name, i2.name)

def genElementwise1(op: str, o: Type, inp: Type):
  return """
const char* libtcCuda%s_%s_%s(libtcCudaStream& stream, void* out, const void* inp, uint64_t n) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  auto err = cudaLaunchKernelEx(&config, %s<%s, %s>, (%s*)out, (%s*)inp, n);
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

""" % (op.capitalize(), o.short, inp.short, op, o.name, inp.name, o.name, inp.name)

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

f.write('''
#include <string>
#include <cstdint>

#include <cuda_runtime.h>

#include <libgpuc_cuda.hpp>
        
template<typename O, typename I1, typename I2>
__global__ void add2(O* out, const I1* in1, const I2* in2, uint64_t n);
        
template<typename O, typename I1, typename I2>
__global__ void sub2(O* out, const I1* in1, const I2* in2, uint64_t n);
        
template<typename O, typename I1, typename I2>
__global__ void mul2(O* out, const I1* in1, const I2* in2, uint64_t n);
        
template<typename O, typename I1, typename I2>
__global__ void div2(O* out, const I1* in1, const I2* in2, uint64_t n);
        
template<typename O, typename I>
__global__ void cast(O* out, const I* in, uint64_t n);
        
extern const char* setupElementwiseKernel(libtcCudaStream& stream, uint64_t n, cudaLaunchConfig_t& config);

#ifdef __cplusplus
extern "C" {

extern const char* libtcCudaAdd2_f64_f64_f64(libtcCudaStream& stream, void* out, const void* in1, const void* in2, uint64_t n);       
extern const char* libtcCudaCast_f64_f32(libtcCudaStream& stream, void* out, const void* inp, uint64_t n);
#endif
''')

for op in ["add", "sub", "mul", "div"]:
  for o in types:
    for i1 in types:
      for i2 in types:
        if op == "add" and o.name == "double" and i1.name == "double" and i2.name == "double":
          continue
        f.write(genElementwise2(op=op, o=o, i1=i1, i2=i2))

for o in types:
  for i in types:
    if o.name == i.name or (o.name == "double" and i.name == "float"):
      continue
    f.write(genElementwise1(op="cast", o=o, inp=i))

f.write('''
#ifdef __cplusplus
}
#endif
''')