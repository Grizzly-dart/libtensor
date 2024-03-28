f = open("src/ewise/ewise_scalar_binary_arith_gen.inc", "w")

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

ops = ["add", "sub", "subLhs", "mul", "div", "divLhs"]

def gen(op: str, o: Type, i1: Type, i2: Type):
  str = """
const char* libtcCuda%s2_%s_%s_%s(libtcCudaStream& stream, void* out, const void* in1, const void* in2, uint64_t n) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

"""

  for o in types:
    for i1 in types:
      for i2 in types:
  return """
const char* libtcCuda%s2_%s_%s_%s(libtcCudaStream& stream, void* out, const void* in1, const void* in2, uint64_t n) {
  cudaLaunchConfig_t config{};
  auto serr = setupElementwiseKernel(stream, n, config);
  if (serr != nullptr) {
    return serr;
  }

  auto err = cudaLaunchKernelEx(&config, %s2<%s, %s, %s>, (%s*)out, (%s*)in1, (%s*)in2, n);


""" % (op.capitalize(), o.short, i1.short, i2.short, op, o.name, i1.name, i2.name, o.name, i1.name, i2.name)

  str = """
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}
"""



