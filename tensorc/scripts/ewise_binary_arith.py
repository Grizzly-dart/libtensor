from typing import TextIO
from common import types


f: TextIO = open("src/ewise/binary_arith_gen.inc", "w")

def gen(op: str):
  for o in types:
    for i1 in types:
      for i2 in types:
        if i2.index > i1.index:
          break
        f.write(f'''
const char* tc{op.capitalize()}_{o.short}_{i1.short}_{i2.short}({o.name}* out, const {i1.name}* inp1, const {i2.name}* inp2, const {i2.name}* scalar, uint64_t nel, uint8_t flip) {{
  return tc{op.capitalize()}(out, inp1, inp2, scalar, nel, flip);
}}''')

f.write("""
#include <stdint.h>
#include <stdlib.h>
        
template<typename O, typename I1, typename I2>
const char* tcPlus(O* out, const I1* inp1, const I2* inp2, const I2* scalar, uint64_t nel, uint8_t flip);
        
#ifdef __cplusplus
extern "C" {
#endif
        
""")

ops = ["plus", "minus", "mul", "div", "pow"]

for op in ops:
  gen(op)

f.write("""

#ifdef __cplusplus
}
#endif""")


f.close()