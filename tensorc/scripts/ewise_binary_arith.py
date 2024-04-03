from typing import TextIO
from common import types


f: TextIO = open("src/ewise/binary_arith_gen.inc", "w")

def gen(op: str):
  for o in types:
    for i1 in types:
      for i2 in types:
        f.write(f'''template const char* tc{op.capitalize()}({o.name}* out, const {i1.name}* inp1, const {i2.name}* inp2, const {i2.name}* scalar, uint64_t nel, uint8_t flip);\n\n''')

f.write("""
#include <stdint.h>
#include <stdlib.h>
        
template<typename O, typename I1, typename I2>
const char* tcPlus(O* out, const I1* inp1, const I2* inp2, const I2* scalar, uint64_t nel, uint8_t flip);
        
""")

ops = ["plus"]

for op in ops:
  gen(op)

f.close()