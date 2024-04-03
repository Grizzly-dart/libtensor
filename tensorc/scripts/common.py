class Type:
  def __init__(self, name, short, index):
    self.name = name
    self.short = short
    self.index = index
    self.isFloat = name in ["double", "float"]

types = [
  Type(name="int8_t", short="i8", index=0),
  Type(name="int16_t", short="i16", index=1),
  Type(name="int32_t", short="i32", index=2),
  Type(name="int64_t", short="i64", index=3),
  Type(name="uint8_t", short="u8", index=4),
  Type(name="uint16_t", short="u16", index=5),
  Type(name="uint32_t", short="u32", index=6),
  Type(name="uint64_t", short="u64", index=7),
  Type(name="float", short="f32", index=8),
  Type(name="double", short="f64", index=9),
]