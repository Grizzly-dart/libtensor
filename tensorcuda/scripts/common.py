class Type:
  def __init__(self, name, short):
    self.name = name
    self.short = short
    self.isFloat = name in ["double", "float"]

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