#pragma once

// tile_api_llvm/tile.h
//
// 这是 tile.h 的“函数式”实验版本：
//   * Tile/Acc/Vec8/Vec32 不再暴露 data、stride、valid_zone、dtype 等字段；
//   * 所有状态访问都通过 get_*/set_* 函数；
//   * 每个函数都只调用预留的 __builtin_yai_* 编译器内建函数；
//   * builtin 已在本目录的 LLVM 21.1.7 Clang 副本中注册；当前先降为 __yai_*
//     外部 LLVM call，尚未进入 YAI MachineInstr 和专用寄存器分配。
//
// 每个对象只持有一个“值引用”：Tile/Acc/Vec 在 LLVM IR 里是 SSA 值，对象
// 本身不保存任何数据，也不保存物理寄存器编号。后端把 Tile 值映射到
// XTT0..XTT15，由寄存器分配器决定具体编号；数据只存在于硬件 Tile 寄存器里。

#include <array>
#include <cstddef>
#include <cstdint>

// DType 编码与 Tile-NPU-Sim / yai-isa 一致：
//   i8 = 0, u8 = 1, i32 = 2, u32 = 3, f32 = 4
// 这些值会原样写进 cfg.setx 的 dtype 字段，不能随意改。
enum class TileDType : std::uint8_t {
  I8 = 0,
  U8 = 1,
  I32 = 2,
  U32 = 3,
  F32 = 4,
};

enum class AccDType : std::uint8_t {
  I8 = 0,
  U8 = 1,
  I32 = 2,
  U32 = 3,
  F32 = 4,
};

enum class Vec8DType : std::uint8_t {
  I8 = 0,
  U8 = 1,
  I32 = 2,
  U32 = 3,
  F32 = 4,
};

enum class Vec32DType : std::uint8_t {
  I8 = 0,
  U8 = 1,
  I32 = 2,
  U32 = 3,
  F32 = 4,
};

static constexpr int TILE = 32;

// Tile/Acc/Vec 在 LLVM IR 里的值载体（vector_size 以字节计）：
//   Tile  = 32x32 i8  = 1024 bytes = <1024 x i8>
//   Acc   = 32x32 i32 = 4096 bytes = <1024 x i32>
//   Vec32 = 32 x i32  = 128 bytes  = <32 x i32>
//   Vec8  = 32 x i8   = 32 bytes   = <32 x i8>
//
// 它们只是“寄存器引用”的编译器载体：对象本身不保存数据，硬件数据在 Tile
// 寄存器里。后端保证这些值只在 XTT 寄存器类中存在，不会被 spill 到内存。
using TileValue = char __attribute__((vector_size(1024)));
using AccValue = int __attribute__((vector_size(4096)));
using Vec32Value = int __attribute__((vector_size(128)));
using Vec8Value = char __attribute__((vector_size(32)));

struct Tile;
struct Acc;
struct Vec8;
struct Vec32;

#if defined(__clang__) || defined(__GNUC__)
#define YAI_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define YAI_ALWAYS_INLINE inline
#endif

// 这些名字是未来编译器需要注册的 builtin 名称。
#define YAI_COMPILER_INTRINSIC(name) __builtin_yai_##name

// 这些声明允许前置声明类型在 inline 方法中把值引用传给其它 builtin；
// 定义放在四个类型都完整定义之后。
YAI_ALWAYS_INLINE TileValue yai_handle(const Tile &);
YAI_ALWAYS_INLINE AccValue yai_handle(const Acc &);
YAI_ALWAYS_INLINE Vec8Value yai_handle(const Vec8 &);
YAI_ALWAYS_INLINE Vec32Value yai_handle(const Vec32 &);

struct Tile {
  // Tile 值的 SSA 载体；物理寄存器由后端分配，程序员不可见。
  TileValue value_;
  YAI_ALWAYS_INLINE Tile()
      : value_(YAI_COMPILER_INTRINSIC(tile_create)()) {}
  YAI_ALWAYS_INLINE explicit Tile(TileValue value) : value_(value) {}

  // ------------------------------------------------------------------------
  // 描述/配置状态访问
  // ------------------------------------------------------------------------
  YAI_ALWAYS_INLINE std::byte *get_data() const {
    return reinterpret_cast<std::byte *>(YAI_COMPILER_INTRINSIC(tile_get_data)(value_));
  }
  YAI_ALWAYS_INLINE void set_data(std::byte *value) {
    YAI_COMPILER_INTRINSIC(tile_set_data)(value_, value);
  }

  YAI_ALWAYS_INLINE std::int32_t get_stride(std::size_t dim) const {
    return YAI_COMPILER_INTRINSIC(tile_get_stride)(value_, dim);
  }
  YAI_ALWAYS_INLINE std::array<std::int32_t, 2> get_stride() const {
    return {get_stride(0), get_stride(1)};
  }
  YAI_ALWAYS_INLINE void set_stride(std::size_t dim, std::int32_t value) {
    YAI_COMPILER_INTRINSIC(tile_set_stride)(value_, dim, value);
  }
  YAI_ALWAYS_INLINE void set_stride(
      const std::array<std::int32_t, 2> &value) {
    set_strides(value[0], value[1]);
  }
  YAI_ALWAYS_INLINE void set_strides(std::int32_t row_stride,
                                     std::int32_t col_stride) {
    YAI_COMPILER_INTRINSIC(tile_set_strides)(value_, row_stride, col_stride);
  }

  YAI_ALWAYS_INLINE std::uint8_t get_valid_zone(std::size_t dim) const {
    return YAI_COMPILER_INTRINSIC(tile_get_valid_zone)(value_, dim);
  }
  YAI_ALWAYS_INLINE std::array<std::uint8_t, 2> get_valid_zone() const {
    return {get_valid_zone(0), get_valid_zone(1)};
  }
  YAI_ALWAYS_INLINE void set_valid_zone_dim(std::size_t dim,
                                            std::uint8_t value) {
    YAI_COMPILER_INTRINSIC(tile_set_valid_zone)(value_, dim, value);
  }
  YAI_ALWAYS_INLINE void set_valid_zone(
      const std::array<std::uint8_t, 2> &value) {
    set_valid_zone(value[0], value[1]);
  }
  YAI_ALWAYS_INLINE void set_valid_zone(std::uint8_t rows,
                                        std::uint8_t cols) {
    YAI_COMPILER_INTRINSIC(tile_set_valid_zone_pair)(value_, rows, cols);
  }

  YAI_ALWAYS_INLINE TileDType get_dtype() const {
    return static_cast<TileDType>(
        YAI_COMPILER_INTRINSIC(tile_get_dtype)(value_));
  }
  YAI_ALWAYS_INLINE void set_dtype(TileDType value) {
    YAI_COMPILER_INTRINSIC(tile_set_dtype)(
        value_, static_cast<std::uint32_t>(value));
  }

  /// 访存后地址自增模式（tm.flags[1:0]，硬件只在访存成功后提交）：
  ///   0 = 关闭，1 = 按行自增（rows * row_stride_bytes），2 = 按列自增
  ///   （cols * col_stride_bytes）。注意它是**跨指令状态**：换块时必须重写 base。
  YAI_ALWAYS_INLINE void set_flags(std::uint8_t value) {
    YAI_COMPILER_INTRINSIC(tile_set_flags)(
        value_, static_cast<std::uint32_t>(value));
  }

  YAI_ALWAYS_INLINE bool get_transpose() const {
    return YAI_COMPILER_INTRINSIC(tile_get_transpose)(value_) != 0;
  }
  YAI_ALWAYS_INLINE void set_transpose(bool value) {
    YAI_COMPILER_INTRINSIC(tile_set_transpose)(value_, value);
  }

  YAI_ALWAYS_INLINE std::byte *get_base() const {
    return reinterpret_cast<std::byte *>(YAI_COMPILER_INTRINSIC(tile_get_base)(value_));
  }
  YAI_ALWAYS_INLINE void set_base(std::byte *value) {
    YAI_COMPILER_INTRINSIC(tile_set_base)(value_, value);
  }

  // ------------------------------------------------------------------------
  // Tile 指令
  // ------------------------------------------------------------------------
  YAI_ALWAYS_INLINE void gather_from_vec(Vec32 &index) {
    YAI_COMPILER_INTRINSIC(tile_gather_from_vec)(value_, yai_handle(index));
  }
  YAI_ALWAYS_INLINE void load() {
    YAI_COMPILER_INTRINSIC(tile_load)(value_);
  }
  YAI_ALWAYS_INLINE void store() {
    YAI_COMPILER_INTRINSIC(tile_store)(value_);
  }
  YAI_ALWAYS_INLINE void fill(std::int8_t value) {
    YAI_COMPILER_INTRINSIC(tile_fill)(value_, value);
  }
  YAI_ALWAYS_INLINE void matmul(Tile &right, Acc &acc, bool zero, bool lt,
                                bool rt) {
    YAI_COMPILER_INTRINSIC(tile_matmul)(value_, yai_handle(right),
                                        yai_handle(acc), zero, lt, rt);
  }
  YAI_ALWAYS_INLINE Acc dequant(const Vec32 &scale,
                                bool col = true) const;
  // Tile i8 -> Acc i32 符号扩展（embedding 的 widen）；out 需先配好
  // dtype=i32、valid_zone = tile 形状。
  YAI_ALWAYS_INLINE void widen(Acc &out) const;
};

struct Acc {
  AccValue value_;
  YAI_ALWAYS_INLINE Acc()
      : value_(YAI_COMPILER_INTRINSIC(acc_create)()) {}
  YAI_ALWAYS_INLINE explicit Acc(AccValue value) : value_(value) {}

  // ------------------------------------------------------------------------
  // 描述/配置状态访问
  // ------------------------------------------------------------------------
  YAI_ALWAYS_INLINE std::byte *get_data() const {
    return reinterpret_cast<std::byte *>(YAI_COMPILER_INTRINSIC(acc_get_data)(value_));
  }
  YAI_ALWAYS_INLINE void set_data(std::byte *value) {
    YAI_COMPILER_INTRINSIC(acc_set_data)(value_, value);
  }

  YAI_ALWAYS_INLINE std::int32_t get_stride(std::size_t dim) const {
    return YAI_COMPILER_INTRINSIC(acc_get_stride)(value_, dim);
  }
  YAI_ALWAYS_INLINE std::array<std::int32_t, 2> get_stride() const {
    return {get_stride(0), get_stride(1)};
  }
  YAI_ALWAYS_INLINE void set_stride(std::size_t dim, std::int32_t value) {
    YAI_COMPILER_INTRINSIC(acc_set_stride)(value_, dim, value);
  }
  YAI_ALWAYS_INLINE void set_stride(
      const std::array<std::int32_t, 2> &value) {
    set_strides(value[0], value[1]);
  }
  YAI_ALWAYS_INLINE void set_strides(std::int32_t row_stride,
                                     std::int32_t col_stride) {
    YAI_COMPILER_INTRINSIC(acc_set_strides)(value_, row_stride, col_stride);
  }

  YAI_ALWAYS_INLINE std::uint8_t get_valid_zone(std::size_t dim) const {
    return YAI_COMPILER_INTRINSIC(acc_get_valid_zone)(value_, dim);
  }
  YAI_ALWAYS_INLINE std::array<std::uint8_t, 2> get_valid_zone() const {
    return {get_valid_zone(0), get_valid_zone(1)};
  }
  YAI_ALWAYS_INLINE void set_valid_zone_dim(std::size_t dim,
                                            std::uint8_t value) {
    YAI_COMPILER_INTRINSIC(acc_set_valid_zone)(value_, dim, value);
  }
  YAI_ALWAYS_INLINE void set_valid_zone(
      const std::array<std::uint8_t, 2> &value) {
    set_valid_zone(value[0], value[1]);
  }
  YAI_ALWAYS_INLINE void set_valid_zone(std::uint8_t rows,
                                        std::uint8_t cols) {
    YAI_COMPILER_INTRINSIC(acc_set_valid_zone_pair)(value_, rows, cols);
  }

  YAI_ALWAYS_INLINE AccDType get_dtype() const {
    return static_cast<AccDType>(
        YAI_COMPILER_INTRINSIC(acc_get_dtype)(value_));
  }
  YAI_ALWAYS_INLINE void set_dtype(AccDType value) {
    YAI_COMPILER_INTRINSIC(acc_set_dtype)(
        value_, static_cast<std::uint32_t>(value));
  }

  /// 访存后地址自增模式（am.flags[1:0]）：同 Tile，0 关闭 / 1 按行 / 2 按列。
  YAI_ALWAYS_INLINE void set_flags(std::uint8_t value) {
    YAI_COMPILER_INTRINSIC(acc_set_flags)(
        value_, static_cast<std::uint32_t>(value));
  }

  YAI_ALWAYS_INLINE std::byte *get_base() const {
    return reinterpret_cast<std::byte *>(YAI_COMPILER_INTRINSIC(acc_get_base)(value_));
  }
  YAI_ALWAYS_INLINE void set_base(std::byte *value) {
    YAI_COMPILER_INTRINSIC(acc_set_base)(value_, value);
  }

  // ------------------------------------------------------------------------
  // Acc 指令
  // ------------------------------------------------------------------------
  YAI_ALWAYS_INLINE void load() {
    YAI_COMPILER_INTRINSIC(acc_load)(value_);
  }
  YAI_ALWAYS_INLINE void store() {
    YAI_COMPILER_INTRINSIC(acc_store)(value_);
  }
  YAI_ALWAYS_INLINE void fill(float value) {
    YAI_COMPILER_INTRINSIC(acc_fill)(value_, value);
  }
  YAI_ALWAYS_INLINE void mask(std::int32_t diag,
                              float value = -3.402823466e+38F) {
    YAI_COMPILER_INTRINSIC(acc_mask)(value_, diag, value);
  }
  YAI_ALWAYS_INLINE void quant(Vec32 &max, Tile &tile) {
    YAI_COMPILER_INTRINSIC(acc_quant)(value_, yai_handle(max),
                                      yai_handle(tile));
  }
  // 用给定的 per-row scale 量化（不修改 scale），对应 tquantx.rows。
  YAI_ALWAYS_INLINE void quant_ext(Vec32 &scale, Tile &tile) {
    YAI_COMPILER_INTRINSIC(acc_quant_ext)(value_, yai_handle(scale),
                                          yai_handle(tile));
  }
  YAI_ALWAYS_INLINE void to_fp32() {
    YAI_COMPILER_INTRINSIC(acc_to_fp32)(value_);
  }

  YAI_ALWAYS_INLINE void add(const Acc &b) {
    YAI_COMPILER_INTRINSIC(acc_add_inplace)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void sub(const Acc &b) {
    YAI_COMPILER_INTRINSIC(acc_sub_inplace)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void mul(const Acc &b) {
    YAI_COMPILER_INTRINSIC(acc_mul_inplace)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void div(const Acc &b) {
    YAI_COMPILER_INTRINSIC(acc_div_inplace)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void add(float value) {
    YAI_COMPILER_INTRINSIC(acc_add_scalar)(value_, value);
  }
  YAI_ALWAYS_INLINE void mul(float value) {
    YAI_COMPILER_INTRINSIC(acc_mul_scalar)(value_, value);
  }
  YAI_ALWAYS_INLINE void abs() {
    YAI_COMPILER_INTRINSIC(acc_abs)(value_);
  }
  YAI_ALWAYS_INLINE void exp() {
    YAI_COMPILER_INTRINSIC(acc_exp)(value_);
  }
  YAI_ALWAYS_INLINE void silu() {
    YAI_COMPILER_INTRINSIC(acc_silu)(value_);
  }
  YAI_ALWAYS_INLINE void sin() {
    YAI_COMPILER_INTRINSIC(acc_sin)(value_);
  }
  YAI_ALWAYS_INLINE void cos() {
    YAI_COMPILER_INTRINSIC(acc_cos)(value_);
  }

  YAI_ALWAYS_INLINE void add(const Vec32 &v, bool col = true) {
    YAI_COMPILER_INTRINSIC(acc_add_vec)(value_, yai_handle(v), col);
  }
  YAI_ALWAYS_INLINE void sub(const Vec32 &v, bool col = true) {
    YAI_COMPILER_INTRINSIC(acc_sub_vec)(value_, yai_handle(v), col);
  }
  YAI_ALWAYS_INLINE void mul(const Vec32 &v, bool col = true) {
    YAI_COMPILER_INTRINSIC(acc_mul_vec)(value_, yai_handle(v), col);
  }
  YAI_ALWAYS_INLINE void div(const Vec32 &v, bool col = true) {
    YAI_COMPILER_INTRINSIC(acc_div_vec)(value_, yai_handle(v), col);
  }

  YAI_ALWAYS_INLINE Vec32 row_max() const;
  YAI_ALWAYS_INLINE Vec32 row_sum() const;
  YAI_ALWAYS_INLINE Vec32 row_sum_sq() const;
  YAI_ALWAYS_INLINE Vec32 row_absmax() const;
  YAI_ALWAYS_INLINE void row_max(Vec32 &v) const;
  YAI_ALWAYS_INLINE void row_sum(Vec32 &v) const;
  YAI_ALWAYS_INLINE void row_sum_sq(Vec32 &v) const;
  YAI_ALWAYS_INLINE void row_absmax(Vec32 &v) const;
  YAI_ALWAYS_INLINE Vec32 row_argmax() const;
  YAI_ALWAYS_INLINE void row_argmax(Vec32 &max_val, Vec32 &idx,
                                    std::int32_t col_base) const;
};

struct Vec8 {
  Vec8Value value_;
  YAI_ALWAYS_INLINE Vec8()
      : value_(YAI_COMPILER_INTRINSIC(vec8_create)()) {}
  YAI_ALWAYS_INLINE explicit Vec8(Vec8Value value) : value_(value) {}

  YAI_ALWAYS_INLINE std::byte *get_data() const {
    return reinterpret_cast<std::byte *>(YAI_COMPILER_INTRINSIC(vec8_get_data)(value_));
  }
  YAI_ALWAYS_INLINE void set_data(std::byte *value) {
    YAI_COMPILER_INTRINSIC(vec8_set_data)(value_, value);
  }
  YAI_ALWAYS_INLINE std::int32_t get_stride() const {
    return YAI_COMPILER_INTRINSIC(vec8_get_stride)(value_);
  }
  YAI_ALWAYS_INLINE void set_stride(std::int32_t value) {
    YAI_COMPILER_INTRINSIC(vec8_set_stride)(value_, value);
  }
  YAI_ALWAYS_INLINE std::int32_t get_valid_zone() const {
    return YAI_COMPILER_INTRINSIC(vec8_get_valid_zone)(value_);
  }
  YAI_ALWAYS_INLINE void set_valid_zone(std::int32_t value) {
    YAI_COMPILER_INTRINSIC(vec8_set_valid_zone)(value_, value);
  }
  YAI_ALWAYS_INLINE Vec8DType get_dtype() const {
    return static_cast<Vec8DType>(
        YAI_COMPILER_INTRINSIC(vec8_get_dtype)(value_));
  }
  YAI_ALWAYS_INLINE void set_dtype(Vec8DType value) {
    YAI_COMPILER_INTRINSIC(vec8_set_dtype)(
        value_, static_cast<std::uint32_t>(value));
  }

  /// 访存后地址自增开关（bm.flags[0]）：1 = 每次访存后 base += len * stride。
  YAI_ALWAYS_INLINE void set_flags(std::uint8_t value) {
    YAI_COMPILER_INTRINSIC(vec8_set_flags)(
        value_, static_cast<std::uint32_t>(value));
  }
  YAI_ALWAYS_INLINE std::byte *get_base() const {
    return reinterpret_cast<std::byte *>(YAI_COMPILER_INTRINSIC(vec8_get_base)(value_));
  }
  YAI_ALWAYS_INLINE void set_base(std::byte *value) {
    YAI_COMPILER_INTRINSIC(vec8_set_base)(value_, value);
  }

  YAI_ALWAYS_INLINE void gather_from_vec(Vec32 &index) {
    YAI_COMPILER_INTRINSIC(vec8_gather_from_vec)(value_, yai_handle(index));
  }
  YAI_ALWAYS_INLINE void load() {
    YAI_COMPILER_INTRINSIC(vec8_load)(value_);
  }
  YAI_ALWAYS_INLINE void store() {
    YAI_COMPILER_INTRINSIC(vec8_store)(value_);
  }
};

struct Vec32 {
  Vec32Value value_;
  YAI_ALWAYS_INLINE Vec32()
      : value_(YAI_COMPILER_INTRINSIC(vec32_create)()) {}
  YAI_ALWAYS_INLINE explicit Vec32(Vec32Value value) : value_(value) {}

  YAI_ALWAYS_INLINE std::byte *get_data() const {
    return reinterpret_cast<std::byte *>(YAI_COMPILER_INTRINSIC(vec32_get_data)(value_));
  }
  YAI_ALWAYS_INLINE void set_data(std::byte *value) {
    YAI_COMPILER_INTRINSIC(vec32_set_data)(value_, value);
  }
  YAI_ALWAYS_INLINE std::int32_t get_stride() const {
    return YAI_COMPILER_INTRINSIC(vec32_get_stride)(value_);
  }
  YAI_ALWAYS_INLINE void set_stride(std::int32_t value) {
    YAI_COMPILER_INTRINSIC(vec32_set_stride)(value_, value);
  }
  YAI_ALWAYS_INLINE std::int32_t get_valid_zone() const {
    return YAI_COMPILER_INTRINSIC(vec32_get_valid_zone)(value_);
  }
  YAI_ALWAYS_INLINE void set_valid_zone(std::int32_t value) {
    YAI_COMPILER_INTRINSIC(vec32_set_valid_zone)(value_, value);
  }
  YAI_ALWAYS_INLINE Vec32DType get_dtype() const {
    return static_cast<Vec32DType>(
        YAI_COMPILER_INTRINSIC(vec32_get_dtype)(value_));
  }
  YAI_ALWAYS_INLINE void set_dtype(Vec32DType value) {
    YAI_COMPILER_INTRINSIC(vec32_set_dtype)(
        value_, static_cast<std::uint32_t>(value));
  }

  /// 访存后地址自增开关（vm.flags[0]）：1 = 每次访存后 base += len * stride。
  YAI_ALWAYS_INLINE void set_flags(std::uint8_t value) {
    YAI_COMPILER_INTRINSIC(vec32_set_flags)(
        value_, static_cast<std::uint32_t>(value));
  }
  YAI_ALWAYS_INLINE std::byte *get_base() const {
    return reinterpret_cast<std::byte *>(YAI_COMPILER_INTRINSIC(vec32_get_base)(value_));
  }
  YAI_ALWAYS_INLINE void set_base(std::byte *value) {
    YAI_COMPILER_INTRINSIC(vec32_set_base)(value_, value);
  }

  YAI_ALWAYS_INLINE void gather_from_vec(Vec32 &index) {
    YAI_COMPILER_INTRINSIC(vec32_gather_from_vec)(value_, yai_handle(index));
  }
  YAI_ALWAYS_INLINE void load() {
    YAI_COMPILER_INTRINSIC(vec32_load)(value_);
  }
  YAI_ALWAYS_INLINE void store() {
    YAI_COMPILER_INTRINSIC(vec32_store)(value_);
  }
  YAI_ALWAYS_INLINE void add(const Vec32 &b) {
    YAI_COMPILER_INTRINSIC(vec32_add)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void sub(const Vec32 &b) {
    YAI_COMPILER_INTRINSIC(vec32_sub)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void mul(const Vec32 &b) {
    YAI_COMPILER_INTRINSIC(vec32_mul)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void div(const Vec32 &b) {
    YAI_COMPILER_INTRINSIC(vec32_div)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void add(float value) {
    YAI_COMPILER_INTRINSIC(vec32_add_scalar)(value_, value);
  }
  YAI_ALWAYS_INLINE void mul(float value) {
    YAI_COMPILER_INTRINSIC(vec32_mul_scalar)(value_, value);
  }
  YAI_ALWAYS_INLINE void rsqrt() {
    YAI_COMPILER_INTRINSIC(vec32_rsqrt)(value_);
  }
  YAI_ALWAYS_INLINE void exp() {
    YAI_COMPILER_INTRINSIC(vec32_exp)(value_);
  }
  YAI_ALWAYS_INLINE void max(const Vec32 &b) {
    YAI_COMPILER_INTRINSIC(vec32_max)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void fill(float value) {
    YAI_COMPILER_INTRINSIC(vec32_fill)(value_, value);
  }
  YAI_ALWAYS_INLINE void copy(const Vec32 &b) {
    value_ = YAI_COMPILER_INTRINSIC(vec32_copy)(yai_handle(b));
  }
};

YAI_ALWAYS_INLINE TileValue yai_handle(const Tile &value) {
  return value.value_;
}
YAI_ALWAYS_INLINE AccValue yai_handle(const Acc &value) {
  return value.value_;
}
YAI_ALWAYS_INLINE Vec8Value yai_handle(const Vec8 &value) {
  return value.value_;
}
YAI_ALWAYS_INLINE Vec32Value yai_handle(const Vec32 &value) {
  return value.value_;
}

// 返回新逻辑值的操作：返回值本身将来由 intrinsic 降为新的 Acc/Vec32 SSA 值。
YAI_ALWAYS_INLINE Acc Tile::dequant(const Vec32 &scale, bool col) const {
  return Acc(YAI_COMPILER_INTRINSIC(tile_dequant)(value_, yai_handle(scale), col));
}
YAI_ALWAYS_INLINE void Tile::widen(Acc &out) const {
  YAI_COMPILER_INTRINSIC(tile_widen)(value_, yai_handle(out));
}

YAI_ALWAYS_INLINE Vec32 Acc::row_max() const {
  return Vec32(YAI_COMPILER_INTRINSIC(acc_row_max)(value_));
}
YAI_ALWAYS_INLINE Vec32 Acc::row_sum() const {
  return Vec32(YAI_COMPILER_INTRINSIC(acc_row_sum)(value_));
}
YAI_ALWAYS_INLINE Vec32 Acc::row_sum_sq() const {
  return Vec32(YAI_COMPILER_INTRINSIC(acc_row_sum_sq)(value_));
}
YAI_ALWAYS_INLINE Vec32 Acc::row_absmax() const {
  return Vec32(YAI_COMPILER_INTRINSIC(acc_row_absmax)(value_));
}
YAI_ALWAYS_INLINE void Acc::row_max(Vec32 &v) const {
  YAI_COMPILER_INTRINSIC(acc_row_max_merge)(value_, yai_handle(v));
}
YAI_ALWAYS_INLINE void Acc::row_sum(Vec32 &v) const {
  YAI_COMPILER_INTRINSIC(acc_row_sum_merge)(value_, yai_handle(v));
}
YAI_ALWAYS_INLINE void Acc::row_sum_sq(Vec32 &v) const {
  YAI_COMPILER_INTRINSIC(acc_row_sum_sq_merge)(value_, yai_handle(v));
}
YAI_ALWAYS_INLINE void Acc::row_absmax(Vec32 &v) const {
  YAI_COMPILER_INTRINSIC(acc_row_absmax_merge)(value_, yai_handle(v));
}
YAI_ALWAYS_INLINE Vec32 Acc::row_argmax() const {
  return Vec32(YAI_COMPILER_INTRINSIC(acc_row_argmax)(value_));
}
YAI_ALWAYS_INLINE void Acc::row_argmax(Vec32 &max_val, Vec32 &idx,
                                       std::int32_t col_base) const {
  YAI_COMPILER_INTRINSIC(acc_row_argmax_merge)(value_, yai_handle(max_val),
                                               yai_handle(idx), col_base);
}

// ===== Acc 二元运算（非成员形式）：生成新 Acc =====
YAI_ALWAYS_INLINE Acc add(const Acc &a, const Acc &b) {
  return Acc(YAI_COMPILER_INTRINSIC(acc_add)(yai_handle(a), yai_handle(b)));
}
YAI_ALWAYS_INLINE Acc sub(const Acc &a, const Acc &b) {
  return Acc(YAI_COMPILER_INTRINSIC(acc_sub)(yai_handle(a), yai_handle(b)));
}
YAI_ALWAYS_INLINE Acc mul(const Acc &a, const Acc &b) {
  return Acc(YAI_COMPILER_INTRINSIC(acc_mul)(yai_handle(a), yai_handle(b)));
}
YAI_ALWAYS_INLINE Acc div(const Acc &a, const Acc &b) {
  return Acc(YAI_COMPILER_INTRINSIC(acc_div)(yai_handle(a), yai_handle(b)));
}
YAI_ALWAYS_INLINE Acc operator+(const Acc &a, const Acc &b) {
  return add(a, b);
}
YAI_ALWAYS_INLINE Acc operator-(const Acc &a, const Acc &b) {
  return sub(a, b);
}
YAI_ALWAYS_INLINE Acc operator*(const Acc &a, const Acc &b) {
  return mul(a, b);
}
YAI_ALWAYS_INLINE Acc operator/(const Acc &a, const Acc &b) {
  return div(a, b);
}

#undef YAI_COMPILER_INTRINSIC
// YAI_ALWAYS_INLINE 保留给用户代码：tile 层函数（签名里带 Tile/Acc/Vec32/Vec8
// 的函数）必须用它。这类值不能跨调用边界（C ABI 不支持），强制内联后没有
// 边界，等价于宏。

