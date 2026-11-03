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

  // 拷贝构造/赋值必须发真正的 acopy。
  //
  // 为什么需要显式写：acc 的机器指令全部是"目的寄存器只作 use、硬件原地写"，
  // 所以 SSA 里源值永远不会被重新定义，默认的按值拷贝会被优化器折叠掉，
  // `Acc t = x;` 实际得到的是同一物理寄存器（别名），改 t 会连带改坏 x。
  // 这里把拷贝显式降到 acc_copy（acopy + cfg.copy），让 C++ 的值语义成立。
  // 禁止隐式拷贝/赋值：`Acc t = x;` 在"指令原地写、IR 无 def"的模型下会被
  // 优化成别名（两个名字同一个物理寄存器），改一个会静默改坏另一个。
  // 要拷贝请显式调用 copy()；要原地算请用带下划线的方法。
  Acc(const Acc &) = delete;
  Acc &operator=(const Acc &) = delete;
  Acc(Acc &&) = default;
  Acc &operator=(Acc &&) = default;

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
  YAI_ALWAYS_INLINE void mask_(std::int32_t diag,
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
  YAI_ALWAYS_INLINE void to_fp32_() {
    YAI_COMPILER_INTRINSIC(acc_to_fp32)(value_);
  }

  YAI_ALWAYS_INLINE void add_(const Acc &b) {
    YAI_COMPILER_INTRINSIC(acc_add_inplace)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void sub_(const Acc &b) {
    YAI_COMPILER_INTRINSIC(acc_sub_inplace)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void mul_(const Acc &b) {
    YAI_COMPILER_INTRINSIC(acc_mul_inplace)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void add_(float value) {
    YAI_COMPILER_INTRINSIC(acc_add_scalar)(value_, value);
  }
  // ── 返回值版本（不带尾下划线）───────────────────────────────────────
  // 语义：返回一个新值，*this 不变。
  // 实现是"派生值"指令（ISA 的 D 与 S 可以是不同寄存器）：后端在 post-RA
  // 只补一条 cfg.copy 克隆描述符，**不搬数据**（没有 acopy）。
  // 唯一的例外是显式 copy()：它就是语义上要一份数据副本。
  // [[nodiscard]] 是关键安全网：写成 `x.mul(k);` 丢掉返回值会被编译器告警。
  [[nodiscard]] YAI_ALWAYS_INLINE Acc add(const Acc &b) const {
    return Acc(YAI_COMPILER_INTRINSIC(acc_add)(value_, yai_handle(b)));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Acc sub(const Acc &b) const {
    return Acc(YAI_COMPILER_INTRINSIC(acc_sub)(value_, yai_handle(b)));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Acc mul(const Acc &b) const {
    return Acc(YAI_COMPILER_INTRINSIC(acc_mul)(value_, yai_handle(b)));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Acc mul(float value) const {
    return Acc(YAI_COMPILER_INTRINSIC(acc_mul_scalar_val)(value_, value));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Acc exp() const {
    return Acc(YAI_COMPILER_INTRINSIC(acc_exp_val)(value_));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Acc mul(const Vec32 &v, bool col = true) const {
    return Acc(YAI_COMPILER_INTRINSIC(acc_mul_vec_val)(value_, yai_handle(v), col));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Acc sub(const Vec32 &v, bool col = true) const {
    return Acc(YAI_COMPILER_INTRINSIC(acc_sub_vec_val)(value_, yai_handle(v), col));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Acc add(float value) const {
    return Acc(YAI_COMPILER_INTRINSIC(acc_add_scalar_val)(value_, value));
  }

  YAI_ALWAYS_INLINE void mul_(float value) {
    YAI_COMPILER_INTRINSIC(acc_mul_scalar)(value_, value);
  }
  /// 返回一个新值（真 acopy）。
  [[nodiscard]] YAI_ALWAYS_INLINE Acc copy() const {
    return Acc(YAI_COMPILER_INTRINSIC(acc_copy)(value_));
  }

  /// 近似倒数 1/x（ISA 没有 div；rcp.approx 的相对误差 ≤ 2^-10）。
  YAI_ALWAYS_INLINE void rcp_() {
    YAI_COMPILER_INTRINSIC(acc_rcp)(value_);
  }
  /// 精确倒数：rcp.approx（误差 ≤ 2^-10）后用 Newton-Raphson 精炼。
  /// r ← r·(2 − x·r)，误差平方收敛：1e-3 → 1e-6 → 1e-12（优于 fp32）。
  /// 依赖"拷贝构造发 acopy"，所以 `Acc t = x` 是真正的拷贝，不会改坏 x。
  YAI_ALWAYS_INLINE void rcp_refine_(const Acc &x, int steps = 2) {
    rcp_();
    for (int i = 0; i < steps; ++i) {
      Acc t = x.copy();   // 显式拷贝
      t.mul_(*this);   // t = x·r
      t.mul_(-1.0f);
      t.add_(2.0f);    // t = 2 − x·r
      mul_(t);         // r = r·(2 − x·r)
    }
  }
  YAI_ALWAYS_INLINE void exp_() {
    YAI_COMPILER_INTRINSIC(acc_exp)(value_);
  }
  YAI_ALWAYS_INLINE void silu_() {
    YAI_COMPILER_INTRINSIC(acc_silu)(value_);
  }

  YAI_ALWAYS_INLINE void add_(const Vec32 &v, bool col = true) {
    YAI_COMPILER_INTRINSIC(acc_add_vec)(value_, yai_handle(v), col);
  }
  YAI_ALWAYS_INLINE void sub_(const Vec32 &v, bool col = true) {
    YAI_COMPILER_INTRINSIC(acc_sub_vec)(value_, yai_handle(v), col);
  }
  YAI_ALWAYS_INLINE void mul_(const Vec32 &v, bool col = true) {
    YAI_COMPILER_INTRINSIC(acc_mul_vec)(value_, yai_handle(v), col);
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

  // 同 Acc：显式发 vcopy，避免"赋值被折叠成别名"。
  // 同 Acc：禁止隐式拷贝，显式 copy() 才算真拷贝。
  Vec32(const Vec32 &) = delete;
  Vec32 &operator=(const Vec32 &) = delete;
  Vec32(Vec32 &&) = default;
  Vec32 &operator=(Vec32 &&) = default;

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
  YAI_ALWAYS_INLINE void add_(const Vec32 &b) {
    YAI_COMPILER_INTRINSIC(vec32_add)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void sub_(const Vec32 &b) {
    YAI_COMPILER_INTRINSIC(vec32_sub)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void mul_(const Vec32 &b) {
    YAI_COMPILER_INTRINSIC(vec32_mul)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void add_(float value) {
    YAI_COMPILER_INTRINSIC(vec32_add_scalar)(value_, value);
  }
  // ── 返回值版本（不带尾下划线），语义同 Acc：
  //    派生值指令（D != S），post-RA 只补 cfg.copy，不搬 128 B 数据 ──
  [[nodiscard]] YAI_ALWAYS_INLINE Vec32 add(const Vec32 &b) const {
    return Vec32(YAI_COMPILER_INTRINSIC(vec32_add_val)(value_, yai_handle(b)));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Vec32 mul(const Vec32 &b) const {
    return Vec32(YAI_COMPILER_INTRINSIC(vec32_mul_val)(value_, yai_handle(b)));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Vec32 exp() const {
    return Vec32(YAI_COMPILER_INTRINSIC(vec32_exp_val)(value_));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Vec32 sub(const Vec32 &b) const {
    return Vec32(YAI_COMPILER_INTRINSIC(vec32_sub_val)(value_, yai_handle(b)));
  }
  [[nodiscard]] YAI_ALWAYS_INLINE Vec32 mul(float value) const {
    return Vec32(YAI_COMPILER_INTRINSIC(vec32_mul_scalar_val)(value_, value));
  }

  YAI_ALWAYS_INLINE void mul_(float value) {
    YAI_COMPILER_INTRINSIC(vec32_mul_scalar)(value_, value);
  }
  YAI_ALWAYS_INLINE void rsqrt_() {
    YAI_COMPILER_INTRINSIC(vec32_rsqrt)(value_);
  }
  /// 近似倒数 1/x（ISA 没有 div；rcp.approx 的相对误差 ≤ 2^-10）。
  YAI_ALWAYS_INLINE void rcp_() {
    YAI_COMPILER_INTRINSIC(vec32_rcp)(value_);
  }
  /// 精确倒数：rcp.approx + Newton-Raphson 精炼（误差平方收敛）。
  YAI_ALWAYS_INLINE void rcp_refine_(const Vec32 &x, int steps = 2) {
    rcp_();
    for (int i = 0; i < steps; ++i) {
      Vec32 t = x.copy();
      t.mul_(*this);
      t.mul_(-1.0f);
      t.add_(2.0f);
      mul_(t);
    }
  }
  YAI_ALWAYS_INLINE void exp_() {
    YAI_COMPILER_INTRINSIC(vec32_exp)(value_);
  }
  YAI_ALWAYS_INLINE void max_(const Vec32 &b) {
    YAI_COMPILER_INTRINSIC(vec32_max)(value_, yai_handle(b));
  }
  YAI_ALWAYS_INLINE void fill(float value) {
    YAI_COMPILER_INTRINSIC(vec32_fill)(value_, value);
  }
  /// 返回一个新值（真 vcopy）。注意与下面的 void copy(const Vec32&) 不同：
  /// 那个是"把 b 拷进 *this"，这个是"把 *this 拷出来"。
  [[nodiscard]] YAI_ALWAYS_INLINE Vec32 copy() const {
    return Vec32(YAI_COMPILER_INTRINSIC(vec32_copy)(value_));
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
// 注意：ISA 已删除 div（只有 rcp.approx / rsqrt.approx / exp2.approx），
// 所以这里不提供 div()/operator/；需要倒数请用 Acc::rcp_()/rcp_refine_()。
YAI_ALWAYS_INLINE Acc operator+(const Acc &a, const Acc &b) {
  return add(a, b);
}
YAI_ALWAYS_INLINE Acc operator-(const Acc &a, const Acc &b) {
  return sub(a, b);
}
YAI_ALWAYS_INLINE Acc operator*(const Acc &a, const Acc &b) {
  return mul(a, b);
}

#undef YAI_COMPILER_INTRINSIC
// YAI_ALWAYS_INLINE 保留给用户代码：tile 层函数（签名里带 Tile/Acc/Vec32/Vec8
// 的函数）必须用它。这类值不能跨调用边界（C ABI 不支持），强制内联后没有
// 边界，等价于宏。

