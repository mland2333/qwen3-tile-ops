#ifndef YAI_QWEN3_OPS_CONFIG_H
#define YAI_QWEN3_OPS_CONFIG_H

// ============================================================================
// 算子库的编译期开关 + 数值常量
//
// 这个头文件只放"编译期决定、所有算子共享"的东西：
//   1) 两个开关：算子是否内联（YAI_OP_INLINE）、块循环怎么展开（YAI_LOOP_UNROLL）
//   2) 硬件/数值常量：块大小、eps、1/127、1/sqrt(head_dim)、dtype 码
//
// 注意：矩阵的形状、跨步、地址**不在这里**，它们由 MatrixView 在运行期携带
// （见 matrix.h）；dtype 码必须和 ISA 里 cfg.setx 的 dtype 字段一致。
// ============================================================================

// ----------------------------------------------------------------------------
// 开关 1：算子级内联
//
// 这些算子的签名只有 MatrixView（纯指针/标量），所以 out-of-line 也合法；
// 但内联后 MatrixView 的维度会变成编译期常量，块循环（unroll(full)）能完全
// 展开，机器码指令数少 ~3.5 倍（0.6B prefill：12.3M vs 42.7M 条）。
//
// 前提是 lowering 必须为每个"新定义的 NPU 值"把它的配置寄存器一起写出来
// （如 vec32_copy 的 cfg.copy vcD, vcS）；否则配置寄存器是跨算子的隐式
// 全局状态，内联/不内联会让数值随寄存器分配变化。
// 需要对照实验时可以 -DYAI_OP_INLINE=inline 关掉。
// ----------------------------------------------------------------------------
#ifndef YAI_OP_INLINE
#define YAI_OP_INLINE YAI_ALWAYS_INLINE
#endif

// ----------------------------------------------------------------------------
// 开关 2：块循环的展开策略
//
// 0.6B prefill 实测（28 层，logits 三者都与 /tmp/ref32.bin 逐位相同）：
//   unroll(full)      指令 12.330M，guest 94 KB   ← 默认，无循环开销
//   unroll(disable)   指令 12.353M（+0.2%），guest 8 KB（小 11 倍）
//   unroll_count(4)   指令 12.307M，guest 13 KB
//
// 结论：展开只影响"代码大小 vs 循环开销"，与数值无关。代码区吃紧时用
//   -DYAI_LOOP_UNROLL='unroll(disable)'（或改这里的默认值）。
// 注意：每个块循环里都会新建 Acc/Tile/Vec32 值（值不跨回边），这是能滚动
// 的前提；若某个值需要跨循环迭代存活，就必须展开成直线代码（见下）。
// ----------------------------------------------------------------------------
#ifndef YAI_LOOP_UNROLL
#define YAI_LOOP_UNROLL unroll(full)
#endif

namespace qwen3_ops {

// 硬件 tile 的边长：一个 Tile/Acc 是 32x32，Vec32 是 32 个元素。
// 所有算子的块循环步长都用它；valid_zone 的尾块用 block_zone() 裁剪。
inline constexpr int kBlock = 32;

// 分块 KV attention（attention_head_hd128_blocks）目前写死 head_dim=128
// （= 4 个 32 宽输出块）。其它 head_dim 需要另写一份或用模板展开。
inline constexpr int kHeadDim128 = 128;

// 单头 RoPE（rope.h，head_dim=32 的小模型）的 rotate-half 半宽。
inline constexpr int kHalf = 16;

// RMSNorm 的 eps：与 host 参考、HF 的 rms_norm_eps=1e-6 一致。
// 注意它是加到"整行平方均值"上的，不是加到每个元素上。
inline constexpr float kEps = 1e-6f;

// per-token 量化的 scale 系数：scale = absmax / 127（int8 的有效范围）。
// 与 host 参考 quant() 里的 1.f/127 完全一致；用常量而不是除法，是因为
// 设备端没有标量浮点除法指令（会退化成 libgcc 的 __divsf3）。
inline constexpr float kInv127 = 1.0f / 127.0f;

// ----------------------------------------------------------------------------
// attention 的 QK 缩放 1/sqrt(head_dim)
//
// 乘的顺序很重要：先乘 Q 的 per-token scale（byrow）、再乘 K 的（bycol）、
// 最后乘这个常量，然后才 mask。这与 host 参考 qwen3_by_tile.cpp::attention
// 里的 `score.mul(1.f / std::sqrt(float(HD)))` 表达式逐位对齐。
//
// 为什么不现场算：设备端没有 sqrt（ISA 只有 vrsqrt.approx），而这个值在
// 编译期就确定，所以按常用 head_dim 直接给常量表。
// ----------------------------------------------------------------------------
inline constexpr float kInvSqrt32 = 0.17677669529663687f;
inline constexpr float kInvSqrt64 = 0.125f;
inline constexpr float kInvSqrt128 = 0.08838834764831845f;

// MatrixView::dtype 的取值，必须与 ISA 的 dtype 字段（也是 AccDType/
// TileDType/Vec32DType 的码值）一致：i8=0, u8=1, i32=2, u32=3, f32=4。
inline constexpr std::uint8_t kDtI8 = 0;
inline constexpr std::uint8_t kDtU8 = 1;
inline constexpr std::uint8_t kDtI32 = 2;
inline constexpr std::uint8_t kDtU32 = 3;
inline constexpr std::uint8_t kDtF32 = 4;

} // namespace qwen3_ops

#endif
