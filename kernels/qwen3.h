#ifndef YAI_QWEN3_OPS_QWEN3_H
#define YAI_QWEN3_OPS_QWEN3_H

// ============================================================================
// Qwen3 算子库（tiny / small / 0.6B 共用同一套 kernel 头）
//
// 分层规则（整个库只有这一条硬约束）：
//   编排层（src/main*.cpp）只碰标量 + MatrixView；
//   Tile/Acc/Vec32/Vec8 这些寄存器值只在算子内部出现，绝不跨函数边界。
//
// 文件一览（按一次前向的数据流排序）：
//   matrix.h     MatrixView + sub_view + block_zone（大矩阵描述符，纯标量）
//   io.h         MatrixView 的 32x32 块 <-> Acc/Tile/Vec32（以及"新值先配配置"的
//                辅助函数），是所有算子和硬件之间的唯一通道
//   rmsnorm.h    rmsnorm_rows：行 RMSNorm（两遍：先 sumsq 再缩放）
//   quant.h      quant_rows：per-token int8 量化（absmax → scale/127 → tquantx）
//   gemm.h       gemm_nt：int8 GEMM（K 块 int32 累加 + 双 scale epilogue）
//   embedding.h  embedding_rows：词表 gather + widen + 反量化
//   rope.h       rope_head：单头 hd=32 的 rotate-half（tiny 模型用）
//   rope_gqa.h   rope_head_hd：任意 head_dim、按 32 列分块、按 32 行分块
//   attention.h  attention_heads：单 32-token 块的多头 attention（tiny 用）
//   attention_gqa.h  三种：
//                 attention_head_hd        单 KV 块、多 K 块的泛化单头
//                 attention_heads_gqa      head-major scale 的 GQA driver
//                 attention_heads_gqa_tokscale  per-token scale + KV 分块
//                                          + online softmax（0.6B 用这个）
//   mlp.h        swiglu（手写 silu，对齐参考的 f32 式子）+ add_rows（残差）
//   config.h     编译期开关 + 数值常量（eps、1/127、1/sqrt(hd)、dtype 码）
//
// 数值目标：与 host 参考 tile_tests/build/qwen3_tile 逐位一致（int8 量化会把
// 1 ulp 的差异放大成 LSB 翻转，所以每个算子里"运算顺序/结合方式"都刻意与参考
// 对齐，并在注释里标出哪一步是刻意的）。
// ============================================================================
#include "attention.h"
#include "config.h"
#include "gemm.h"
#include "io.h"
#include "matrix.h"
#include "mlp.h"
#include "quant.h"
#include "rmsnorm.h"
#include "rope.h"

#endif
