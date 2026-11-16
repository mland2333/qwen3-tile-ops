# per-128-group GEMM 原语（第一步）

## 动机

激活按 128 个元素一组量化时，一个组在 acc 里放得下；但 K 维会跨多个组，
不能像 per-token 那样 K 全累加完再乘一次 scale。需要：

```
for 每个 K 组 g:
    int32 partial = A_g · B_g
    f32 partial_f = partial * a_scale[g] * b_scale
    f32 accum    += partial_f
```

## 实现

[`kernels/group_gemm.h`](../kernels/group_gemm.h)：

```cpp
YAI_OP_INLINE void gemm_nt_group_store(const MatrixView &a,
                                       const MatrixView &a_scale,
                                       const MatrixView &b,
                                       const MatrixView &b_scale,
                                       const MatrixView &out,
                                       std::uint32_t group_k = 128);
```

- `a`：`[rows][K]` i8，`a_scale`：`[rows][K/128]` f32（每个 K 组、每行一个 scale）。
- `b`：`[N][K]` i8，`b_scale`：`[N]` f32（per-channel）。
- `out`：`[rows][N]` f32。
- N 按 128 列一组（4 个 32 列块）。

acc 预算：

| 用途 | acc 数 |
|---|---:|
| int32 partial（4 个 32 列块） | 4 |
| f32 partial | 4 |
| f32 accum | 4 |
| **合计** | **12 / 16** |

tile 预算：4 个 N 块 × 4 个 K 块 = 16 个 B tile，加 4 个 A tile = **20 / 32**。

`zero` 标志必须编译期常量，所以每个 K 组的第一个 32 块单独放在循环外，
避免后端把 `if` 折成 select。

## 验证

`make group-gemm`：32×256 i8 @ 128×256 i8，K 分两组 128。

- 被测：`gemm_nt_group_store`，两组 scale 都是 1.0。
- 对照：两次 `gemm_nt`（各 K=128）+ `add_rows`。
- 结果：**bit-exact**（`ok per-group gemm ... bit-exact`）。

这验证了 per-group K 累加、12 acc 预算、`adequant` 的 per-group 行 scale 和
per-channel 列 scale，以及 20 tile 的权重装载。

## per-head 量化原语

[`kernels/quant.h`](../kernels/quant.h) 新增：

```cpp
YAI_OP_INLINE void quant_rows_heads(const MatrixView &x,
                                    const MatrixView &qi8,
                                    const MatrixView &scale,
                                    std::uint32_t group_cols);
```

`x [rows][cols] f32 -> qi8 [rows][cols] i8 + scale [rows][groups] f32`，
每 `group_cols`（Qwen3 head_dim=128）一组、每行一个 scale。op 内部仍是
“第一遍组内 absmax、第二遍 tquantx”，只是 scale 从整行一个变成每组一个。

`make group-gemm` 同时验证：

```text
ok   per-group gemm  (32x256 @ 128x256, K split 2x128) bit-exact
ok   per-head quant (32x256 f32 -> i8, 2 groups of 128) bit-exact
```

per-head 量化的对照是两次 `quant_rows`（各 128 列）+ 独立的 per-token
scale，逐位相同。

## Q/K/V per-head 已接入模型

- `src/main_big.cpp`：`w.qs` 改成 `[N][heads]`，`kcscale`/`vcscale` 改成
  `[max_seq][kv_heads]`；q/k/v 全部走 `quant_rows_heads(..., HD)`。
- `attention_heads_gqa_tokscale`：按 head slice `q_scale[..,h]` /
  `k_scale[..,kv]` / `v_scale[..,kv]`。
- blob layout：`kQS = 4*seq*heads`、`kKS/kVS = 4*seq*kv_heads`；
  KV cache 的 scale 段是 `4*max_seq*kv_heads`，`kKV_VOFF`/`kKV_STRIDE` 跟着变。
- 参考：`quant_rows_heads` + per-head attention slice + `Cache.k_s/v_s`
  改成 `[max_seq][kv_heads]`。
- 当前全量/小模型 `make check` 两种模式都是 **0 stage(s) differ**；
  新 reference 的 fused best=**12.406** @400；`make chat` 6 token 为
  `151667 198 99692 3837 20002 104029`。

注意：现在 Q/K/V 仍然是 `gemm_nt` 先写 f32，再 `quant_rows_heads` 读回来，
所以 per-head 量化的**带宽收益还没有拿到**；它先把 scale 布局/attention 路径
打通。下一步是把 norm/rope/quant 融进 Q/K/V GEMM 的 epilogue，省掉 f32 q/k/v
的写出/读入。

## 全模型 per-group 已完成

- q/k/v 按 head=128；h8 / attn-o8 / SwiGLU-s8 / final norm 用 256 列一组。
- Q/K/V/gate/up/O/down 全部 `gemm_nt_group_store`；lm_head mode 1 也是
  per-group，mode 0 用 `gemm_nt_group_argmax`。
- `make check` full/small 两种模式 **0 stage(s) differ**；fused best=12.6227
  @400；`fp32_ref.py` greedy 6 token 与设备完全一致。
- 代价：per-group GEMM 是 correctness-first，N 按 128 块、K 按 128 组，
  A tile 的复用不如旧 G=8 快路径，32 token prefill 指令数 ~2.4M -> 6.98M（128 组）-> 6.17M（256 组）。
  下一步是性能优化（更大 N 分组、减少 b_scale 重载、epilogue 融合），
  量化粒度本身已经到位。

## 已融合：O/down residual

`gemm_nt_group_store(..., const MatrixView *add)`：O 和 down 的 GEMM 结果直接
加到 `x` 上，不再写 o f32 再 `add_rows`。`compare.py` 跳过 o stage（x 仍然
逐位比较）。32 token prefill steps 6.168M -> **6.134M**，省掉每层 2×128 KiB
的 o f32 write+read。

## 已融合：attention 输出 quant

`attention_head_hd128_blocks` 的尾部直接对 r0..r3 做 per-head absmax、
`quant_ext`，写 o8 + os；不再写 f32 attn，也不再单独跑 `quant_rows_heads`。
`compare.py` 跳过 attn stage，o8/os 和 x 仍然逐位比较。

全量/小模型 `make check` 两种模式 0 stage；fused best=12.674 @400；
`make chat` 6 token 仍与 fp32 一致。steps 约 6.14M（attention quant 让
O GEMM 变成 per-head 128 组，指令数基本持平）。

## 已融合：Q/K norm + rope + per-head quant

`kernels/norm_rope_quant.h::norm_rope_quant_heads`：按 head 读一次 f32 q/k，
在 acc 上做 rmsnorm、rope、per-head absmax/quant，只写 q8/qs（K 写 KV
cache/scale）。q/k 的 f32 norm/rope 中间写回被消掉；`compare.py` 跳过 q/k
stage（它们现在是 GEMM 的 pre-norm f32，参考记录的是 post-norm）。

全量/小模型 `make check` 两种模式 0 stage；fused best=12.674 @400；
`make chat` 6 token 仍与 fp32 一致。steps 6.14M -> **6.049M**。

## 下一步

这个原语目前只把结果存成 f32，还没有接进模型。要落地 Q/K/V per-head：

1. 在 `gemm_nt_group_store` 后面加 per-head 的 epilogue：norm/rope/quant，
   输出 i8 tile + `[rows][heads]` scale。
2. attention driver 把 `q_scale/k_scale/v_scale` 按 head slice。
3. 更新 `tools/ref_qwen3.py` 的 per-head 量化/attention 参考和 `compare.py`
   的 scale stage 形状。
4. O projection 和 down projection 用同样的 per-group K 累加，N=128；
   gate/up + SwiGLU + quant 也按 128 列一组融合。

注意：per-head/per-group 会改变模型的量化粒度，`make check` 的逐位参考
必须重新生成；这一步不是“只改 kernel、结果不变”的优化。
