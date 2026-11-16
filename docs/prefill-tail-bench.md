# prefill 最后一层 tail 优化：不同上下文收益

## 测法

- 模型：Qwen3-0.6B，28 层，hidden=1024，vocab=151936。
- 只测 prefill：`n_gen=0`，`mode=0`（融合 lm_head+argmax）。
- 输入：32 token 一块，超过 32 的部分按 chunk 继续；prompt ids 用
  `100,110,120,...` 的固定模式。
- `steps` 来自 `sim_bridge` 的 `steps=N`，是模拟器的指令数；wall time 包含
  791 MB blob 装载，噪声较大，不作为主指标。
- 对照：

| 名称 | 最后一层行为 |
|---|---|
| baseline | 所有 prefill chunk 都算完整最后一层（Q/attn/O/MLP 所有行） |
| fused-opt | 当前 mode 0：非最终 chunk 跳过 tail；最终 chunk 只算最后一行 |
| nonfused-opt | 当前 mode 1：同样的 tail 跳过；最终 chunk 算所有行，写 logits |

baseline 是把 `do_tail` 临时改成 `true` 编出来的对照镜像，其余代码相同。

## 结果

| prompt tokens | baseline steps | fused-opt steps | 节省 | 节省% | nonfused-opt steps |
|---:|---:|---:|---:|---:|---:|
| 16 | 2,449,244 | 2,454,080 | **-4,836** | **-0.20%** | 2,363,838 |
| 32 | 2,450,141 | 2,454,977 | **-4,836** | **-0.20%** | 2,364,735 |
| 64 | 4,438,896 | 4,392,666 | **46,230** | **1.04%** | 4,302,424 |
| 96 | 6,497,539 | 6,398,179 | **99,360** | **1.53%** | 6,307,937 |
| 128 | 8,626,070 | 8,471,516 | **154,554** | **1.79%** | 8,381,274 |
| 192 | 13,092,796 | 12,821,662 | **271,134** | **2.07%** | 12,731,420 |
| 256 | 17,839,074 | 17,443,104 | **395,970** | **2.22%** | 17,352,862 |

## 结论

1. **≤32 token 没有收益，反而略亏 0.2%。**
   只有一个 final chunk，没有非最终 chunk 的 tail 可跳；最终 chunk 的 1 行
   tail 仍然是固定 32 行 tile 指令，指令数不降，额外的 tail view/分支还多
   了约 4.8k 条指令。

2. **≥64 token 开始有收益，且随上下文增长。**
   每个非最终 32-token chunk 可以整段跳过最后一层的 Q/attn/O/MLP，只保留
   K/V。节省的 per-chunk 指令数随 KV 长度增长：

   | 上下文 | 非最终 chunk 数 | 总节省 | 平均每 chunk |
   |---:|---:|---:|---:|
   | 64 | 1 | 46,230 | 46,230 |
   | 96 | 2 | 99,360 | 49,680 |
   | 128 | 3 | 154,554 | 51,518 |
   | 192 | 5 | 271,134 | 54,227 |
   | 256 | 7 | 395,970 | 56,567 |

   256 token 时约省 **2.2%** 的 prefill 指令；更长的上下文占比会继续上升。

3. **1 行 tail 的收益不在指令数，在有效 MAC/访存。**
   tile 指令是固定 32 行块，最终 chunk 只算 1 行时指令数几乎不变；真正进入
   systolic array 的有效行数、Q/O/MLP 的访存量会下降。lm_head 在 baseline 和
   optimized 里本来就都只算最后一行，这部分大 saving 不在这张表里。

4. **prefill-only 下 nonfused-opt 比 fused-opt 更快。**
   `n_gen=0` 时 mode 1 不跑 `argmax_logits`，只是 `gemm_nt` 写 logits；mode 0
   每个 32 列块还要做 row_max/argmax/vcmp/vselect。融合的收益在**生成阶段**：
   每个 token 省掉 4,748 个块的 logits 写 + 读 + 标量 argmax 往返。

## 备注

- 这张表是 32-token chunk 的固定切分；如果将来改成 64 或 128 token 的
  prefill chunk，非最终 chunk 的 tail 跳过会更早生效，收益曲线会前移。
- 如果需要进一步降低短上下文（≤32）的 0.2% 回退，可以给“只有一个 chunk”
  的情况保留旧的全 tail 编译路径（模板特化），但收益很小。
