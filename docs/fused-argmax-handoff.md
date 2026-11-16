# lm_head 融合 argmax（已完成）

## 目标

在 `qwen3-tile-ops` 里把 lm_head 的 GEMM 和 argmax 融合：每个 32 列输出块算完
（i32 → adequant → ×列 scale）后不写 logits，而是立即：

```
tmax = of.row_max()        # f32
tidx = of.row_argmax()     # i32
gidx = tidx + n            # 全局索引
mask = (best < tmax)       # vcmp.lt
best = select(mask, tmax, best)
bidx = select(mask, gidx, bidx)
```

`(best, bidx)` 是循环携带的 Vec32 值，整轮 n 循环不落内存；每个 m 行块结束
只 store 一次。单 token 推理时就是整个 lm_head 只写回一次。

## 两种模式

控制块 `CTRL = [n_prompt, n_gen, max_seq, mode]`：

| mode | 路径 | lm_head | 选 token |
|---|---|---|---|
| 0 | **融合**（生产） | `gemm_nt_argmax`，不写 logits | 读 `argi[0]`（bidx）一次 |
| 1 | **非融合**（参考） | `gemm_nt`，写完整 logits | `argmax_logits` 扫 logits |

两种模式跑同样的 prefill/decode 循环，生成 id 写回 `ids[pos]`。`make check`
会对同一个 blob 各跑一遍：

- mode 0：`compare.py --fused`，跳过 logits stage，比较 `argf`/`argi` 与
  `ref_logits` 的 max/argmax；
- mode 1：`compare.py`，逐 stage 比较，包括完整 logits。

## 融合实现

### epilogue

[`kernels/lm_argmax.h`](../kernels/lm_argmax.h)：

```cpp
YAI_ALWAYS_INLINE void argmax_epilogue(const Vec32 &as, Vec32 &ws,
                                       Acc &o, Vec32 &best, Vec32 &bidx,
                                       std::uint8_t rows, std::uint8_t cols,
                                       std::uint32_t n, std::int32_t *counter) {
  Acc of{};
  init_f32_acc(of, rows, cols);
  ws.load();                      // 本 n 块的列 scale
  adequant(of, o, ws, as);        // 与 gemm_nt 同一乘序：(f32(o)·ws[j])·as[i]
  Vec32 tmax = of.row_max();
  Vec32 gidx = of.row_argmax();
  gidx.set_dtype(Vec32DType::I32);
  gidx.add_(bits_of(n));          // 全局下标
  Vec32 mask = cmp_lt(best, tmax);
  best = select(tmax, best, mask);
  bidx = select(gidx, bidx, mask);
}
```

`gemm_nt_argmax` 复用 `gemm_nt` 的 G=8 快路径；组内 8 个输出块算完 K 之后，
逐个调用上面的 epilogue。n 循环结束才 `store_vec(best, best_out, m)` /
`store_vec(bidx, bidx_out, m)`。

### prefill 最后一层 tail

`forward_chunk(..., optimize_tail)` 把最后一层单独处理：

- 层 0..L-2：所有行都要完整 attention + MLP，因为下一层的 K/V 依赖当前层输出。
- 层 L-1：所有行都要 K/V projection + k/v norm/rope/quant + 写 KV cache。
  - 非最终 prefill chunk（`lm_out == None`）：到此为止，最后一层的
    Q/attention/O/MLP 整段跳过；
  - 最终 chunk：`optimize_tail=true`（mode 0）只算最后一行，
    `optimize_tail=false`（mode 1 参考）仍然算所有行。
- final norm + lm_head/argmax 仍然只算最后一行。

为什么最后一层可以这样：非最后 token 的最后一层 Q/attention/O/MLP 没有消费
者——它们的 K/V 已经写进 cache，后续 token 只需要这些 K/V；只有需要 logits
的那个 token（最后一行）才需要继续算 Q/attn/O/MLP。层 0..L-2 不能这样做，
因为它们的输出是下一层的输入。

验证时，mode 0 的 `compare.py --fused` 对最后一层的 tail stage 只比较
`[last row]`，K/V 和 KV cache 仍然比较所有行；mode 1 仍然比较完整最后一层，
所以参考路径没有被放松。

注意：tile 指令是固定 32 行块，最终 chunk 的 1 行 tail 在模拟器里仍然发同样
多的 tile 指令，steps 不会明显下降；收益在有效 MAC/访存。真正的指令数节省
来自非最终 prefill chunk 整段 tail 跳过（64 token prefill：4,438,896 →
4,392,666）。

### 编译器支持

- `cmp_lt` / `select` 都是**值形式**：返回新的 Vec32 SSA 值，调用方句柄会更新。
- TD：`vcmp.lt` / `vselect` 改成真正的 `(outs XTVOp:$d)`，并新增
  `XAIcoreVCmpLtVal` / `XAIcoreVSelectVal` 伪指令；lowering 用 `emitDef`。
- post-RA 只补 `cfg.copy` 克隆描述符，不搬数据。
- `vcmp` 的 mask 目的必须是 u32，而源 A 是 f32：lowering 先做一个 u32 的
  cfg 载体，D 用 tie 约束钉在 cfg 上（D==cfg）。
- `vselect` 的 D 用 tie 约束钉在 else 源 B 上（D==B），实现真正的就地更新；
  A/M 与 D 同时活跃，RA 不会把 D 分到 mask 的寄存器上。

生成的关键汇编：

```asm
vcmp.lt   v17, v0, v3
cfg.copy  vc0, vc3
vselect   v0, v17, v3, v0      # best = mask ? tmax : best
cfg.copy  vc1, vc4
vselect   v1, v17, v4, v1      # bidx = mask ? gidx : bidx
```

### 修掉的两个问题

1. **状态别名**：早期 WIP 里 `st_f`/`st_i` 都从 `argf.data` 起、又都读写
   row 0，best 和 bidx 落在同一个地址。现在状态是 `gemm_nt_argmax` 内部的
   `Vec32` 循环携带值，输出分别写 `w.argf` / `w.argi`。
2. **vcmp/vselect 的 SSA 链**：早期是 void 原地内建，lowering 直接写目标
   寄存器，调用方句柄不更新。现在改成值形式。

## 验证

| 验证 | 结果 |
|---|---|
| `make check` mode 0 | **0 stage(s) differ**；fused best=**12.620536** @198 与 ref_logits 逐位相同 |
| `make check` mode 1 | **0 stage(s) differ**；完整 logits max\|Δ\|=0，生成 id 一致 |
| `make small`（1 层 / 1024 词表，生成 2） | 两种模式都 **0 stage(s) differ**；生成 id `986 986` 一致 |
| `make chat`（15 token prompt，生成 6） | `151667 198 99692 3837 3837 35946`，与迁移前记录一致 |
| `make value-ops` | 19/19，0 case differ |

## 构建 / 运行

```bash
export PATH=/nfs/home/share/riscv/bin:$PATH
make big
make check       # mode 0 + mode 1 各跑一遍

# 编译器（改 TD/lowering 后）
cd ../llvm-project/build && CCACHE_DISABLE=1 ninja clang

# 手动跑指定模式：先复制 blob，再只改 CTRL[3]
python3 tools/patch_blob.py --blob /tmp/mode1.bin --header build/full/model_big.h \
    --mode-only --mode 1
build/sim_bridge build/qwen3_full.bin /tmp/mode1.bin /tmp/out.bin
```

## 限制

- 融合快路径要求 `K % 32 == 0`、`N % 32 == 0`（Qwen3-0.6B 的 K=1024、
  N=151936 满足）。
- 只做贪心：mode 0 融合，mode 1 非融合；host 采样需要另加单步接口。
