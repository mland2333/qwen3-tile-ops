# qwen3-tile-ops

**在 Tile NPU（`+xaicore` 扩展）上跑完整 Qwen3-0.6B 推理的算子库**，外加一个把算子串成
prefill + KV cache + decode + argmax 的 driver。

只包含"算子"这一层：没有 tokenizer、没有 Python 权重转换、没有词表/blobs、没有测试数据。
那些属于上层项目（见文末"与其它仓库的关系"）。

## 目录

```
include/tile.h        tile API：Tile / Acc / Vec32 / Vec8 四个寄存器类的 C++ 封装
test_include/         -nostdinc++ 用的极简 <cstdint>/<cstddef>/<array>
kernels/              算子（13 个头文件，见下表）
config/model_big.h    权重/激活在 blob 里的布局（kLOGITS、kKV_BASE…），由上层工具生成
src/main_big.cpp      0.6B 的 driver：forward_chunk(pos, rows) + 生成循环
src/crt0.S            guest 启动代码（sp 取 0x8017F000，结束用 kernel.end）
link.ld               链接脚本（入口 0x80000000）
tools/check_logits.py 从跑完的 blob 里取 logits 和参考比（验证用，不属于算子库）
```

## 算子清单

| 文件 | 算子 | 关键点 |
|---|---|---|
| `gemm.h` | `gemm_nt` | int8 GEMM（右操作数逻辑转置）+ 双向 scale。**n 方向 8 块寄存器分块**（1 A tile + 8 B tile + 8 acc），描述符按列自增，k 循环只剩 `tload`+`mma`（19 条/迭代）。K 全程 i32 累加，末尾一次 `acvt` + 两次 scale |
| `quant.h` | `quant_rows` | per-token absmax → i8 + f32 scale；两趟（写 scale、再量化） |
| `rmsnorm.h` | `rmsnorm_rows` | 两趟：`Σx²` → `rsqrt` → 逐列乘 weight |
| `embedding.h` | `embedding_rows`、`embedding_rmsnorm_rows` | `tgather` 按 id 取行；后者把 rmsnorm 的 `sumsq` 顺手累积掉一趟读 |
| `attention.h` | `attention_head`、`attention_heads` | 单 KV 块版（KV ≤ 32、head-major scale），tiny/small 模型用 |
| `attention_gqa.h` | `attention_heads_gqa_tokscale`、`attention_head_hd128_blocks`、`pv_block` | **0.6B 实际用**：KV 32 行分块 + online softmax（running max/den + `corr=exp(m_old-m_new)` 重缩放）、S/P 不落内存、P 就地量化成 i8、因果整块跳过 |
| `rope.h` / `rope_gqa.h` | `rope_head`、`rope_head_hd` | 旋转位置编码，cos/sin 表由上层生成后放进 blob |
| `mlp.h` | `swiglu`、`add_rows` | `silu(g)*u` 逐元素；残差相加 |
| `matrix.h` | `sub_view`、`block_zone` | 视图切片 / 尾块尺寸（32 的倍数向下取整，至少 1） |
| `io.h` | `config_tile/acc/vec`、`set_zone`、`load_*_at`、`store_*_at`、`init_*` | 描述符封装。**外提约定**见文件头：stride/dtype 提到最外、zone 提到形状不变的那层、内层只写 base |
| `config.h` | 常量（`kBlock=32`、`kInv127`、`kEps`…）与 dtype 枚举 | |
| `qwen3.h` | 权重视图集合 `ModelViews` | 把 blob 里的偏移变成 `MatrixView` |
| `attention.h` / `mlp.h` / `rope.h` | 单头/单块版本 | 小配置（hidden 64/128、head_dim 32）用 |

## 数值契约（换模型/换量化方案前必读）

- **权重**：int8，per-output-channel f32 scale（`b_scale`）；**激活**：int8，per-token f32 scale（`a_scale`）。
- **累加**：整个 K 方向在 i32 里累加，只在 epilogue 转 f32 并乘两次 scale —— 这样与 host 的
  f32 参考**逐位一致**；中间转 f32 会多一次舍入。
- **softmax**：`exp2` + `log2(e)` 折叠，`1/den` 留到最后才除。
- 任何"为了快而重排 f32 运算顺序"的改动都会破坏逐位一致；int8 量化会把 1 ulp 放大。

## 寄存器与形状约束

| 约束 | 值 | 超了会怎样 |
|---|---|---|
| 同时活跃 tile / acc / vec32 / vec8 | 16 / 16 / 32 / 32 | 硬报错 `ran out of registers during register allocation` |
| tile 宽度 | 32 列 | 所有算子按 32 列分块（`kBlock`） |
| 一次处理的行数 | ≤ 32 | driver 按 32 分块 |
| head_dim | 32 的倍数（0.6B = 128） | block attention 写死 4 个输出块（128/32） |
| 不能 spill | 无 load/store 编码 | 同上硬报错 |

## 依赖

| 依赖 | 说明 |
|---|---|
| **clang（含 `+xaicore`）** | 我们改过的 LLVM fork：`-march=rv64im_xaicore`，114 个 `__builtin_yai_*` |
| `riscv64-unknown-elf-{as,ld,objcopy}` | 汇编/链接/取二进制 |
| **Tile-NPU 模拟器** | 实现 93 条 YAI 指令 + 4 个寄存器文件 + 描述符；`sim_bridge` 负责加载/导出 |
| **blob** | 0.6B 的 int8 权重 + 激活区 + KV cache，约 780 MB（`--max-seq 256`） |

## 构建 / 运行 / 验证

```bash
# 1) 构建 guest（默认 CLANG=$HOME/llvm-project-xaicore/build/bin/clang++）
make
#    clang 在别处就显式指定（路径可带空格）：
make CLANG="$HOME/llvm-project-xaicore/build/bin/clang++"

# 2) 跑一次（blob 里的 CTRL = [n_prompt, n_gen, max_seq, mode] 决定行为）
make run BLOB=/path/to/qwen3_big_data.bin BRIDGE=/path/to/sim_bridge

# 3) 与参考 logits 比对（逐位 + max|Δ| + argmax）
make check BLOB=/path/to/blob.bin BRIDGE=/path/to/sim_bridge REF=/path/to/ref_logits.bin
```

`config/model_big.h` 里的偏移由上层工具（`convert_pqwf.py`）按权重/序列长度生成；
换 `--max-seq` 或换量化方案时必须重新生成它，否则 blob 布局对不上。

## 实测（Qwen3-0.6B，int8，`--max-seq 256`）

| 场景 | 指令数 | 与 host f32 参考 |
|---|---|---|
| prefill 32（默认） | **2,319,041** | **逐位相同**（max\|Δ\| = 0） |
| prefill 64 / 96 | 3,966,000 / 6,235,473 | 逐位相同 |
| prefill 255（8 个 KV 块） | 17,052,696 | 逐位相同 |
| prompt 5 + 生成 4 | 12,253,111 | 生成 token 一致 |

指令数的演进（同一个 kernel，只改描述符/分块/自增）：12,394,777 → 6,471,417 → 3,724,836
→ 3,672,509 → **2,319,041（-81%）**，logits 全程逐位不变。

## 与其它仓库的关系

```
qwen3-tile-ops（本仓库）        算子 + 前向 driver
        ↑ 用 clang 编译
llvm-project (xaicore 分支)     114 个 __builtin_yai_* / 93 条指令 / 4 个寄存器类
        ↑ 执行
Tile-NPU-Sim                    指令语义 + 描述符状态 + sim_bridge
        ↑ 权重与输入
上层推理项目                    tokenizer、convert_pqwf.py、chat.py、参考实现
```
