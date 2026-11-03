# tiny Qwen3 推理项目（YAICore）

用 `tile.h` 的成员函数 API 写 Qwen3（tiny: hidden=64、2 头 × hd=32、1 层）的整层
前向，编译成 YAICore 机器码，在 Tile-NPU-Sim 上跑出 logits 并与 Python 参考对比。

## 目录

```
include/tile.h      YAI tile API（从编译器目录复制）
kernels/            每个算子一个头文件（定义都在头里，YAI_ALWAYS_INLINE）
  config.h          数值常量与 dtype 码
  matrix.h          MatrixView（地址/类型/跨步/形状）+ 子视图/块裁剪
  io.h              MatrixView 的 32x32 块 <-> Acc/Tile/Vec32
  rmsnorm.h         rmsnorm_rows（行 RMSNorm）
  quant.h           quant_rows（per-token absmax + tquantx）
  gemm.h            gemm_nt（K 块累加 + 双 scale）
  rope.h            rope_head（rotate-half）
  attention.h       attention_head / attention_heads
  mlp.h             swiglu / add_rows
  qwen3.h           汇总头
src/main.cpp        模型布局常量 + main 编排
src/crt0.S          启动代码（sp、call main、kernel.end）
test_include/       最小 C++ 头（cstdint/cstddef/array）
tools/              权重生成、参考前向、PQWF、模拟器对比
```

## 构建与运行

```bash
make                 # 编译 build/qwen3_guest.bin
make weights         # 生成 PQWF 权重 + 模拟器 blob
make check           # 在 Tile-NPU-Sim 上跑并与 Python 参考对比 logits
make clean
```

可覆盖变量：`CLANG=../build/bin/clang++`、`CROSS=riscv64-unknown-elf-`、
`LINK_LD=$HOME/Tile-NPU-Sim/tests/guest/link.ld`，运行时用 `NPUCC` /
`TILE_NPU_SIM` 环境变量指定 sim_bridge 与模拟器位置。

## 算子接口：MatrixView

算子之间传的是**大矩阵描述符** `MatrixView{data, row_bytes, col_bytes,
rows, cols, dtype}`（纯标量/指针），tile 拆分在算子内部完成：

```cpp
struct MatrixView { std::byte *data; std::int32_t row_bytes, col_bytes;
                    std::uint32_t rows, cols; std::uint8_t dtype; };
```

`main.cpp` 只构造视图并编排：

```cpp
rmsnorm_rows(x, n1, h);          // 内部按 32 列块两遍（sumsq 合并 + 缩放）
quant_rows(h, h8, hs);           // per-token absmax -> tquantx
gemm_nt(h8, wq, wqs, hs, q);     // 内部按 32x32 宏块、K 用 mma.nt.acc 累加
attention_heads(q8, k8, v8, qs, ks, vs, attn, 2, 32, true);
```

## XAIcore 寄存器在调用约定中的位置（约定）

- 后端把 XTT/XTA/XTV/XTB 加进了 **call-preserved mask**：调用点不再把它们标成
  clobbered。语义上它们是“私有寄存器文件”，普通函数（libgcc 的软浮点辅助
  函数、scalar helper）不使用也不保存它们。
- 契约：**所有使用 tile/acc/vec 的函数必须内联**；外部函数（libgcc、scalar
  helper）不得使用这些寄存器。违反契约（有非内联函数用 NPU 寄存器）会让
  调用者的 NPU 值被静默破坏。
- 因此 soft-float 调用（`__divsf3`、`__floatunsisf`）可以与 NPU 值的 live
  range 共存；不再强制“标量调用必须提到 tile 值之前”。
- 注意：trap/中断/上下文切换不会自动保存 NPU 寄存器；裸机模拟器没有抢占，
  真实系统需要在 trap handler 里保存。

## 数据寄存器与算子边界（约定）

1. **tile/acc/vec 值不跨算子存活**：算子之间只传 `MatrixView` + 内存，
   数据寄存器只在算子内部出现。这是本库的设计前提。
2. 满足 1 时，算子**不一定要内联**：实测 `noinline` 的模板算子
   （维度是模板参数、内部块循环是常量）可以正常编译，调用点没有存活的
   tile 值，普通 C ABI 调用合法。
3. 但算子**内部**的块循环必须是编译期常量（内联或模板参数都行），
   否则 tile 值会跨运行期循环回边，RA 因 no-spill 报错
   （实测：`cols` 为运行期的 `rmsnorm_rows`，即使在 `main` 里也编译失败）。
4. 签名里带 `Tile`/`Acc`/`Vec32`/`Vec8` 的函数必须 `YAI_ALWAYS_INLINE`
   且定义在头文件：这类值不能跨调用边界（ABI 落内存 → `store (s32768)`）。
5. 运行期**外层**循环（如 28 层）可以，只要没有 tile 值跨层回边
   （实测通过）；这正好匹配 0.6B 的层循环。
6. 当前库为了简单统一标了 `YAI_ALWAYS_INLINE`：内联后维度变成常量、块循环
   展开，满足 3；如果以后要把算子改成真正的独立函数，需要把维度做成模板
   参数（或生成 constexpr 配置）来保证展开。
- 权重是 `[out][in]` int8 + 每通道 f32 scale；激活 per-token int8
  （`tquantx`），GEMM 的 epilogue 乘双向 scale。
- 自定义权重格式是 PQWF（`tools/tiny_qwen3_mh.py` 的 `write_pqwf_mh`）。

## 数据布局

见 `src/main.cpp` 顶部：x/权重/norm/rope 表在 blob 的低地址（与
`tools/tiny_qwen3_mh.py::build_blob_mh` 一致），激活/中间缓冲从 `0x40000`
开始（h/h8/hs、q/k/v/q8/k8/v8/qs/ks/vs、attn、o/o8/os、g/u/sw/s8/ss）；
logits 仍写回 `LOGITS_OFF=24576` 供 checker 读取。

## 结果

- `make check`：1451→2774 条指令（描述符版把中间结果走内存，换来接口简洁/
  可复用），logits 与之前的寄存器融合版**逐位一致**（max|Δ|=0）。

## 真实 Qwen3-0.6B：自回归推理（`make big`）

`src/main_big.cpp` 是完整 0.6B（hidden=1024、28 层、16Q/8KV、hd=128、inter=3072、
vocab=151936）的 **自回归** 前向：

```
prompt（任意长度，按 32 token 分块 prefill）
  → 每层 k/v per-token 量化后写进 KV cache（int8 + f32 scale，28 层各一份）
  → attention 用 online-softmax 按 32 token 分块读 KV（支持任意长度 / decode rows=1）
  → 取最后一行 logits → argmax（贪心）→ 写回 ids[pos] → 下一步拿它当输入
```

与 host 参考 `tile_tests/build/qwen3_tile` 的数值关系（实测）：

| 场景 | 结果 |
|---|---|
| prefill 32 / 64 / 80 / 96 token | logits **逐位相同**（max\|Δ\|=0） |
| 生成（贪心）短 prompt 5+4、40+4、32+4 | 生成的 token **完全一致** |

```bash
make big-data                      # 默认：ids=100+200×31、max_seq=64、n_gen=0（只 prefill）
make big
../../npucc/build/sim_bridge build/qwen3_big.bin build/qwen3_big_data.bin /tmp/out.bin 0 3000000000 64

# 生成 4 个 token：把 n_gen 写进 blob 的控制块
python3 tools/convert_pqwf.py --max-seq 64 --n-gen 4 \
        --ids 100,200,300,400,500 --header build/model_big.h --blob build/qwen3_big_data.bin
```

控制块 `CTRL = [n_prompt, n_gen, max_seq, 0]`，生成的 id 依次写回 `kIDS`（host 侧
读输出 blob 即可）。命令行的封装见 `tools/chat.py --engine guest`。

编译注意：Makefile 必须带 `-ffixed-x30 -ffixed-x31`（blob 基址/栈顶约定，
编译器不得挪用），详见 `tools/known_issues.md`。

## 工具

| 工具 | 用途 |
|---|---|
| `tools/convert_pqwf.py` | PQWF → guest blob + `model_big.h`；`--seq`（块大小）/`--max-seq`（KV 容量）/`--n-gen`（生成数）/`--ids` |
| `tools/gen_rope_table.cpp` | RoPE cos/sin 表，用同一 libm 现算（与 host 参考逐位一致） |
| `tools/chat.py` | 文本进/文本出：`--engine host`（多 token 生成）/`--engine guest`（编译版）/`--self-test` |
| `tools/qwen3_tokenizer.py` | 自包含 Qwen3 byte-level BPE（只需要 tokenizer.json） |
| `tools/fp32_baseline.py` | numpy 的 fp32 基线（原版 bf16 权重 / int8 反量化 / bf16 激活），用于精度对比 |
| `tools/known_issues.md` | 后端已知问题（vcopy+LICM 崩溃、x31 保留、尾块 vec 长度） |

## 文本进 / 文本出（端到端）

唯一的外部资产是 **tokenizer.json**（Qwen3 词表）。本机访问不了 huggingface.co，
但 hf-mirror.com 可用：

```bash
mkdir -p tile_api_llvm_ds/qwen3_ops/assets
curl -sL -o tile_api_llvm_ds/qwen3_ops/assets/tokenizer.json \
  "https://hf-mirror.com/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"     # 11 MB
```

`tools/chat.py` 会自动搜索这个位置（也认 `--tokenizer`、`$QWEN3_TOKENIZER`、
`~/Tile-NPU-Sim/external/qwen3/tokenizer.json`）。实测（编译版 guest，15 token prompt
+ 6 token 生成，24 秒）：

```bash
$ python3 tile_api_llvm_ds/qwen3_ops/tools/chat.py "你好，请用一句话介绍你自己。" \
      --engine guest --max-new 6 --show-ids
prompt: 15 tokens  ('<|im_start|>user\n你好，请用一句话介绍你自己。<|im_end|>\n<|im_start|>assistant\n')
new ids   : [151667, 198, 99692, 3837, 3837, 35946]
--- 输出（6 tokens，编译版 guest 生成）---
<think>
好的，，我
```

同 prompt 用 host 引擎（`--engine host`）生成的 token **完全一致**，只是能跑更长：

```
$ python3 .../chat.py "你好，请用一句话介绍你自己。" --engine host --max-new 24
<think>
好的，，我是一个AI助手，，，，。我致力于提供帮助和解答问题，，，
```

两点说明：
* Qwen3 默认开 thinking，会先输出 `<think>…`；想要直接回答就在 prompt 末尾加
  `/no_think`（`chat.py "...。/no_think"`），实测输出 `</think>\n\n我是AI助手，<|im_end|>`。
* 编译版 guest 的序列上限由 blob 的 `--max-seq` 决定（默认 64 token，包含 prompt +
  生成的总长度）。加大它只增加 KV cache 的大小：每 +32 token ≈ 28×2×32×1024 ≈ 1.8 MB。

## 采样（temperature / top-k / top-p）

贪心是设备端做的（`acc_row_max` + `acc_row_argmax` + 整型 key 比较，见
`known_issues.md` 第 6 条）。需要采样时改成 **guest 单步 + host 采样**：

```
CTRL = [n_prompt, n_gen, max_seq, mode]
mode=0  内置生成循环（prefill 分块 + 贪心，一次跑完 n_gen 个 token）
mode=1  单步：只处理绝对位置 CTRL[0] 的 1 个 token，写 logits 后退出
```

`chat.py --temperature T` 会走单步模式：先 prefill 一次，然后每步把采样到的 token
写进 blob 的 `ids[pos]`、把 `CTRL` 设为 `[pos, 0, max_seq, 1]`，用**上一次的输出
blob** 当输入再跑（KV cache 在 data 区里随 blob 传递），最后统一 decode 成文本。

```bash
# 贪心（设备端，一次跑完）
python3 .../tools/chat.py "你好，请用一句话介绍你自己。" --engine guest --max-new 8
# 采样（host 侧，逐 token 跑）
python3 .../tools/chat.py "你好，请用一句话介绍你自己。" --engine guest --max-new 8 \
      --temperature 0.9 --top-p 0.9 --sample-top-k 50 --seed 7
```

采样器自检（对同一份 logits 采 2000 次）：T=1.0/top-k=50 → 9 个不同 id（top-1 占
98.8%）；T=1.5 → 12 个；T≤0 → 恒等于 argmax。注意在很自信的前缀上（`<think>` 之后）
前几个 token 几乎必然相同，多样性要在后面才显现。

## 序列长度

`--max-seq` 默认 **256**（prompt + 生成的总长度上限）。KV cache 大小 =
`28 层 × 2 × max_seq × 1028 B`：max_seq=256 → 14.7 MB（blob 791 MB，主要仍是
embedding + lm_head 的 311 MB 权重）。改大只需重新生成 blob，guest 代码不用改。

## 性能：六处优化（累计 -81%）

| 优化 | 做法 | 0.6B prefill 32 token |
|---|---|---|
| 描述符外提 | `config_*`（外层）+ `set_zone`（形状不变那层）+ `load_*_at`（只写 base） | 12,394,777 → 6,471,417（**-48%**） |
| GEMM n 方向寄存器分块 | 1 A tile + 8 B tile + 8 acc；一次 A 载入喂 8 个 mma | → 3,724,836（**-42%**） |
| embedding + rmsnorm 融合 | gather 时顺手累积 sumsq；第 0 层跳过重复的 norm | → 3,720,645 |
| GEMM epilogue 配置外提 | acc 的 strides 每组建一次、`b_scale` 的 dtype/len/stride 全程一次 | → 3,674,143 |
| attention 的 causal 块跳过 | 整块被 mask 的 KV 块直接不迭代（贡献恰为 0） | → 3,672,509 |
| **描述符自增** | GEMM 的 9 个 tile 用 `flags=2`（按列）、epilogue 的 `ws` 用 `flags=1`，k 循环里只发 `tload` | → **2,319,041（-36.8%）** |
| **合计** | | **-81.3%**，logits 全程逐位不变 |

### 描述符自增（ISA 早有、编译器以前没接）

```
tm/am.flags[1:0] = 0 关闭 / 1 按行(rows*row_stride) / 2 按列(cols*col_stride)
vm/bm.flags[0]   = 1 时每次访存后 base += len*stride
只在访存成功后提交；gather 永不提交。
```

接入前 GEMM 的 k 循环是 36 条/迭代（9 `tload` + 9 `cfg.setx` + 8 地址 `add` + 8 `mma` + 循环），
接入后是 **19 条/迭代**（9 `tload` + 8 `mma` + 循环）——每 (n,k) 三元组 4.5 → **2.4 条**。
因为自增的步长定义为"一个块的字节宽度"，它对任意 `col_stride/row_stride` 都成立。

**它是跨指令状态**，所以：每组开头必须重写 `base_addr`；兜底路径（尾块、K 不是 32 倍数）
用 `load_tile_at` 显式写 base，天然覆盖自增的效果。

### 参数扫描与失败尝试（都是实测）

| 配置 | 指令数 | 说明 |
|---|---|---|
| G=6 / **8** / 9 / 10 / 12（n 分块） | 3.87 / **3.67** / 3.86 / 3.75 / 3.84 M | **G=8 最优**（2 的幂） |
| acc + tile 都提到 n 循环外 | 3.79 M（G=12） | 更差：活区间太长，RA 没法复用 |
| tiles 提到函数级、acc 留组内 | ✓ 最优 | tile 每轮重载、acc 短命 |
| 展开 k 循环（`YAI_LOOP_UNROLL`） | **10.7 M（+65%）** | 触发上层连锁展开，禁止 |
| attention 也改自增（12 个常驻 tile） | 3,672,509→2,343,108 态 | **比 GEMM-only 差 1%**：寄存器压满，已回退 |

### 其他场景

| 场景 | 指令数 |
|---|---|
| prefill 32（默认） | **2,319,041** |
| prefill 64 / 96 | 3,966,000 / 6,235,473 |
| prefill 255（8 个 KV 块） | 17,052,696 |
| prompt 5 + 生成 4 | 12,253,111 |

attention 是算法意义上的 FlashAttention：KV 按 32 行分块、online softmax（running max /
denominator + `corr = exp(m_old-m_new)` 重缩放）、S/P 从不落全局内存、P 就地量化成 i8；
因果时整块被 mask 的 KV 块直接跳过（贡献恰为 0，逐位不变）。

tile 层的运行期循环（含嵌套、跨迭代值、tile PHI、`Vec32::copy`）都已验证可用，
边界是同时活跃 ≤16 tile / 16 acc / 32 vec32 / 32 vec8，超了是硬报错
（`ran out of registers`），详见 `tools/known_issues.md` §9。
