# Qwen3-0.6B on the Tile NPU (YAI XAIcore)

把 Qwen3-0.6B（hidden=1024、28 层、16 Q 头 / 8 KV 头、head_dim=128、inter=3072、
vocab=151936）的整层前向写成 YAI tile 算子，用带 `xaicore` 扩展的 clang 编译成
RISC-V + NPU 机器码，在 [Tile-NPU-Sim](../Tile-NPU-Sim) 上做自回归贪心推理。

```
kernels/*.h  ──clang(xaicore)──►  qwen3_full.bin (rv64im + NPU)
                                      │
tools/convert_pqwf.py ──► blob ───────┤  sim_bridge
   (int8 权重量化 + 布局)             ▼
                              Tile-NPU-Sim ──► logits / token ids
                                      │
tools/ref_qwen3.py (numpy f32 逐指令镜像) ──► 逐 stage 对比
```

## 目录

```
include/tile.h        YAI tile C++ API（Tile/Acc/Vec32 的 builtin 封装）
kernels/              算子库：matrix/io/config 基础设施 + gemm/quant/rmsnorm/
                      embedding/rope/attention_gqa/mlp
src/main_big.cpp      模型布局、forward_chunk()、贪心生成循环
src/crt0.S            启动代码（sp、call main、kernel.end）
link.ld               0x80000000 起的链接脚本
config/model_big.h    默认（28 层 / 全词表）布局常量，由 tools 生成
test_include/         最小 C++ 头
tools/
  pqwf.py             布局（与 model_big.h 逐字节一致）、safetensors 流式读取、int8 量化
  convert_pqwf.py     PQWF → blob + model_big.h + numpy 参考（ref.npz）
  ref_qwen3.py        Qwen3 前向的 numpy f32 镜像（逐条指令对齐 tile 语义）
  fp32_ref.py         独立的 f32 参考（无 int8、无分块，用来校验模型本身）
  compare.py          设备 dump 与 ref.npz 的逐 stage 对比（含生成 id）
  sim_bridge.cpp      模拟器 harness：装 guest + blob、x31=blob 基址、跑完 dump 回写
  tokenizer.py        Qwen3 byte-level BPE（自包含）
  patch_blob.py       往 blob 里写 prompt ids / CTRL
  run_chat.py         文本进/文本出
```

## 环境搭建（Ubuntu，无 root）

服务器上已有 `riscv64-unknown-elf-` 工具链（`/nfs/home/share/riscv/bin`）和
xmake（`~/.local/bin/xmake`）。需要自己编的只有 LLVM/Clang 和模拟器。

```bash
# 1) 带 XAIcore 扩展的 clang（约 20 分钟；ccache 要放在可写目录）
cd llvm-project
cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=clang -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_ZSTD=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache
CCACHE_DIR=$PWD/../.ccache ninja -C build -j 48 clang llc llvm-objdump

# 2) 模拟器
export PATH=$HOME/.local/bin:$PATH
export XMAKE_GLOBALDIR=$HOME/npu/.xmake-home     # ~/.xmake 只读时用仓库内目录
cd Tile-NPU-Sim
xmake f -y --riscv_cross=riscv64-unknown-elf-
xmake -y -j 32
xmake build sim_tests && ./build/linux/x86_64/release/sim_tests   # 219/219
python3 tests/run.py guest --build                                # 18/18
```

已做的环境修复（否则在 Ubuntu 上跑不起来）：

* `xmake` 的 `~/.xmake` 在只读挂载上 → 用 `XMAKE_GLOBALDIR` 指到仓库内；
* 旧的 `riscv64-unknown-elf-ld` 不认识 clang 21 生成的 `zmmul`/`xaicore`
  属性 → 链接前 `objcopy --remove-section=.riscv.attributes`（Makefile 自动做）；
* guest 链接地址 `0x80000000` 超出 medlow 范围 → 所有 guest 构建加
  `-mcmodel=medany`（已写进 Tile-NPU-Sim 的 `xmake/modules/guestimg.lua` 等）；
* 原来的 `sim_bridge` 没有随仓库迁移过来 → `tools/sim_bridge.cpp` 重建：镜像装
  `0x80000000`，blob 装 `--blob-base`（默认 `0x81000000`）并写 `x31`，
  `kernel.end`/`ebreak` 结束，`--dump FILE@OFF:SIZE` 回写。

## 模型与权重

```bash
mkdir -p ../assets/qwen3-0.6b && cd ../assets/qwen3-0.6b
for f in config.json model.safetensors tokenizer.json generation_config.json; do
  curl -sL -O "https://hf-mirror.com/Qwen/Qwen3-0.6B/resolve/main/$f"
done
```

## 构建与验证

```bash
make data      # 全量：build/full/{model_big.h, qwen3_big_data.bin, ref.npz}
make big       # 编译 + 链接 build/qwen3_full.bin
make check     # 逐 stage 对拍：mode 0 融合 + mode 1 非融合各跑一遍
make small     # 1 层 / 1024 词表的冒烟测试（~30 MB blob，10 秒）
make value-ops # Acc/Vec32 值形式 vs 显式拷贝的逐位对比（不需要模型）
make group-gemm # per-128-group GEMM 原语测试（不需要模型）
make chat TEXT="你好，请用一句话介绍你自己。" MAX_NEW=8
```

`make data` 的参数（`CONVERT_ARGS`，见 `tools/convert_pqwf.py`）：`--layers N`
（只保留前 N 层）、`--vocab V`（截断词表）、`--n-prompt/--n-gen/--ids`。

## 结果（本机实测）

> **当前状态（全 per-group 在线量化）**：Q/K/V 仍按 head=128 量化；h8、
> attention 输出/o8、SwiGLU/s8、final norm/lm_head 已改成 256 列一组，
> GEMM 全部走 per-group K 累加。`make check` 两种模式都是 **0 stage(s)
> differ**，fused best=**12.6227** @400；`make chat` 6 token 为
> `151667 198 99692 3837 20002 104029`，与 `fp32_ref.py` 的 greedy 6 token
> 完全一致。下面的表格是 per-token 版本的历史记录。32 token prefill 指令数
> 在 256 组下约 **6.17M**（128 组时 6.98M，per-token 2.4M）；O/down 的
> residual 已融进 per-group GEMM epilogue，attention 输出也在
> `attention_head_hd128_blocks` 尾部直接按 head 量化，省掉 f32 attn 的
> write+read；Q/K 的 norm/rope/per-head quant 也融成一个 kernel，q/k f32
> 只读一次、norm/rope 不再写回。32 token prefill 约 **6.05M**。下一步是
> V 的 GEMM 输出融合和 gate/up 的 SwiGLU/quant epilogue。

设备输出与 `tools/ref_qwen3.py`（逐指令对齐 tile 语义的 numpy 镜像）对比：

| 场景 | 结果 |
|---|---|
| 1 层 / 1024 词表，40 token prompt + 生成 2 | 全部 stage 逐位相同，生成 id 一致（覆盖 2 个 prefill chunk + decode rows=1） |
| **28 层 / 151936 词表，15 token prefill** | 全部 stage 逐位相同，logits max\|Δ\|=0 |
| **28 层 / 151936 词表，32 token prefill** | 全部 stage 逐位相同，logits（151936 个）max\|Δ\|=0 |
| **28 层，15 token prompt + 贪心生成 6** | 全部 stage 逐位相同；生成 id `151667 198 99692 3837 3837 35946` 与参考一致 |

> 现在生产路径（mode 0）的 lm_head 与 argmax 已融合，不再写 logits；
> mode 1 是非融合参考路径，仍然写完整 logits。`make check` 会对两种模式各跑
> 一遍：mode 0 比较 `argf`/`argi` 与 `ref_logits` 的 max/argmax，mode 1 比较
> 完整 logits 和生成 id。最后一层的 Q/attention/O/MLP 在 mode 0 里只算需要
> logits 的行，所以 `--fused` 对这些 stage 只比较 `[last row]`；K/V 和
> KV cache 仍然比较所有行。

与“真正的模型”对比（`tools/fp32_ref.py`，无 int8、无分块、无 online softmax）：

* 同一 prompt 的 top-1 完全相同（`151667` = `<think>`），logits Pearson 0.93；
* 贪心前 4 个 token 与 fp32 一致，之后因 w8a8 量化在近似并列处翻转——迁移前
  CPU 参考的结论相同；
* 设备上 6 个 token 的文本 `<think>\n好的，，我…` 与仓库迁移前记录的验证结果
  逐 token 一致。

运行时：32 token prefill，融合 mode 0 **2,454,977** 条指令，非融合 mode 1
**2,364,735**（prefill-only 下 mode 1 不跑 argmax）。最后一层的
Q/attention/O/MLP 只对需要 logits 的行算；非最终 prefill chunk 整段 tail 跳过
（64 token prefill：4,438,896 → 4,392,666）。tile 指令是固定 32 行块，所以
最终 chunk 的 1 行 tail 在指令数上几乎不降，收益在有效 MAC/访存；lm_head
本来就只有 1 行。生成阶段 mode 0 不再写/读 logits，也不再跑 `argmax_logits`
的 4,748 块标量往返。模拟器约 10 s，不含 791 MB dump；
`build/qwen3_full.bin` 约 168 KB；`make data` 约 40 s（含参考前向）。

## blob 布局（`tools/pqwf.py` 生成，与 `config/model_big.h` 一致）

```
kIDS   [max_seq] i32          kCTRL [n_prompt,n_gen,max_seq,mode] i32
kX     [seq][hidden] f32      （残差流）
kEMB   [vocab][hidden] i8 + [vocab] f32 scale
kLM    [vocab][hidden] i8 + [vocab] f32 scale
kNF    [hidden] f32           kCOS/kSIN [max_seq][head_dim/2] f32
kLQ..  每层 [out][in] i8 + [out] f32（q/k/v/o/gate/up/down），层间 stride 固定
kN1    每层 n1[hidden] + nq[hd] + nk[hd] + n2[hidden]（stride 9216）
kLOGITS [seq][vocab] f32
kH..   激活/scale/量化缓冲 + 每层一份 KV cache（max_seq×kv_width i8 + max_seq f32）
```

约定：

* `x31` = blob 基址、`x30` = 栈顶（`crt0.S` 设 sp），编译器用 `-ffixed-x30/31`；
* 权重是 `[out][in]` 行主序 int8，scale 每输出通道一个 f32；
* 激活是 per-token int8（`absmax/127`，RNE 饱和到 ±127，NaN→0）；
* tile/acc/vec 值不跨算子边界（算子全部 inline），跨算子只传 `MatrixView`；
* KV cache 是**每层一份**（`kKV_BASE + layer*kKV_STRIDE`）；
* `q8/qs/o` 的第 0 行在算最后一行 logits 时被复用为 final-norm 的输出，
  `compare.py` 对这三处做了别名说明。

## 值传递 API（D ≠ S，不搬数据）

`tile.h` 里带尾下划线的是原地版本，不带下划线的是**派生值**版本。派生值版本现在
走 ISA 的 D≠S 形态：后端在 post-RA 只补一条 `cfg.copy`（克隆 base/stride/zone/
dtype），**不发 `acopy`/`vcopy`**：

| 表达式 | 生成 | 数据搬运 |
|---|---|---|
| `Acc t = x.mul(k)` / `x.add(k)` | `cfg.copy` + `amulx`/`aaddx` | 无 |
| `Acc t = x.add(y)`（含 `x + y`） | `cfg.copy` + `aadd` | 无 |
| `Acc t = x.sub(y)` / `x.mul(y)`（含 `-`、`*`） | `cfg.copy` + `asub`/`amul` | 无 |
| `Acc t = x.mul(v, byrow/bycol)` / `x.sub(v, ...)` | `cfg.copy` + `amulv`/`asubv` | 无 |
| `Acc t = x.exp()` | `cfg.copy` + `amulx` + `aexp2` | 无 |
| `Vec32 t = u.add(v)` / `sub` / `mul` | `cfg.copy` + `vadd`/`vsub`/`vmul` | 无 |
| `Vec32 t = u.exp()` | `cfg.copy` + `vmulx` + `vexp2` | 无 |
| `Vec32 t = u.mul(k)` | `cfg.copy` + `vmulx` | 无 |
| `Acc t = x.rcp()` | `cfg.copy` + `arcp.approx` | 无 |
| `Vec32 t = u.rcp()` / `rsqrt()` | `cfg.copy` + `vrcp.approx`/`vrsqrt.approx` | 无 |
| `Vec32 t = u.max(v)` | `cfg.copy` + `vmax` | 无 |
| `t.dequant(scale, out)` | `tdequant.rows.f32 out, t, scale` | 无（写已有 Acc） |
| `Acc/Vec32 t = x.copy()` | `cfg.copy` + `acopy`/`vcopy` | 是（语义就是要副本） |

`make value-ops` 是这套语义的回归：19 个 case 各自算“值形式”和参考路径，
逐位相同且与 numpy 一致（不需要模型，1 秒跑完）。其中 `tile dequant` 的参考
路径是 `tdequant(ones) + mul(byrow)`（新 ISA 已删除 acvt/twiden，原来的
`widen + to_fp32 + mul` 不再可编码），`rsqrt` 用 `allclose(equal_nan=True)`。attention 里原来的
`corr.copy(maximum); corr.sub_(nmm)` 已改成 `Vec32 corr = maximum.sub(nmm)`，
prefill 32 的指令数因此 2,370,235 → **2,369,787**（每 KV 块省 1 条 `vcopy`）。

### 新 ISA（CVT 已删除）下的转换写法

当前模拟器分支 `port-new-sim` 删除了 `acvt`/`vcvt`/`twiden`（`major4 = 0011`
整体保留），所以 kernel 里所有 `to_fp32_()`/`widen()` 都必须换掉：

| 旧写法 | 新写法 | 说明 |
| --- | --- | --- |
| `t.widen(a); a.to_fp32_(); a.mul_(s, byrow)` | `t.dequant(s, a)` | `tdequant.rows.f32`，i8×scale→f32 一条指令 |
| `a.to_fp32_(); a.mul_(as, byrow); a.mul_(ws, bycol)` | `adequant(f, a, ws, as)` | 一条指令 = `f32(a)·ws[j]·as[i]`；gemm 的 epilogue 与 attention 的 QK scale 都是这个形态 |
| `a.to_fp32_(); a.mul_(s, byrow)` | `adequant(f, a, ones, s)` | `f = f32(a)·1·s`；乘 1.0f 精确（O/PV 的 epilogue：scale 与分母/ps 分开乘） |

`ones` 由 `kernels/io.h::config_ones` 生成，`adequant` 声明在 `include/tile.h`。
gemm 快路径把 8 个 epilogue 共用的 f32 目的 acc 提到组外配置，每组只发 8 条
`adequant`。

**注意乘序**：`adequant` 是 `(x · vCol) · vRow`，而旧写法是 `(x · byrow) · bycol`。
合并后每个元素差 1 ulp 级，经 int8 量化的边界会被放大：`ref_qwen3.py` 已按新乘序
同步（`make check` 与新的 numpy 参考**逐位相同**），贪心生成的 token 与迁移前
记录的一致，但 logits 绝对值会移动 0.1 量级（median |Δ|≈0.13）。

## 本轮修复的编译器编码错误

ISA 更新（`mma.nn` 删除、除法删除、Vec8 删除、Acc 减到 16、argmax 双目的地、
gather 索引 u32 等）之后，LLVM 编译器（`RISCVInstrInfoXAIcore.td`）里有两处
编码没跟上，是“算子重写后推理不对”的直接原因：

1. **`aquantx` / `aquant` 的基址与字段错位**（两个 def 写反）：
   `funct3` 区分动态/外部 scale，`vScale5` 在 `[21:17]`、保留的 `index5` 在
   `[16:12]`。原来的 `0x40000002` / `0x40400002` 让外部 scale 形态带上
   `index5=5`，模拟器直接 `illegal_instruction`。
   修正：`aquantx = 0x40400002`、`aquant = 0x40000002`。
2. **`amask.tril` 字段交换 + 选错 fill 来源**：`xDelta` 应在 `[21:17]`、
   `fill5` 应在 `[16:12]`，立即数 fill 必须走 `FS=0`（原来的 `0x46400000` 是
   `amaskx` 的标量 fill 形态）。于是因果 mask 实际用 `xDelta=8`、被 mask 的
   位置填入某个标量寄存器 —— 表现为“只有最后一行 logits 对”。
   修正：`amask.tril = 0x44400000` + 字段对调。

修复后 1 层与 28 层模型的每个 stage 都与参考逐位相同。编译器其余部分
（mma.nt、reduce 双目的地 argmax、广播、gather/widen、描述符自增、cfg 类型码、
post-RA 的原地/派生形态展开）都在这套端到端测试里被覆盖。

详细说明见 `../llvm-project/XAICORE-CHANGES.md` §11。

## 已知限制

* 只做贪心（mode 0 为 `gemm_nt_argmax` 融合，mode 1 为非融合参考）；采样需要另加 host 侧单步接口。
* `max_seq` 默认 256（blob 791 MB，其中 embedding + lm_head 占 312 MB）；
  `--max-seq` 改大只需重新生成 blob。
* 与 fp32 相比是 w8a8 量化模型：top-1 一致、logits 有 ~1 量级漂移，长序列贪心
  会在近似并列处翻转。
* `attention_head_hd128_blocks` 里 head_dim 固定 128（0.6B）；换模型要改这一处。
  单块版本（`attention.h`、`attention_head_hd`）已适配 `mma.nt`，但只做过编译
  检查，没有对应的测试模型。
* 模拟器是单 hart 功能模型：32 token prefill 约 10 s；每个 decode token 的
  动态指令数和 32-token prefill 同量级（mma 条数只取决于 K/N 分块，不取决于
  M 行数）。
