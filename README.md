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
make check     # 跑到模拟器上，逐 stage 与 numpy 参考对比
make small     # 1 层 / 1024 词表的冒烟测试（~30 MB blob，10 秒）
make value-ops # Acc/Vec32 值形式 vs 显式拷贝的逐位对比（不需要模型）
make chat TEXT="你好，请用一句话介绍你自己。" MAX_NEW=8
```

`make data` 的参数（`CONVERT_ARGS`，见 `tools/convert_pqwf.py`）：`--layers N`
（只保留前 N 层）、`--vocab V`（截断词表）、`--n-prompt/--n-gen/--ids`。

## 结果（本机实测）

设备输出与 `tools/ref_qwen3.py`（逐指令对齐 tile 语义的 numpy 镜像）对比：

| 场景 | 结果 |
|---|---|
| 1 层 / 1024 词表，40 token prompt + 生成 2 | 全部 stage 逐位相同，生成 id 一致（覆盖 2 个 prefill chunk + decode rows=1） |
| **28 层 / 151936 词表，15 token prefill** | 全部 stage 逐位相同，logits max\|Δ\|=0 |
| **28 层 / 151936 词表，32 token prefill** | 全部 stage 逐位相同，logits（151936 个）max\|Δ\|=0 |
| **28 层，15 token prompt + 贪心生成 6** | 全部 stage 逐位相同；生成 id `151667 198 99692 3837 3837 35946` 与参考一致 |

与“真正的模型”对比（`tools/fp32_ref.py`，无 int8、无分块、无 online softmax）：

* 同一 prompt 的 top-1 完全相同（`151667` = `<think>`），logits Pearson 0.93；
* 贪心前 4 个 token 与 fp32 一致，之后因 w8a8 量化在近似并列处翻转——迁移前
  CPU 参考的结论相同；
* 设备上 6 个 token 的文本 `<think>\n好的，，我…` 与仓库迁移前记录的验证结果
  逐 token 一致。

运行时：32 token prefill 2,369,787 条指令（模拟器约 10 s，不含 791 MB dump）；
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
| `Acc/Vec32 t = x.copy()` | `cfg.copy` + `acopy`/`vcopy` | 是（语义就是要副本） |

`make value-ops` 是这套语义的回归：14 个 case 各自算“值形式”和“copy+原地”，
逐位相同且与 numpy 一致（不需要模型，1 秒跑完）。attention 里原来的
`corr.copy(maximum); corr.sub_(nmm)` 已改成 `Vec32 corr = maximum.sub(nmm)`，
prefill 32 的指令数因此 2,370,235 → **2,369,787**（每 KV 块省 1 条 `vcopy`）。

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

* 只做贪心（设备端 `argmax_logits`）；采样需要 host 侧单步循环（`CTRL[3]=1`）。
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
