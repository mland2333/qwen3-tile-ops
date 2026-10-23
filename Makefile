# Qwen3-0.6B 的 tile 算子库（算子 + 一个把算子串成完整前向/生成的 driver）
#
# 需要外部提供（不在本仓库）：
#   CLANG   支持 -march=…+xaicore 的 clang（我们改过的 LLVM fork）
#   CROSS   riscv64-unknown-elf binutils（as/ld/objcopy）
#   BRIDGE  Tile-NPU 模拟器的 sim_bridge（只有 `make run` 需要）
#   BLOB    0.6B 的 int8 权重 + 输入 blob（约 780 MB）
#
# 用法：
#   make                     # 编译 guest → build/qwen3_ops.bin
#   make run  BLOB=/path/x.bin
#   make check BLOB=… REF=/path/logits.bin
#
# 覆盖变量示例：
#   make CLANG=$HOME/llvm-project-xaicore/build/bin/clang++

CLANG   ?= $(HOME)/llvm-project-xaicore/build/bin/clang++
CROSS   ?= riscv64-unknown-elf-
BRIDGE  ?= $(HOME)/npucc/build/sim_bridge
BLOB    ?= weights/qwen3_big_data.bin
REF     ?= /tmp/ref32.bin

# x31 = blob 基址、x30 = 栈顶，由 crt0/sim_bridge 约定；编译器不得挪用。
TARGET  := --target=riscv64-unknown-elf -march=rv64im_xaicore -mabi=lp64 \
           -ffixed-x30 -ffixed-x31 -std=c++20 -O1 -nostdinc++
INC     := -isystem test_include -I include -I kernels -I config
BUILD   := build
HDRS    := include/tile.h config/model_big.h $(wildcard kernels/*.h)

.PHONY: all run check clean size

all: $(BUILD)/qwen3_ops.bin

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/main_big.o: src/main_big.cpp $(HDRS) | $(BUILD)
	"$(CLANG)" $(TARGET) $(INC) -c $< -o $@

$(BUILD)/crt0.o: src/crt0.S | $(BUILD)
	$(CROSS)as -mabi=lp64 $< -o $@

$(BUILD)/qwen3_ops.elf: $(BUILD)/main_big.o $(BUILD)/crt0.o
	$(CROSS)ld -T link.ld $^ -o $@

$(BUILD)/qwen3_ops.bin: $(BUILD)/qwen3_ops.elf
	$(CROSS)objcopy -O binary $< $@
	@printf "guest: %s  (%s 字节)\n" $@ "$$(wc -c < $@ | tr -d ' ')"

# 跑一次 prefill（blob 里的 CTRL 决定 n_prompt/n_gen/max_seq/mode）
run: $(BUILD)/qwen3_ops.bin
	"$(BRIDGE)" $< $(BLOB) $(BUILD)/out.bin 0 600000000 64

# 与参考 logits 比：先看是否逐位相同，再看 max|Δ|
check: $(BUILD)/qwen3_ops.bin
	"$(BRIDGE)" $< $(BLOB) $(BUILD)/out.bin 0 600000000 64
	python3 tools/check_logits.py --header config/model_big.h \
	                              --out $(BUILD)/out.bin --ref "$(REF)"

size: $(BUILD)/qwen3_ops.bin
	$(CROSS)size $(BUILD)/qwen3_ops.elf

clean:
	rm -rf $(BUILD)
