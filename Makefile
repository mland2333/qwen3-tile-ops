# 独立的 tiny Qwen3 推理项目。
#
# 依赖：本仓库的 YAI clang（默认 ../build/bin/clang++）、
# riscv64-unknown-elf 工具链、Tile-NPU-Sim 的 guest link.ld。
# 可用变量覆盖：CLANG / CROSS / LINK_LD / NPUCC / TILE_NPU_SIM。

CLANG     ?= ../build/bin/clang++
CROSS     ?= riscv64-unknown-elf-
LINK_LD   ?= $(HOME)/Tile-NPU-Sim/tests/guest/link.ld

# x31 = blob 基址、x30 = 栈顶：由 sim_bridge/crt0 约定传入，编译器不得挪用，
# 否则 main 里 asm("mv %0, x31") 可能读到被清零的 x31（96 token 时实测踩过）。
TARGET    := --target=riscv64-unknown-elf -march=rv64im_xaicore -mabi=lp64 \
             -ffixed-x30 -ffixed-x31 \
             -std=c++20 -O1 -nostdinc++ -isystem test_include
INC       := -I include
BUILD     := build
KERNELS   := $(wildcard kernels/*.h)

.PHONY: all weights check small check-small big big-data check-big clean

all: $(BUILD)/qwen3_guest.bin

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/main.o: src/main.cpp $(KERNELS) include/tile.h | $(BUILD)
	$(CLANG) $(TARGET) $(INC) -c $< -o $@

$(BUILD)/crt0.o: src/crt0.S | $(BUILD)
	$(CROSS)as -mabi=lp64 $< -o $@

$(BUILD)/qwen3_guest.elf: $(BUILD)/main.o $(BUILD)/crt0.o
	$(CROSS)ld -T $(LINK_LD) $^ -o $@

$(BUILD)/qwen3_guest.bin: $(BUILD)/qwen3_guest.elf
	$(CROSS)objcopy -O binary $< $@
	@echo "guest: $@"

weights: | $(BUILD)
	python3 tools/build_weights.py --pqwf $(BUILD)/tiny_qwen3_mh.pqw \
	                               --blob $(BUILD)/tiny_qwen3_mh.bin

check: all weights
	python3 tools/run_check.py --guest $(BUILD)/qwen3_guest.bin \
	                           --blob $(BUILD)/tiny_qwen3_mh.bin

SMALL_BIN  := $(BUILD)/qwen3_small.bin
SMALL_HDR  := $(BUILD)/model_small.h
SMALL_DATA := $(BUILD)/qwen3_small_data.bin

$(SMALL_HDR): tools/qwen3_small.py | $(BUILD)
	python3 tools/qwen3_small.py --header $@ --blob $(SMALL_DATA)

$(BUILD)/main_small.o: src/main_small.cpp $(KERNELS) $(SMALL_HDR) | $(BUILD)
	$(CLANG) $(TARGET) $(INC) -I $(BUILD) -c $< -o $@

$(SMALL_BIN): $(BUILD)/main_small.o $(BUILD)/crt0.o
	$(CROSS)ld -T $(LINK_LD) $^ -o $(BUILD)/qwen3_small.elf
	$(CROSS)objcopy -O binary $(BUILD)/qwen3_small.elf $@
	@echo "small guest: $@"

small: $(SMALL_BIN)

check-small: small $(SMALL_DATA)
	python3 tools/run_check_small.py --guest $(SMALL_BIN) --blob $(SMALL_DATA) --seed 1

BIG_BIN  := $(BUILD)/qwen3_big.bin
BIG_HDR  := $(BUILD)/model_big.h
BIG_DATA := $(BUILD)/qwen3_big_data.bin

$(BIG_HDR) $(BIG_DATA): tools/convert_pqwf.py | $(BUILD)
	python3 tools/convert_pqwf.py --header $(BIG_HDR) --blob $(BIG_DATA)

$(BUILD)/main_big.o: src/main_big.cpp $(KERNELS) $(BIG_HDR) | $(BUILD)
	$(CLANG) $(TARGET) $(INC) -I $(BUILD) -c $< -o $@

$(BIG_BIN): $(BUILD)/main_big.o $(BUILD)/crt0.o
	$(CROSS)ld -T $(LINK_LD) $^ -o $(BUILD)/qwen3_big.elf
	$(CROSS)objcopy -O binary $(BUILD)/qwen3_big.elf $@
	@echo "big guest: $@"

big: $(BIG_BIN)
big-data: $(BIG_HDR) $(BIG_DATA)
check-big: big big-data
	python3 tools/run_big.py --guest $(BIG_BIN) --blob $(BIG_DATA)

clean:
	rm -rf $(BUILD)
