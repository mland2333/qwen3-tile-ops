# Tile-NPU Qwen3 operator library: build the guest, the blob and the checks.
#
# Dependencies:
#   * clang++ with the XAIcore extension   (CLANG, default ../llvm-project/build/bin/clang++)
#   * riscv64-unknown-elf binutils         (CROSS)
#   * a built Tile-NPU-Sim                 (TILE_NPU_SIM, for libsim_core.a)
#
# Common flows:
#   make data            # full 0.6B: build/<variant>/model_big.h + blob + reference
#   make big             # compile+link the guest against the current header
#   make check           # run the guest on the simulator and diff every stage
#   make small           # 1 layer / 1024 vocab smoke test (~30 MB blob)
#   make chat TEXT="..." # text in / text out (greedy, on the device)
#
# Variables: CLANG, CROSS, LINK_LD, TILE_NPU_SIM, VARIANT, CONVERT_ARGS.

CLANG        ?= ../llvm-project/build/bin/clang++
CROSS        ?= riscv64-unknown-elf-
TILE_NPU_SIM ?= ../Tile-NPU-Sim
SIM_LIB      := $(TILE_NPU_SIM)/build/linux/x86_64/release/libsim_core.a
LINK_LD      ?= link.ld

BUILD   := build
VARIANT ?= full
VDIR    := $(BUILD)/$(VARIANT)
HDR     := $(VDIR)/model_big.h
DATA    := $(VDIR)/qwen3_big_data.bin
REF     := $(VDIR)/ref.npz
OUT     := $(VDIR)/out.bin
GUEST   := $(BUILD)/qwen3_$(VARIANT).bin
OBJECT  := $(BUILD)/main_$(VARIANT).o
BRIDGE  := $(BUILD)/sim_bridge

# x31 = blob base, x30 = stack top: fixed by the sim_bridge/crt0 convention, so
# the compiler must not allocate them.
TARGET  := --target=riscv64-unknown-elf -march=rv64im_xaicore -mabi=lp64 \
           -ffixed-x30 -ffixed-x31 -std=c++20 -O1 -nostdinc++ -isystem test_include
INC     := -I include -I $(VDIR) -I config
KERNELS := $(wildcard kernels/*.h)

.PHONY: all data big bridge check chat small clean help

all: big

help:
	@sed -n '2,17p' Makefile

# ---- model blob, header and numpy reference --------------------------------
# Grouped targets: one converter run produces all three. Delete build/<variant>
# to force a regeneration (e.g. after changing CONVERT_ARGS).
$(DATA) $(REF) $(HDR) &: tools/convert_pqwf.py tools/pqwf.py tools/ref_qwen3.py
	python3 tools/convert_pqwf.py --header $(HDR) --blob $(DATA) --ref $(REF) $(CONVERT_ARGS)

data: $(DATA)

# ---- guest ----------------------------------------------------------------
$(OBJECT): src/main_big.cpp $(KERNELS) include/tile.h $(HDR)
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) $(INC) -c $< -o $@
	# binutils older than the target clang rejects the `zmmul`/`xaicore`
	# attributes clang 21 emits; the linker script discards them anyway.
	$(CROSS)objcopy --remove-section=.riscv.attributes $@ $@.link

$(BUILD)/crt0.o: src/crt0.S
	@mkdir -p $(BUILD)
	$(CROSS)as -mabi=lp64 $< -o $@

$(GUEST): $(OBJECT) $(BUILD)/crt0.o
	$(CROSS)ld -T $(LINK_LD) $(OBJECT).link $(BUILD)/crt0.o -o $(BUILD)/qwen3_$(VARIANT).elf
	$(CROSS)objcopy -O binary $(BUILD)/qwen3_$(VARIANT).elf $@
	@echo "guest: $@ ($$(stat -c%s $@) bytes)"

big: $(GUEST)

# ---- simulator bridge ------------------------------------------------------
$(SIM_LIB):
	@echo "missing $(SIM_LIB): build Tile-NPU-Sim first (xmake in $(TILE_NPU_SIM))" >&2
	@false

$(BRIDGE): tools/sim_bridge.cpp $(SIM_LIB)
	@mkdir -p $(BUILD)
	g++ -std=c++20 -O2 -I $(TILE_NPU_SIM)/include -I $(TILE_NPU_SIM)/src \
	    tools/sim_bridge.cpp $(SIM_LIB) -o $@
	@echo "bridge: $@"

bridge: $(BRIDGE)

# ---- run + per-stage comparison -------------------------------------------
check: $(GUEST) $(BRIDGE) $(DATA) $(REF) $(HDR)
	./$(BRIDGE) $(GUEST) $(DATA) $(OUT)
	python3 tools/compare.py --header $(HDR) --out $(OUT) --ref $(REF)

# ---- small end-to-end smoke test ------------------------------------------
small:
	$(MAKE) VARIANT=small CONVERT_ARGS="--layers 1 --vocab 1024 --n-prompt 40 --n-gen 2" \
	        data big check

# ---- text in / text out ----------------------------------------------------
chat: $(GUEST) $(BRIDGE) $(HDR) $(DATA)
	python3 tools/run_chat.py "$(TEXT)" --max-new $(or $(MAX_NEW),8) \
	        --header $(HDR) --blob $(DATA) --guest $(GUEST) --bridge $(BRIDGE)

clean:
	rm -rf $(BUILD)
