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
LAYOUT  := $(VDIR)/serve_layout.h
DATA    := $(VDIR)/qwen3_big_data.bin
REF     := $(VDIR)/ref.npz
OUT     := $(VDIR)/out.bin
CHECK_BLOB := $(BUILD)/check_blob.bin
GUEST   := $(BUILD)/qwen3_$(VARIANT).bin
OBJECT  := $(BUILD)/main_$(VARIANT).o
BRIDGE  := $(BUILD)/sim_bridge
LLVM_OBJDUMP ?= $(dir $(CLANG))llvm-objdump
ASM_S   := $(BUILD)/main_$(VARIANT).s
ASM_DBG_OBJ := $(BUILD)/main_$(VARIANT)_dbg.o
ASM_DBG_ELF := $(BUILD)/qwen3_$(VARIANT)_dbg.elf
ASM_TXT := $(BUILD)/qwen3_$(VARIANT)_asm.txt
ASM_DIS := $(BUILD)/qwen3_$(VARIANT)_disasm.txt

# x31 = blob base, x30 = stack top: fixed by the sim_bridge/crt0 convention, so
# the compiler must not allocate them.
TARGET  := --target=riscv64-unknown-elf -march=rv64im_xaicore -mabi=lp64 \
           -ffixed-x30 -ffixed-x31 -std=c++20 -O1 -nostdinc++ -isystem test_include
INC     := -I include -I $(VDIR) -I config
KERNELS := $(wildcard kernels/*.h)

.PHONY: all data big bridge check chat small value-ops group-gemm paged-attn engine-test asm serve serve-debug serve-test serve-debug-test serve-test-long serve-test-pages clean help

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

# ---- IDE: compile_commands.json（clangd / VS Code 的跳转依赖它）----
compile_commands: tools/gen_compile_commands.py
	$(PY) tools/gen_compile_commands.py --variant $(VARIANT)

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
	@set -e; \
	cp --reflink=auto $(DATA) $(CHECK_BLOB) 2>/dev/null || cp $(DATA) $(CHECK_BLOB); \
	for mode in 0 1; do \
	    python3 tools/patch_blob.py --blob $(CHECK_BLOB) --header $(HDR) \
	        --mode-only --mode $$mode; \
	    ./$(BRIDGE) $(GUEST) $(CHECK_BLOB) $(OUT); \
	    if [ $$mode -eq 0 ]; then \
	        python3 tools/compare.py --header $(HDR) --out $(OUT) --ref $(REF) --fused --paged; \
	    else \
	        python3 tools/compare.py --header $(HDR) --out $(OUT) --ref $(REF) --paged; \
	    fi; \
	done

# ---- assembly / disassembly text ------------------------------------------
$(ASM_S): src/main_big.cpp $(KERNELS) include/tile.h $(HDR)
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) $(INC) -S -fverbose-asm $< -o $@

$(ASM_DBG_OBJ): src/main_big.cpp $(KERNELS) include/tile.h $(HDR)
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) $(INC) -g -c $< -o $@
	$(CROSS)objcopy --remove-section=.riscv.attributes $@ $@.link

$(ASM_DBG_ELF): $(ASM_DBG_OBJ) $(BUILD)/crt0.o
	$(CROSS)ld -T $(LINK_LD) $(ASM_DBG_OBJ).link $(BUILD)/crt0.o -o $@

$(ASM_TXT): $(ASM_S)
	@mkdir -p $(BUILD)
	{ \
	  echo "Qwen3 tile-NPU inference assembly"; \
	  echo "variant: $(VARIANT)"; \
	  echo "generated: $$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	  echo "source: src/main_big.cpp + kernels/*.h (inlined)"; \
	  echo "note: pre-link clang -S assembly; custom XAIcore mnemonics are authoritative."; \
	  echo; \
	  cat $(ASM_S); \
	} > $@
	@echo "asm: $@ ($$(stat -c%s $@) bytes)"

$(ASM_DIS): $(ASM_DBG_ELF)
	@mkdir -p $(BUILD)
	{ \
	  echo "Qwen3 tile-NPU inference linked disassembly"; \
	  echo "variant: $(VARIANT)"; \
	  echo "note: some custom XAIcore encodings may appear as <unknown> in llvm-objdump;"; \
	  echo "      use $(ASM_TXT) for the named assembly."; \
	  echo; \
	  $(LLVM_OBJDUMP) -d -S --demangle --mattr=+xaicore $(ASM_DBG_ELF); \
	} > $@
	@echo "disasm: $@ ($$(stat -c%s $@) bytes)"

asm: $(ASM_TXT) $(ASM_DIS)

# ---- serve: multi-request step guest -------------------------------------
$(LAYOUT): $(HDR) tools/serve_layout.py
	@mkdir -p $(VDIR)
	python3 tools/serve_layout.py --header $(HDR) --cpp $@

$(BUILD)/serve.bin: src/serve.cpp $(KERNELS) include/tile.h $(HDR) $(LAYOUT) $(BUILD)/crt0.o
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) -mcmodel=medany $(INC) -c $< -o $(BUILD)/serve_main.o
	$(CROSS)objcopy --remove-section=.riscv.attributes $(BUILD)/serve_main.o \
	    $(BUILD)/serve_main.o.link
	$(CROSS)ld -T $(LINK_LD) $(BUILD)/serve_main.o.link $(BUILD)/crt0.o \
	    -o $(BUILD)/serve.elf
	$(CROSS)objcopy -O binary $(BUILD)/serve.elf $@

serve: $(BUILD)/serve.bin
	@echo "serve: $@"

# Debug guest: same forward path plus sentinel/snap/dump/before scaffolding.
$(BUILD)/serve_debug.bin: src/serve.cpp $(KERNELS) include/tile.h $(HDR) $(LAYOUT) $(BUILD)/crt0.o
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) -mcmodel=medany -DYAI_SERVE_DEBUG $(INC) -c $< \
	    -o $(BUILD)/serve_debug_main.o
	$(CROSS)objcopy --remove-section=.riscv.attributes $(BUILD)/serve_debug_main.o \
	    $(BUILD)/serve_debug_main.o.link
	$(CROSS)ld -T $(LINK_LD) $(BUILD)/serve_debug_main.o.link $(BUILD)/crt0.o \
	    -o $(BUILD)/serve_debug.elf
	$(CROSS)objcopy -O binary $(BUILD)/serve_debug.elf $@

serve-debug: $(BUILD)/serve_debug.bin
	@echo "serve-debug: $@"

# Batch decode must produce the same tokens as one request at a time.
serve-test: $(BUILD)/serve.bin $(BRIDGE) $(DATA) $(HDR)
	python3 tools/run_serve.py --blob $(DATA) --header $(HDR) \
	    --guest $(BUILD)/serve.bin --bridge $(BRIDGE) \
	    --workdir $(abspath $(BUILD)/serve_work) --max-new 4 --test

# Cross-page prefill/decode + scheduler acceptance.
serve-test-long: $(BUILD)/serve.bin $(BRIDGE) $(DATA) $(HDR)
	python3 tools/serve_tests.py --blob $(DATA) --header $(HDR) \
	    --guest $(BUILD)/serve.bin --bridge $(BRIDGE) \
	    --workdir $(abspath $(BUILD)/serve_tests) --case long

# Page-pool exhaustion / waiting / release / reuse.
serve-test-pages: $(BUILD)/serve.bin $(BRIDGE) $(DATA) $(HDR)
	python3 tools/serve_tests.py --blob $(DATA) --header $(HDR) \
	    --guest $(BUILD)/serve.bin --bridge $(BRIDGE) \
	    --workdir $(abspath $(BUILD)/serve_tests) --case pages

serve-debug-test: $(BUILD)/serve_debug.bin $(BRIDGE) $(DATA) $(HDR)
	python3 tools/debug_serve.py --blob $(DATA) --header $(HDR) \
	    --guest $(BUILD)/serve_debug.bin --bridge $(BRIDGE) \
	    --workdir $(abspath $(BUILD)/serve_debug)

# ---- small end-to-end smoke test ------------------------------------------
small:
	$(MAKE) VARIANT=small CONVERT_ARGS="--layers 1 --vocab 1024 --n-prompt 40 --n-gen 2" \
	        data big check

# ---- value-form (no-copy) guest test --------------------------------------
$(BUILD)/value_ops.bin: tests/guest/value_ops.cpp include/tile.h $(BUILD)/crt0.o
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) $(INC) -c $< -o $(BUILD)/value_ops_main.o
	$(CROSS)objcopy --remove-section=.riscv.attributes $(BUILD)/value_ops_main.o \
	    $(BUILD)/value_ops_main.o.link
	$(CROSS)ld -T $(LINK_LD) $(BUILD)/value_ops_main.o.link $(BUILD)/crt0.o \
	    -o $(BUILD)/value_ops.elf
	$(CROSS)objcopy -O binary $(BUILD)/value_ops.elf $@

value-ops: $(BUILD)/value_ops.bin $(BRIDGE)
	python3 tools/check_value_ops.py --guest $(BUILD)/value_ops.bin --bridge $(BRIDGE)

# ---- per-group GEMM primitive test ---------------------------------------
$(BUILD)/group_gemm.bin: tests/guest/group_gemm.cpp include/tile.h kernels/group_gemm.h $(BUILD)/crt0.o
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) -mcmodel=medany $(INC) -c $< -o $(BUILD)/group_gemm_main.o
	$(CROSS)objcopy --remove-section=.riscv.attributes $(BUILD)/group_gemm_main.o \
	    $(BUILD)/group_gemm_main.o.link
	$(CROSS)ld -T $(LINK_LD) $(BUILD)/group_gemm_main.o.link $(BUILD)/crt0.o \
	    -o $(BUILD)/group_gemm.elf
	$(CROSS)objcopy -O binary $(BUILD)/group_gemm.elf $@

group-gemm: $(BUILD)/group_gemm.bin $(BRIDGE)
	python3 tools/check_group_gemm.py --guest $(BUILD)/group_gemm.bin --bridge $(BRIDGE)

# ---- isolated gate/up fusion test ----------------------------------------
$(BUILD)/gate_up.bin: tests/guest/gate_up.cpp include/tile.h kernels/group_gemm.h kernels/swiglu_quant.h $(BUILD)/crt0.o
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) -mcmodel=medany $(INC) -c $< -o $(BUILD)/gate_up_main.o
	$(CROSS)objcopy --remove-section=.riscv.attributes $(BUILD)/gate_up_main.o \
	    $(BUILD)/gate_up_main.o.link
	$(CROSS)ld -T $(LINK_LD) $(BUILD)/gate_up_main.o.link $(BUILD)/crt0.o \
	    -o $(BUILD)/gate_up.elf
	$(CROSS)objcopy -O binary $(BUILD)/gate_up.elf $@

gate-up: $(BUILD)/gate_up.bin $(BRIDGE)
	python3 tools/check_gate_up.py --guest $(BUILD)/gate_up.bin --bridge $(BRIDGE)

# ---- paged attention primitive test ---------------------------------------
$(BUILD)/paged_attn.bin: tests/guest/paged_attn.cpp include/tile.h kernels/attention_gqa.h kernels/kv_pages.h $(BUILD)/crt0.o
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) -mcmodel=medany $(INC) -c $< -o $(BUILD)/paged_attn_main.o
	$(CROSS)objcopy --remove-section=.riscv.attributes $(BUILD)/paged_attn_main.o \
	    $(BUILD)/paged_attn_main.o.link
	$(CROSS)ld -T $(LINK_LD) $(BUILD)/paged_attn_main.o.link $(BUILD)/crt0.o \
	    -o $(BUILD)/paged_attn.elf
	$(CROSS)objcopy -O binary $(BUILD)/paged_attn.elf $@

paged-attn: $(BUILD)/paged_attn.bin $(BRIDGE)
	python3 tools/check_paged_attn.py --guest $(BUILD)/paged_attn.bin --bridge $(BRIDGE)

# ---- host engine scaffolding test ---------------------------------------
engine-test:
	python3 tools/check_engine.py

# ---- multi-request paged attention test ----------------------------------
$(BUILD)/paged_attn_multi.bin: tests/guest/paged_attn_multi.cpp include/tile.h kernels/attention_gqa.h kernels/kv_pages.h $(BUILD)/crt0.o
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) -mcmodel=medany $(INC) -c $< -o $(BUILD)/paged_attn_multi_main.o
	$(CROSS)objcopy --remove-section=.riscv.attributes $(BUILD)/paged_attn_multi_main.o \
	    $(BUILD)/paged_attn_multi_main.o.link
	$(CROSS)ld -T $(LINK_LD) $(BUILD)/paged_attn_multi_main.o.link $(BUILD)/crt0.o \
	    -o $(BUILD)/paged_attn_multi.elf
	$(CROSS)objcopy -O binary $(BUILD)/paged_attn_multi.elf $@

paged-attn-multi: $(BUILD)/paged_attn_multi.bin $(BRIDGE)
	python3 tools/check_paged_attn_multi.py --guest $(BUILD)/paged_attn_multi.bin --bridge $(BRIDGE)

# ---- batch K/V scatter test -----------------------------------------------
$(BUILD)/paged_kv_batch.bin: tests/guest/paged_kv_batch.cpp include/tile.h kernels/kv_pages.h $(BUILD)/crt0.o
	@mkdir -p $(BUILD)
	$(CLANG) $(TARGET) -mcmodel=medany $(INC) -c $< -o $(BUILD)/paged_kv_batch_main.o
	$(CROSS)objcopy --remove-section=.riscv.attributes $(BUILD)/paged_kv_batch_main.o \
	    $(BUILD)/paged_kv_batch_main.o.link
	$(CROSS)ld -T $(LINK_LD) $(BUILD)/paged_kv_batch_main.o.link $(BUILD)/crt0.o \
	    -o $(BUILD)/paged_kv_batch.elf
	$(CROSS)objcopy -O binary $(BUILD)/paged_kv_batch.elf $@

paged-kv-batch: $(BUILD)/paged_kv_batch.bin $(BRIDGE)
	python3 tools/check_paged_kv_batch.py --guest $(BUILD)/paged_kv_batch.bin --bridge $(BRIDGE)

# ---- text in / text out ----------------------------------------------------
chat: $(GUEST) $(BRIDGE) $(HDR) $(DATA)
	python3 tools/run_chat.py "$(TEXT)" --max-new $(or $(MAX_NEW),8) \
	        --header $(HDR) --blob $(DATA) --guest $(GUEST) --bridge $(BRIDGE)

clean:
	rm -rf $(BUILD)
