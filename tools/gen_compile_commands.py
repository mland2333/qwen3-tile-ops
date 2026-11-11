#!/usr/bin/env python3
"""生成给 clangd / VS Code 用的 compile_commands.json。

  * src/main_big.cpp —— 设备侧 guest（riscv64；真实构建用
    -march=rv64im_xaicore，clangd 自带的 clang 不认识这个扩展，所以 IDE 这里
    退化成标准 -march=rv64im，只影响语义解析，不影响构建）；
  * 头文件（include/tile.h、kernels/*.h、以及自动生成的 build/<variant>/model_big.h）
    由 clangd 从包含它的这条 TU 推断编译命令。

用法：tools/gen_compile_commands.py [--variant full]
"""

import argparse
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
clang = str((root / ".." / "llvm-project" / "build" / "bin" / "clang++").resolve())

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="full")
args = parser.parse_args()

vdir = f"build/{args.variant}"
entry = {
    "directory": str(root),
    "file": str(root / "src" / "main_big.cpp"),
    "arguments": [
        clang,
        "--target=riscv64-unknown-elf",
        "-march=rv64im",          # IDE 近似；真实构建用 rv64im_xaicore
        "-mabi=lp64",
        "-ffixed-x30",
        "-ffixed-x31",
        "-std=c++20",
        "-O1",
        "-nostdinc++",
        "-isystem", "test_include",
        "-I", "include",
        "-I", vdir,
        "-I", "config",
        "-c", "src/main_big.cpp",
        "-o", f"build/main_{args.variant}.o",
    ],
}

out = root / "compile_commands.json"
out.write_text(json.dumps([entry], indent=2) + "\n")
print(f"wrote {out} (variant={args.variant})")
