# Tile-NPU-Sim 的 guest 启动代码。
#
# 与 ~/Tile-NPU-Sim/tests/guest/crt0.S 的区别：结束指令用 NPU 的 kernel.end
# （0x86000002），而不是 ebreak。sim_bridge 只把 kernel_end 当成正常结束；
# ebreak 会得到 StepStatus::breakpoint，被当作失败。
#
# 链接脚本把 .text.crt0 放在 0x80000000 的最前面，所以 _start 就是入口。

    .section .text.crt0, "ax", @progbits
    .globl _start
_start:
    # sim_bridge 的 RAM 只有 1MiB 代码区 + 数据区（数据区会被 harness
    # padding 到 512KiB），所以栈必须落在 0x80100000 + 512KiB 以内。
    li      sp, 0x8017F000
    call    main
    .word   0x86000002          # kernel.end
