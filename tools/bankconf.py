#!/usr/bin/env python3
"""统计双操作数运算的两个寄存器是否落在同一个 bank。

四类寄存器文件各分两个 bank（前半 = bank0，后半 = bank1）：
  tile  XTT: t0..t7  / t8..t15          （mma 的 A/B）
  acc   XTA: a0..a7  / a8..a15          （aadd/amul/... 的原地读 + 源）
  vec32 XTV: v0..v15 / v16..v31         （vadd/vmax/... 的两个源）
  vec8  XTB: b0..b15 / b16..b31

同 bank 仍然正确，但要串行读两次（约慢一倍）。
"""
import re
import sys
from collections import Counter, defaultdict

BANKS = {'t': 8, 'a': 8, 'v': 16, 'b': 16}   # 每类 bank0 的个数
GROUPS = {
    'tile  (mma A/B)':   (re.compile(r'^\s*mma\.[a-z0-9.]+\s+[atv]\d+\s*,\s*(t\d+)\s*,\s*(t\d+)'), 2),
    'vec32 (v* 双源)':   (re.compile(r'^\s*(?:vadd|vsub|vmul|vdiv|vmax|vmulx|vaddx)\s+([^\n]*)'), -1),
    'acc   (a* 双源)':   (re.compile(r'^\s*(?:aadd|asub|amul|adiv)\s+([^\n]*)'), -1),
}


def bank(reg):
    """reg 形如 t3/a12/v20 → bank0=0 / bank1=1"""
    p, n = reg[0], int(reg[1:])
    return 0 if n < BANKS[p] else 1


def regs_of(line, prefix):
    return [r for r in re.findall(r'\b([tavb]\d+)\b', line) if r[0] == prefix]


def main(path):
    stats = defaultdict(Counter)
    for line in open(path):
        # tile：mma 的两个 source
        m = re.match(r'^\s*mma\.[a-z0-9.]+\s+[atv]\d+\s*,\s*(t\d+)\s*,\s*(t\d+)', line)
        if m:
            a, b = m.groups()
            stats['tile  (mma A/B)'][('same', bank(a) == bank(b))] += 1
            continue
        # vec32：只数"两个不同 vec 源"的指令
        if re.match(r'^\s*(?:vadd|vsub|vmul|vdiv|vmax)\s', line):
            uniq = list(dict.fromkeys(regs_of(line, 'v')))   # 去重后再取两个不同的
            if len(uniq) >= 2:
                a, b = uniq[0], uniq[1]
                stats['vec32 (v* 双源)'][('same', bank(a) == bank(b))] += 1
            continue
        # acc：只数"两个不同 acc"的指令
        if re.match(r'^\s*(?:aadd|asub|amul|adiv)\s', line):
            uniq = list(dict.fromkeys(regs_of(line, 'a')))
            if len(uniq) >= 2:
                a, b = uniq[0], uniq[1]
                stats['acc   (a* 双源)'][('same', bank(a) == bank(b))] += 1

    total_all = same_all = 0
    for name in ('tile  (mma A/B)', 'vec32 (v* 双源)', 'acc   (a* 双源)'):
        c = stats[name]
        total = c[('same', True)] + c[('same', False)]
        same = c[('same', True)]
        total_all += total; same_all += same
        if total == 0:
            print(f'  {name:20s} （无此类指令）')
            continue
        print(f'  {name:20s} 共 {total:>6,}  同 bank {same:>6,} ({same/total*100:5.1f}%)  '
              f'跨 bank {total-same:>6,} ({(total-same)/total*100:5.1f}%)')
    if total_all:
        print(f'  {"合计":20s} 共 {total_all:>6,}  同 bank {same_all:>6,} ({same_all/total_all*100:5.1f}%)')


if __name__ == '__main__':
    main(sys.argv[1])
