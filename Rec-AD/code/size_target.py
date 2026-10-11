"""Before redesigning features, establish what table size the contribution needs.

The contribution is: FDIA detectors exceed a substation node's resource
envelope, so the embedding table must be compressed to fit. That claim requires
the table to actually NOT fit. This computes the thresholds.
"""

VRAM_BUDGET_GB = 8.0          # code/config.yaml -> budget.vram_gb
TABLE_II_ROWS = 19_530_000    # Table II, IEEE118-Bus
DIMS = (16, 32, 64)

GB = 1024**3

print("=" * 82)
print("1. 论文自己声称的数字，放得进预算吗？")
print("=" * 82)
print()
size = TABLE_II_ROWS * 16 * 4 / GB
print(f"  预算                    {VRAM_BUDGET_GB} GB   (config.yaml budget.vram_gb)")
print(f"  Table II 声称 IEEE118   {TABLE_II_ROWS:,} 行 x 16 维 x 4B = {size:.2f} GB")
print()
verdict = "放得进" if size < VRAM_BUDGET_GB else "放不进"
print(f"  => {verdict}。占预算的 {size / VRAM_BUDGET_GB * 100:.1f}%。")
print()
print("  ** 即使接受论文原来的 19.53M / 1.22 GB，IEEE118 这一行也不构成内存问题。**")
print()

print("=" * 82)
print("2. 要让嵌入表「装不进」，需要多大")
print("=" * 82)
print()
print(f"{'维度':>6}{'超标所需行数':>18}{'表大小':>14}")
print("-" * 82)
for d in DIMS:
    rows = int(VRAM_BUDGET_GB * GB / (d * 4)) + 1
    print(f"{d:>6}{rows:>18,}{VRAM_BUDGET_GB:>12.1f} GB+")
print()
need16 = int(VRAM_BUDGET_GB * GB / (16 * 4)) + 1
print(f"  取维度 16：需要 > {need16:,} 行 才超过 8 GB。")
print(f"  这是原 Table II 数字的 {need16 / TABLE_II_ROWS:.1f} 倍。")
print()

print("=" * 82)
print("3. 在电力系统里，多大的特征基数才能到那个量级")
print("=" * 82)
print()
print("  最有物理意义的高基数特征是 `(量测点, 时间桶)` 交叉：")
print("  真实 SCADA 是连续流，这个交叉的基数随时间单调增长——")
print("  这正是论文描述的『industrial-scale embedding tables』的来源。")
print()
print(f"{'量测点数':>10}{'需要的时间桶数':>18}{'表行数':>18}{'表大小(16维)':>16}")
print("-" * 82)
for n_meas, label in ((297, "IEEE 118"), (709, "IEEE 300"), (5000, "地调规模"), (20000, "省级规模")):
    buckets = need16 / n_meas
    rows = n_meas * buckets
    print(f"{n_meas:>10,}{buckets:>18,.0f}{rows:>18,.0f}{rows * 16 * 4 / GB:>14.1f} GB")
print()
print("  时间桶数与采样间隔的关系（IEEE 118，297 个量测点）：")
print()
for interval, secs in (("1 秒", 1), ("10 秒", 10), ("1 分钟", 60), ("5 分钟", 300)):
    buckets_needed = need16 / 297
    days = buckets_needed * secs / 86400
    print(f"    {interval:<8} 需要 {days:>8.1f} 天的连续数据")
print()

print("=" * 82)
print("4. 含义")
print("=" * 82)
print()
print("  两个独立的结论：")
print()
print("  (a) 『把 Table II 改对』和『让第一条贡献成立』是两个不同的任务。")
print(f"      即使保留 19.53M（{size:.2f} GB），它也放得进 8 GB 预算——")
print("      这一行无论填 799 还是 19.53M，都不构成内存问题。")
print()
print(f"  (b) 要让表真的超标（维度 16），需要 > {need16:,} 行，")
print("      这要求量测点数与时间桶数的乘积达到亿级。")
print("      单靠 IEEE 118（297 个量测点）做不到——除非时间跨度极长。")
print()
print("  => 特征重设计的同时，很可能还需要扩大算例规模或采样时长。")
print("     这是比「改特征」大一级的改动，需要老师确认。")
