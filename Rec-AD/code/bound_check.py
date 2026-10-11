"""Does the finding depend on the dataset I generated? Test it without my data.

The challenge is fair: the feature design in data_cache/*.npz is mine, not the
paper's. Only bus_id is named in the paper; time_window and the four bins were
invented. So `799 rows` is a statement about my implementation, not about the
paper's data.

This script makes the opposite kind of argument: one that uses ONLY the
advisor's own numbers and a counting bound, with no reference to any dataset.
"""

S = 174_375          # 样本数, implied by the advisor's formula
F = 7                # 稀疏特征数, from Table II
D = 16               # 维度, from Table II
CLAIMED = 19_530_000  # Table II 的 Rows 值

print("=" * 80)
print("不依赖任何数据集的论证")
print("=" * 80)
print()
print(f"前提（全部来自老师与 Table II，与我生成的数据无关）：")
print(f"  样本数 = {S:,}")
print(f"  稀疏特征数 = {F}")
print(f"  维度 = {D}")
print(f"  Table II 声称的 Rows = {CLAIMED:,}")
print()

print("=" * 80)
print("第 1 步：嵌入表行数的硬上界")
print("=" * 80)
print()
print("  嵌入表的每一行对应一个『被观测到的特征取值』。")
print("  而每一个被观测到的取值，至少要出现在一条样本的该特征位置上。")
print()
print(f"  所以  总观测次数 = 样本数 x 特征数 = {S:,} x {F} = {S * F:,}")
print(f"        表行数 <= 总观测次数 = {S * F:,}")
print()
print("  这是纯计数上界：无论特征怎么设计、基数怎么分布，都不可能超过它。")
print()

print("=" * 80)
print("第 2 步：与 Table II 对比")
print("=" * 80)
print()
print(f"  Table II 声称      {CLAIMED:,}")
print(f"  硬上界             {S * F:,}")
print(f"  比值               {CLAIMED / (S * F):.4f}")
print()
print(f"  => 比值恰好等于维度 {D}。")
print()
print(f"  也就是说：Table II 的 Rows 单元格里装的，正好是")
print(f"            样本数 x 特征数 x 维度 = {S:,} x {F} x {D} = {CLAIMED:,}")
print()
print("  这个乘积的含义是：如果每个特征在每条样本上都取一个全新值，")
print("  那么这张表会有多少【参数】。")
print("  它=【最大可能行数】x【维度】。")
print()

print("=" * 80)
print("第 3 步：即使按最宽松的假设，表也远达不到 19.53M")
print("=" * 80)
print()
best_rows = S * F
best_params = best_rows * D
print(f"  最宽松假设（每个特征取值都不同、且全部出现过）：")
print(f"      行数上界   {best_rows:,}")
print(f"      参数量     {best_params:,}   <- 这才是 {CLAIMED:,} 对应的量")
print(f"      显存       {best_params * 4 / 1024**2:.1f} MB  (fp32)")
print()
print(f"  Table II 声称的显存：1.22 GB = {1.22 * 1024:.0f} MB")
print(f"  差 {1.22 * 1024 / (best_params * 4 / 1024**2):.1f} 倍")
print()

print("=" * 80)
print("结论")
print("=" * 80)
print()
print("  这个论证不需要我生成的数据，只需要老师给的样本数。")
print()
print("  1. 表格的 Rows 列装的是 `样本数 x 特征数 x 维度`，即参数量，不是行数。")
print("  2. 无论特征怎么设计，行数都不可能超过 样本数 x 特征数 = "
      f"{best_rows:,}。")
print("  3. 所以 19.53M 这个数，作为『嵌入表行数』在任何特征设计下都不成立。")
print()
print("  仍然需要我生成的数据才能回答的问题是：")
print("      『真实的表到底有多少行？』—— 这取决于当年那 7 个特征是什么。")
