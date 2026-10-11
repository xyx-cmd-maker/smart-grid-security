"""What an embedding table actually is, measured on the real IEEE datasets.

Answers the question: Table II says IEEE118-Bus has 19.53M embedding rows.
This computes what the table actually is under the DLRM convention, decomposes
it by feature, and tests the stated derivation `features x dimension x data
volume` against it.

Every cardinality is read from the generated contract file's own `sparse`
array -- the values the model would actually index. Nothing is estimated.
No GPU and no training required.

Usage:
    cd code
    python analyze_embedding.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data_cache"

DIM = 16                 # Table II "Dimension" column for IEEE118-Bus
BYTES = 4                # fp32
CLAIMED_ROWS = 19.53e6
CLAIMED_SIZE_GB = 1.22
CASES = ("ieee14", "ieee57", "ieee118", "ieee300")


def load(name: str):
    with np.load(CACHE / f"{name}.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        sparse = z["sparse"]
    cols = meta["sparse_columns"]
    cards = [int(sparse[:, c].max()) + 1 for c in range(sparse.shape[1])]
    return cols, cards, sparse.shape[0]


def main() -> None:
    print("=" * 84)
    print("1. 每个 IEEE 算例的嵌入表实际有多大（DLRM 约定）")
    print("=" * 84)
    print()
    print("DLRM 约定：每个稀疏特征一张独立嵌入表")
    print("          总行数 = 各特征基数之和（不是乘积）")
    print("          显存   = 总行数 x 维度 x 4 字节 (fp32)")
    print()

    res = {}
    for name in CASES:
        cols, cards, n = load(name)
        rows = sum(cards)
        size_kb = rows * DIM * BYTES / 1024
        res[name] = dict(cols=cols, cards=cards, rows=rows, size_kb=size_kb, n=n)
        print(f"--- {name}  (样本数 {n:,}) ---")
        for c, card in zip(cols, cards):
            print(f"      {c:<20} 基数 {card:>6,}")
        print(f"      {'合计行数':<20}      {rows:>6,}")
        print(f"      {'显存 (16 维 fp32)':<20} {size_kb:>9.1f} KB")
        print()

    print("=" * 84)
    print("2. 与 Table II 声称的对比")
    print("=" * 84)
    print()
    a = res["ieee118"]
    print(f"  Table II 声称 IEEE118-Bus : {CLAIMED_ROWS:,.0f} 行 , {CLAIMED_SIZE_GB} GB")
    print(f"  实际算得                  : {a['rows']:,} 行 , {a['size_kb']:.1f} KB")
    print(f"  倍数差                    : {CLAIMED_ROWS / a['rows']:,.0f} x")
    print()

    print("=" * 84)
    print("3. 检验公式：特征 x 维度 x 数据量 == 19.53M ?")
    print("=" * 84)
    print()
    n_sparse = len(a["cols"])
    implied = CLAIMED_ROWS / (n_sparse * DIM)
    print(f"  稀疏特征数 = {n_sparse} , 维度 = {DIM}")
    print(f"  => 反解出的『数据量』= {implied:,.1f}")
    print(f"     是整数吗: {'是' if abs(implied - round(implied)) < 1e-6 else '否'}")
    print()
    print("  用真实样本数代入：")
    for name in CASES:
        n = res[name]["n"]
        print(f"    {name:<9} 样本 {n:>7,}  ->  {n_sparse * DIM * n:>16,}")
    print()

    print("=" * 84)
    print("4. 反解出的『数据量』对得上什么")
    print("=" * 84)
    print()
    print(f"  {implied:,.0f} 与系统内各自然量的比值：")
    for label, v in (("母线数", a["cards"][2]),
                     ("量测数", a["cards"][1]),
                     ("特征基数之和", a["rows"]),
                     ("本次样本数", a["n"])):
        print(f"    {label:<14} {v:>10,}   差 {implied / v:>10,.1f} 倍")
    print()

    print("=" * 84)
    print("5. 结论")
    print("=" * 84)
    print()
    print("  嵌入表行数由『特征有多少种取值』决定，与样本数无关。")
    print("  一万条样本和一百万条样本，只要特征基数不变，嵌入表大小完全相同。")
    print()
    print("  而 `样本数 x 特征数 x 维度` 是每次前向传播的查表输出量：")
    print("  它随样本数变化，是计算量，不是存储量。")
    print()
    print(f"  按 DLRM 约定，IEEE118 的表是 {a['rows']:,} 行 / {a['size_kb']:.0f} KB。")
    print(f"  要得到 19.53M 行，特征基数之和必须达到 19.53M，")
    print(f"  即平均每个特征 {CLAIMED_ROWS / n_sparse:,.0f} 个取值——")
    print(f"  在 118 节点的电网里不可能（单个特征要有 {CLAIMED_ROWS / n_sparse / 1e6:.1f}M 种取值）。")


if __name__ == "__main__":
    main()
