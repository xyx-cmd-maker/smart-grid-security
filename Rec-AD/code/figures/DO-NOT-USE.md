# DO NOT USE THESE FIGURES

**★ 本目录下的 `fig11_*.pdf` … `fig17_*.pdf` 不得用于论文、回复信、cover letter 或任何对外文本。**

## 原因

它们由 `code/analysis/plot.py` 生成，而该脚本用：

```python
for path in sorted(results_dir.glob("*.json")):
```

**读取 `code/results/` 下的【全部】JSON —— 包括 `*__selftest_*.json`。**

而本项目的硬约束（`AGENTS.md` §4.2）写着：

> **`*__selftest_*` 是合成小数据的管线自检产物，永远不得进入论文、回复信、cover letter 或任何对外文本。**

**★ 因此这 7 张图的数值序列可能来自合成自检数据，而图上看不出来。**

## 受影响的文件

| 文件 | 状态 |
|---|---|
| `fig11_pipeline.pdf` | ★ 不可用 |
| `fig12_deployability.pdf` | ★ 不可用 |
| `fig13_rank_sweep.pdf` | ★ 不可用（且 `rank_sweep` 族**只有** selftest 数据） |
| `fig14_scaling.pdf` | ★ 不可用 |
| `fig15_reorder.pdf` | ★ 不可用 |
| `fig16_ablation.pdf` | ★ 不可用 |
| `fig17_fedavg.pdf` | ★ 不可用 |

## 用这些代替

**★ 从真实数据重画的图在 `code/figures/pub2/`，并由 `code/analysis/plot_pub.py` 一类脚本生成，
只读【名字里带真实算例标识】的文件（`ablation_ieee118_*` / `scaling_local_*` / `fedavg_*`）。**

| 替代图 | 数据来源 |
|---|---|
| `pub2/figR_ablation.pdf` | `ablation_ieee118_local_abl3_topo.json` |
| `pub2/figR_reorder.pdf` | `reorder_ieee118_local_c4_reorder.partial.json` |
| `pub2/figR_latency.pdf` | `power_latency.json` |
| `pub2/figR_fedavg_*` | `fedavg_*.json` |

## 若要重画

**判据：绘图脚本读文件时必须【排除】`selftest` / `synth` / `partial` / `smoke`，并在
注释里写明它排除了什么。**
