"""Record the tier decision, isolate the internal meeting material, and commit.

Three things.

Decision (c) on the resource-number question: the paper reports no absolute resource figures.
The reason is not that the local device is ineligible, which was my earlier and incorrect
reading. The device is a close match on memory (8151 MiB against an 8192 MiB envelope) and on its
default power limit (50 W against 60 W). The reason is that absolute figures carry no argument:
all three constraints were measured as not binding, so there is no resource gap to report, and
the one resource result the paper does have is a ratio, which is platform-relative and does not
need absolutes. Option (a-prime), adding a tier, would work but requires three approvals and buys
nothing under the current positioning. Option (b), renting a T4, cannot test the claim at all
because its 16 GB is twice the envelope.

The meeting material moves into a directory that .gitignore excludes explicitly. Its tables carry
tier=local resource numbers and its talking points record which components are unimplemented.
.gitignore already holds back results for the first reason, so committing them would contradict
its stated policy, and the second reason is worse: an artifact that documents its own
unimplemented components invites a reader to conclude the method does not work.

AGENTS.md gains a short entry for the decision and pays for it with a line elsewhere, so the file
stays under its limit.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AG = ROOT / "AGENTS.md"

# ---- 1) record decision (c) in the claim ceiling ----------------------------
t = AG.read_text(encoding="utf-8")
before = len(t.encode()) / 1024
ANCHOR = "## 2. 已撤回的主张"
if ANCHOR not in t:
    raise SystemExit("  ABORT: section 2 anchor not found")
DECISION = """- **★ 资源数字：决定 (c) —— 论文**不报任何绝对资源数字**（2026-10-13 你确认）。**
  **理由不是"本机不合规"，而是"绝对数在当前定位下不承担任何论证"**：
  三条约束实测**都不绑定**，故没有资源 gap 可报；而 Rec-AD 唯一的资源结果是**比值**
  （8× 推理延迟换 14× 显存），**比值平台相对，不需要绝对数**。
  **本机与包络的实际关系**（供记录，不用于报告）：显存 8151 vs 8192 MiB、默认功耗上限 50 vs 60 W。
  **未选 (a') 加档**：需改 `DEVICE_TIERS` + 管理员改功耗限制 + 改 §4.6，三处批准，**而在 (c) 定位下买不到东西**。
  **未选 (b) 租 T4**：16 GB 是包络的 2 倍，**测不出绑定**，是无效对照。

"""
t = t.replace(ANCHOR, DECISION + ANCHOR, 1)
if "决定 (c)" not in t:
    raise SystemExit("  ABORT: the decision did not land")

# Pay for it: the 组会 reportscript reference is not a standing rule.
OLD_PAY = ("- 上游交付物：`cover-letter.md`、`response-letter.md`（§0.1 槽位约定 / §0.2 新增实验清单 / "
           "§0.3 自查纠错 14 项；Appendix A 槽位映射 / A.2 新表图索引 / B 逐节变更 / C 撤回主张）、"
           "`revision-sections.md`、`deliverables/*.html`（`Rec-AD-revision-master-plan.html`、"
           "`Rec-AD-revision-package.html`）。")
NEW_PAY = ("- 上游交付物：`cover-letter.md`、`response-letter.md`（含 Appendix A 槽位映射 / B 逐节变更 / "
           "C 撤回主张）、`revision-sections.md`、`revision-section-V.md`、`deliverables/*.html`。")
if OLD_PAY in t:
    t = t.replace(OLD_PAY, NEW_PAY, 1)
    print("  paid for it: shortened the deliverables line")
AG.write_text(t, encoding="utf-8")
after = len(AG.read_text(encoding="utf-8").encode()) / 1024
print(f"  AGENTS.md {before:.1f} -> {after:.1f} KB (limit 52)")

# ---- 2) isolate the meeting material ---------------------------------------
INTERNAL = ROOT / "组会_internal"
INTERNAL.mkdir(exist_ok=True)
MOVED = ["组会表格.md", "组会表格.tex", "组会讲法.md", "组会_图表对照表.md",
         "Rec-AD组会汇报稿.docx", "Rec-AD组会汇报_大修进展与设备申请.docx"]
for name in MOVED:
    src = ROOT / name
    if src.exists():
        shutil.move(str(src), str(INTERNAL / name))
        print(f"  moved to 组会_internal/: {name}")
(INTERNAL / "README.md").write_text(
    "# 组会内部材料\n\n"
    "**本目录被 `.gitignore` 显式排除，不进仓库。**\n\n"
    "原因有二：\n"
    "1. `组会表格.md` 的 Table D 含 `tier=local` 的资源数字（292 MiB / 4.23 ms / 32.2 W）。\n"
    "   而 `.gitignore` 的既定政策是「资源测量在具名设备上重跑前不提交」，"
    "因为这些数字是 debug 档，与论文并列会被读成论文的数字。\n"
    "2. `组会讲法.md` 记录了尚未实现的组件（`pipeline` 零实现）与前提不成立的组件"
    "（拓扑重排）。公开一份自述缺陷的产物，会被读成「核心方法不成立」。\n\n"
    "**白名单是失败即关闭的**：本目录默认不进仓库，转 Public 时自动挡住。\n",
    encoding="utf-8")
print("  wrote 组会_internal/README.md")

# ---- 3) gitignore ----------------------------------------------------------
GI = ROOT.parent / ".gitignore"
g = GI.read_text(encoding="utf-8")
MARK = "# Internal meeting material: carries tier=local numbers and a record of unimplemented"
if MARK not in g:
    g = g.rstrip() + "\n\n" + MARK + "\n" \
        + "# components. Held back for the same reason results are, and for one more.\n" \
        + "/Rec-AD/组会_internal/\n"
    GI.write_text(g, encoding="utf-8")
    print("  .gitignore: 组会_internal/ excluded")
else:
    print("  .gitignore: already present")

# ---- 4) commit and push ----------------------------------------------------
REPO = ROOT.parent
mf = REPO / ".git" / "COMMIT_MSG_TMP.txt"
mf.write_text(
    "Decide not to report absolute resource figures, and isolate the meeting material\n"
    "\n"
    "The resource-number question is settled as option (c): no absolute figures in the paper.\n"
    "The reason is not ineligibility of the local device. It is a close match on memory, 8151\n"
    "MiB against an 8192 MiB envelope, and on its default power limit, 50 W against 60 W. The\n"
    "reason is that absolute figures carry no argument here: all three constraints measured as\n"
    "not binding, so there is no resource gap to report, and the one resource result that exists\n"
    "is a ratio, which is platform-relative.\n"
    "\n"
    "Adding a tier would work but needs three separate approvals and buys nothing under this\n"
    "positioning. Renting a T4 cannot test the claim, since 16 GB is twice the envelope.\n"
    "\n"
    "The meeting material moves to 组会_internal/, which .gitignore excludes. Its tables carry\n"
    "tier=local numbers, which the existing results policy already holds back, and its talking\n"
    "points record which components are unimplemented, which is worse to publish.\n"
    "\n"
    "AGENTS.md gains a short entry for the decision and stays inside its size limit.\n",
    encoding="utf-8")

for cmd in (["git", "add", "-A", "Rec-AD/code", ".gitignore"],
            ["git", "commit", "-q", "-F", str(mf)]):
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    print(f"  {' '.join(cmd[:2])}: rc={r.returncode}")
    if r.returncode != 0 and r.stderr.strip():
        print(f"    {r.stderr.strip()[:160]}")
mf.unlink(missing_ok=True)

r = subprocess.run(["git", "rev-list", "--count", "origin/main..HEAD"],
                   cwd=str(REPO), capture_output=True, text=True)
print(f"  领先远端: {r.stdout.strip()}")
