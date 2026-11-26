#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 13：桑吉图（Sankey）流量分析（对应 2、引申出的分析方法 → 6 桑吉图）
====================================================================
【原 Power BI 实现内涵】教材用外部视觉对象 SankeyDiagram 展示网站页面之间的
用户流转：来源→目标→流量，观察流量在页面间的分配与汇聚。Power BI 只做拖拽
呈现，节点平衡与关键路径需要人工判读。
【Python 迁移要点】用 pandas 计算节点的流入/流出/净流量与平衡校验；用
matplotlib.path + PathPatch 直接绘制带宽度的桑基流带（不依赖 pyecharts /
plotly 等额外包），并自动识别起点、终点、最大流量路径与流失节点。
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "sankey", "flow.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "sankey")


def draw_sankey(flows, nodes, node_flow, ax):
    """用贝塞尔曲线绘制桑基流带。flows: DataFrame[来源,目标,流量]。"""
    pos = {n: i for i, n in enumerate(nodes)}  # 节点按拓扑层级排序后的纵向位置
    scale = 900.0 / max(node_flow.values())     # 流带宽度缩放
    cmap = plt.get_cmap("tab20")
    colors = {n: cmap(i / max(len(nodes), 1)) for i, n in enumerate(nodes)}

    # 先画流带（半透明），再画节点矩形
    for _, row in flows.iterrows():
        src, dst, val = row["来源"], row["目标"], float(row["流量"])
        y0 = pos[src] * 1.6 + val * scale / 2
        y1 = pos[dst] * 1.6 - val * scale / 2
        thick = max(val * scale, 1.5)
        x0, x1 = 0.35, 2.65
        mid = (x0 + x1) / 2
        verts = [
            (x0, y0), (mid, y0), (mid, y1), (x1, y1),  # 上边界
            (x1, y1 - thick), (mid, y1 - thick), (mid, y0 - thick), (x0, y0 - thick),
            (x0, y0),
        ]
        codes = [Path.MOVETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3 + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=colors[src],
                               edgecolor="none", alpha=0.45))

    for n, i in pos.items():
        h = max(node_flow[n] * scale, 4)
        ax.add_patch(plt.Rectangle((0.3, i * 1.6 - h / 2), 0.08, h,
                                   facecolor=colors[n], edgecolor="black", linewidth=0.6))
        ax.add_patch(plt.Rectangle((2.65, i * 1.6 - h / 2), 0.08, h,
                                   facecolor=colors[n], edgecolor="black", linewidth=0.6))
        ax.text(0.25, i * 1.6, n, ha="right", va="center", fontsize=10)
        ax.text(2.80, i * 1.6, n, ha="left", va="center", fontsize=10)

    ax.set_xlim(-1.0, 4.0)
    ax.set_ylim(-1.5, (len(nodes) - 1) * 1.6 + 1.5)
    ax.axis("off")


def main():
    os.makedirs(OUT, exist_ok=True)
    flows = pd.read_csv(SRC, dtype={"来源": str, "目标": str})
    flows["流量"] = pd.to_numeric(flows["流量"], errors="coerce").fillna(0).astype(int)

    section("1. 原始数据（来源→目标→流量）")
    print(flows.to_string(index=False))

    # 节点流入/流出/净流量
    inflow = flows.groupby("目标")["流量"].sum().to_dict()
    outflow = flows.groupby("来源")["流量"].sum().to_dict()
    nodes = sorted(set(flows["来源"]) | set(flows["目标"]),
                   key=lambda n: (n not in outflow, -outflow.get(n, 0), n))
    node_flow = {n: max(inflow.get(n, 0), outflow.get(n, 0)) for n in nodes}
    stats = pd.DataFrame([
        {"节点": n, "流入": inflow.get(n, 0), "流出": outflow.get(n, 0),
         "净流量": outflow.get(n, 0) - inflow.get(n, 0)}
        for n in nodes
    ])

    section("2. 节点流量与流失分析")
    print(stats.to_string(index=False))
    # 中间节点的"流入>流出"差额 = 在该页面流失的用户量（业务含义）
    mids = stats[(stats["流入"] > 0) & (stats["流出"] > 0)].copy()
    mids["流失量"] = mids["流入"] - mids["流出"]
    mids["流失率"] = (mids["流失量"] / mids["流入"] * 100).round(1)
    print("\n中间页面流失（流入−流出=未继续下钻的用户）：")
    print(mids[["节点", "流入", "流出", "流失量", "流失率"]].to_string(index=False))
    top_loss = mids.nlargest(1, "流失量").iloc[0]
    print(f"\n流失最大：{top_loss['节点']}（流失 {int(top_loss['流失量'])}，"
          f"流失率 {top_loss['流失率']:.1f}%）")

    section("3. 流量路径分解")
    # DFS 枚举从起点(无流入)到终点(无流出)的所有路径及流量（取路径上最小流量）
    starts = [n for n in nodes if inflow.get(n, 0) == 0]
    ends = set(n for n in nodes if outflow.get(n, 0) == 0)
    adj = defaultdict(list)
    for _, r in flows.iterrows():
        adj[r["来源"]].append((r["目标"], int(r["流量"])))
    paths = []

    def dfs(node, path, cap):
        if node in ends:
            paths.append((path, cap))
            return
        for nxt, v in adj[node]:
            dfs(nxt, path + [nxt], min(cap, v))

    for s in starts:
        dfs(s, [s], 10 ** 9)
    path_df = pd.DataFrame([
        {"路径": " → ".join(p), "路径流量": cap, "路径长度": len(p) - 1}
        for p, cap in sorted(paths, key=lambda x: -x[1])
    ])
    print(path_df.to_string(index=False))

    flow_sum = flows["流量"].sum()
    top = path_df.iloc[0]
    print(f"\n总流量 {flow_sum:,}；最大流量路径：{top['路径']}（{top['路径流量']:,}）")

    section("4. 绘制桑基流量图")
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_sankey(flows, nodes, node_flow, ax)
    ax.set_title("网站页面用户流转（桑吉图）\n流带宽度∝流量，节点高度=max(流入,流出)",
                 fontsize=14)
    fig.tight_layout()
    save_fig(fig, "sankey_flow.png", OUT)

    stats.to_csv(os.path.join(OUT, "node_stats.csv"), index=False, encoding="utf-8-sig")
    path_df.to_csv(os.path.join(OUT, "paths.csv"), index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至 {OUT}")


if __name__ == "__main__":
    main()
