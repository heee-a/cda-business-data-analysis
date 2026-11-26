#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验：DCMM 数据管理能力成熟度自评（第11章 数据治理介绍）
=========================================================
【教材依据】第11章配套 6 份规范 PDF：GB/T 36073-2018《数据管理能力成熟度
评估模型》(DCMM)、GB/T 42129-2022、GB/T34960.5-2018、IBM 数据治理统一流程、
《银行业金融机构数据治理指引》、《信息技术 大数据 数据治理实施指南》。
这些 PDF 只给规范条文，没有可运行的实验；本脚本把 DCMM 的 8 个能力域、
5 级成熟度模型落成可复用的 Python 自评工具。
【Python 迁移要点】按标准能力域/能力项组织评分 → 计算域得分与总体等级 →
识别短板域与优先改进项 → 雷达图可视化。评分来自 data/level1/dcmm/scores.csv
（演示用示例分；替换该文件即可用于真实自评）。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "dcmm", "scores.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "dcmm")

# GB/T 36073-2018 的 8 个能力域及其能力项（标准结构）
CAPABILITIES = {
    "数据战略": ["数据战略规划", "数据战略实施", "数据战略评估"],
    "数据治理": ["数据治理组织", "数据制度建设", "数据治理沟通"],
    "数据架构": ["数据模型", "数据分布", "数据集成与共享", "元数据管理"],
    "数据应用": ["数据分析", "数据开放共享", "数据服务"],
    "数据安全": ["数据安全策略", "数据安全管理", "数据安全审计"],
    "数据质量": ["数据质量需求", "数据质量检查", "数据质量分析", "数据质量提升"],
    "数据标准": ["业务术语", "参考数据和主数据", "数据元", "指标数据"],
    "数据生存周期": ["数据需求", "数据设计和开发", "数据运维", "数据退役"],
}
LEVELS = {1: "初始级", 2: "受管理级", 3: "稳健级", 4: "量化管理级", 5: "优化级"}


def demo_scores():
    """演示用示例评分（非教材原始数据），按教材规范结构构造。"""
    rows = []
    rng = np.random.default_rng(2025)
    for domain, items in CAPABILITIES.items():
        base = {"数据战略": 2.1, "数据治理": 2.4, "数据架构": 2.8, "数据应用": 3.0,
                "数据安全": 3.2, "数据质量": 2.6, "数据标准": 2.3,
                "数据生存周期": 2.0}[domain]
        for item in items:
            score = int(np.clip(round(base + rng.uniform(-0.7, 0.7)), 1, 5))
            rows.append({"能力域": domain, "能力项": item, "得分": score})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(SRC):
        scores = pd.read_csv(SRC, dtype=str)
        scores["得分"] = pd.to_numeric(scores["得分"], errors="coerce")
        print(f"读取自评数据：{SRC}")
    else:
        scores = demo_scores()
        print("未找到 scores.csv，使用演示用示例评分（替换该文件即可用于真实自评）")
        scores.to_csv(os.path.join(OUT, "demo_scores.csv"), index=False, encoding="utf-8-sig")

    scores = scores.dropna(subset=["能力域", "能力项", "得分"])
    scores["得分"] = scores["得分"].clip(1, 5)

    section("1. 评分明细")
    print(scores.to_string(index=False))

    section("2. 能力域得分汇总")
    domain = scores.groupby("能力域")["得分"].agg(["mean", "min", "max", "count"])
    domain.columns = ["平均得分", "最低分", "最高分", "能力项数"]
    domain = domain.reindex(list(CAPABILITIES))
    domain["平均得分"] = domain["平均得分"].round(2)
    print(domain.to_string())

    overall = scores["得分"].mean()
    level = int(np.floor(overall)) if overall < 5 else 5
    section("3. 总体成熟度判定")
    print(f"总体平均得分：{overall:.2f} / 5.00")
    print(f"对应成熟度等级：第 {level} 级 — {LEVELS.get(level, '未知')}")
    print("（GB/T 36073-2018 五级：初始级→受管理级→稳健级→量化管理级→优化级）")

    section("4. 短板域与优先改进项")
    short = domain[domain["平均得分"] < overall].sort_values("平均得分")
    print("低于总体平均的能力域：")
    print(short[["平均得分", "最低分"]].to_string())
    weak_items = scores[scores["得分"] <= 2].sort_values("得分")
    print(f"\n得分≤2 的能力项共 {len(weak_items)} 项，优先改进 TOP10：")
    print(weak_items[["能力域", "能力项", "得分"]].head(10).to_string(index=False))

    section("5. 雷达图")
    labels = list(domain.index)
    values = domain["平均得分"].tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.plot(angles, values, "o-", linewidth=2, color="#4C78A8")
    ax.fill(angles, values, alpha=0.25, color="#4C78A8")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title(f"DCMM 数据管理能力成熟度雷达图\n总体 {overall:.2f} 分 · 第{level}级"
                 f"（{LEVELS.get(level, '')}）", fontsize=14, pad=20)
    fig.tight_layout()
    save_fig(fig, "dcmm_radar.png", OUT)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    d = domain.sort_values("平均得分")
    colors = ["#E45756" if v < overall else "#54A24B" for v in d["平均得分"]]
    ax2.barh(d.index, d["平均得分"], color=colors)
    ax2.axvline(overall, color="gray", linestyle="--", label=f"总体平均 {overall:.2f}")
    ax2.set_xlim(0, 5)
    ax2.set_title("各能力域得分（红=低于平均，绿=高于平均）")
    ax2.legend()
    fig2.tight_layout()
    save_fig(fig2, "domain_scores.png", OUT)

    domain.to_csv(os.path.join(OUT, "domain_scores.csv"), encoding="utf-8-sig")
    weak_items.to_csv(os.path.join(OUT, "weak_items.csv"), index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至 {OUT}")


if __name__ == "__main__":
    main()
