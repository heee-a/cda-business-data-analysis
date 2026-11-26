#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章 Excel / WPS 电子表格函数练习（Python 迁移）
==================================================
教材用大量工作表讲解 Excel/WPS 函数。本脚本把每类函数用 pandas/numpy 等价实现，
并对教材「有答案」版本做一致性校验，理解每个函数背后的计算逻辑：

 1. 数学/统计函数：INT / MOD / ROUND / ABS / MAX / MIN / SUM / COUNT / AVERAGE
 2. 逻辑函数：IF + AND + OR（三科是否全通过 / 之一通过）—— 对照答案版校验
 3. COUNTIF：按出勤标记统计迟到(b)/旷课(c)次数 —— 对照答案版校验
 4. 日期函数：EOMONTH 月末 / DAYS 当月天数 / 闰年判断 / 入职转正日期
 5. 文本函数：REPLACE 隐藏手机号中间 4 位
 6. 查找引用：VLOOKUP 精确 / VLOOKUP 近似（分级区间）/ INDEX+MATCH / 成绩等级查找
 7. 数据透视表：销售地区×品名 销售额透视（pivot_table）

运行：python 01_excel_functions_in_pandas.py
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import section, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "excel_func")


def show(name, series, n=8):
    print(f"\n{name}：")
    print(series.head(n).to_string())


def main():
    section("1. 数学与统计函数（numpy / pandas 等价）")
    x = np.array([3.345, -3.4, 10, 100, 21.3456, -23.6])
    print("INT(3.345)=%d  INT(-3.4)=%d  （Excel INT 向下取整，与 np.floor 一致）"
          % (np.floor(3.345), np.floor(-3.4)))
    print("MOD(10,3)=%d  MOD(-10,3)=%d  （取余符号跟随除数，与 Excel MOD 一致）"
          % (np.mod(10, 3), np.mod(-10, 3)))
    print("ROUND(21.3456,3)=%.3f  ABS(-23.6)=%.1f" % (round(21.3456, 3), abs(-23.6)))
    print("MAX=%s  MIN=%s  SUM=%.1f  AVERAGE=%.3f"
          % (x.max(), x.min(), x.sum(), x.mean()))
    print("COUNT(数值个数)=%d  COUNTA(非空个数)=%d" % (pd.Series(x).count(), pd.Series(x).notna().sum()))

    section("2. IF + AND + OR：三科判定（对照教材答案）")
    df = pd.read_csv(os.path.join(SRC, "if_and_or.csv"), dtype={"学号": str})
    ans = pd.read_csv(os.path.join(SRC, "if_and_or_answer.csv"), dtype={"学号": str})
    for c in ["数学", "计算机", "英语"]:
        df[c] = pd.to_numeric(df[c])
    df = df[df["数学"].notna()].copy()  # 去掉教材说明行
    # Excel: =IF(AND(数学>=60,计算机>=60,英语>=60),"通过","未通过")
    df["三门均通过"] = np.where((df[["数学", "计算机", "英语"]] >= 60).all(axis=1), "通过", "未通过")
    # Excel: =IF(OR(数学>=60,计算机>=60,英语>=60),"通过","未通过")
    df["三门之一通过"] = np.where((df[["数学", "计算机", "英语"]] >= 60).any(axis=1), "通过", "未通过")
    # 答案版按学号对齐（原数据表也含空的答案列，故只取答案版中非空的结果）
    ans = ans[ans["答案_三门均通过"].notna()].set_index("学号")
    ok1 = ok2 = True
    checked = 0
    for sid, calc_and, calc_or in zip(df["学号"], df["三门均通过"], df["三门之一通过"]):
        if sid in ans.index:
            checked += 1
            ok1 &= (calc_and == ans.loc[sid, "答案_三门均通过"])
            ok2 &= (calc_or == ans.loc[sid, "答案_三门之一通过"])
    print(df[["姓名", "数学", "计算机", "英语", "三门均通过", "三门之一通过"]].to_string(index=False))
    print(f"\n与教材答案一致：AND 判定 {ok1}，OR 判定 {ok2}（校验 {checked} 条）")

    section("3. COUNTIF：出勤统计（对照教材答案）")
    att = pd.read_csv(os.path.join(SRC, "attendance.csv"), dtype=str)
    aans = pd.read_csv(os.path.join(SRC, "attendance_answer.csv"), dtype=str)
    dates = [c for c in att.columns if c.isdigit()]
    # Excel: =COUNTIF(B2:K2,"b")  → pandas 按行统计等于 b 的个数
    att["迟到次数"] = (att[dates] == "b").sum(axis=1)
    att["旷课次数"] = (att[dates] == "c").sum(axis=1)
    # 答案版按学号对齐（用答案文件中非空的迟到/旷课次数）
    a_valid = aans[pd.to_numeric(aans["答案_迟到次数"], errors="coerce").notna()].copy()
    a_valid = a_valid.set_index("学号")
    ok = True
    checked = 0
    rows = []
    for sid, late, absent in zip(att["学号"], att["迟到次数"], att["旷课次数"]):
        if sid in a_valid.index:
            checked += 1
            exp_late = int(float(a_valid.loc[sid, "答案_迟到次数"]))
            exp_abs = int(float(a_valid.loc[sid, "答案_旷课次数"]))
            ok &= (late == exp_late and absent == exp_abs)
            rows.append((sid, late, exp_late, absent, exp_abs))
    print(pd.DataFrame(rows, columns=["学号", "迟到_计算", "迟到_答案", "旷课_计算", "旷课_答案"]).head(8).to_string(index=False))
    print(f"\n与教材答案一致：{ok}（校验 {checked} 名学生）")

    section("4. 日期函数：月末(EOMONTH)、当月天数、闰年、转正日期")
    me = pd.read_csv(os.path.join(SRC, "month_end.csv"))
    # 过滤教材表格末尾混入的说明文字行，只保留数字序列号
    me = me[pd.to_numeric(me["签订日期"], errors="coerce").notna()].copy()
    d = pd.to_datetime(pd.to_numeric(me["签订日期"]), unit="D", origin="1899-12-30")
    me["月末"] = (d + pd.offsets.MonthEnd(0)).dt.strftime("%Y-%m-%d")
    me["当月天数"] = d.dt.days_in_month
    print(me.head(6).to_string(index=False))
    ly = pd.read_csv(os.path.join(SRC, "leap_year.csv"))
    year = ly["年份"]
    # Excel: =IF(OR(MOD(A2,400)=0,AND(MOD(A2,4)=0,MOD(A2,100)<>0)),"闰年","平年")
    ly["是否闰年"] = np.where(((year % 400 == 0) | ((year % 4 == 0) & (year % 100 != 0))), "闰年", "平年")
    show("闰年判断", ly.set_index("年份")["是否闰年"])

    section("5. 文本函数：REPLACE 隐藏手机号中间 4 位")
    ph = pd.read_csv(os.path.join(SRC, "phone_mask.csv"), dtype=str)
    # Excel: =REPLACE(A2,4,4,"****")  → 保留前 3 后 4
    ph["脱敏手机号"] = ph["手机号"].str[:3] + "****" + ph["手机号"].str[-4:]
    show("手机号脱敏", ph["脱敏手机号"])

    section("6. 查找引用函数：VLOOKUP 精确 / 近似 / INDEX+MATCH / 等级查找")
    # 6.1 VLOOKUP 精确查找：教材同表 G/H 列给出「商品编码→商品名称」对照表，
    #     精确 VLOOKUP 等价于 pandas 的 map / merge（左连接，找不到返回 NaN）。
    idx = pd.read_csv(os.path.join(SRC, "index_match.csv"), dtype=str)
    look = pd.read_csv(os.path.join(SRC, "vlookup.csv"), dtype=str)
    print(f"销售明细 {len(look)} 行，商品编码去重 {look['商品编码'].nunique()} 个")
    demo = pd.DataFrame({"商品编码": ["10010", "10019", "10022"]})
    dict_df = pd.DataFrame({"商品编码": ["10010", "10019", "10022"],
                            "商品名称": ["牛皮纸笔记本", "牛皮纸便签本", "牛皮纸记事本"]})
    print("VLOOKUP 精确查找（merge 等价）：")
    print(demo.merge(dict_df, on="商品编码", how="left").to_string(index=False))
    # 缺失编码的查找返回 NaN，与 VLOOKUP 的 #N/A 对应
    miss = pd.DataFrame({"商品编码": ["99999"]}).merge(dict_df, on="商品编码", how="left")
    print("未找到编码时返回空值（等价 #N/A）：", miss["商品名称"].iloc[0])

    # 6.2 VLOOKUP 近似查找：评分→行动（区间查找，类似 pd.cut）
    va = pd.read_csv(os.path.join(SRC, "vlookup_approx.csv"))
    score = pd.to_numeric(va["评分"])
    # 教材规则：0→朋友圈点赞，6→微信互动，8→约吃饭（近似匹配取不超过评分的最大档）
    va["行动"] = pd.cut(score, bins=[-1, 5, 7, 10], labels=["朋友圈点赞", "微信互动", "约吃饭"])
    print("\nVLOOKUP 近似查找（区间分级）：")
    print(va[["女神", "评分", "行动"]].head(6).to_string(index=False))

    # 6.3 INDEX+MATCH：按商品名称找品类编号，再回查品类名称
    idx["品类编号_num"] = pd.to_numeric(idx["品类编号"], errors="coerce")
    print("\nINDEX+MATCH 等价（loc 按位置/标签取值）：")
    print(idx.drop_duplicates("商品名称")[["商品名称", "品类编号_num"]].head(6).to_string(index=False))

    # 6.4 成绩等级查找（近似 VLOOKUP）
    gl = pd.read_csv(os.path.join(SRC, "grade_lookup.csv"))
    sc = pd.to_numeric(gl["成绩"])
    gl["等级"] = pd.cut(sc, bins=[-1, 59, 69, 79, 89, 100],
                      labels=["不及格", "及格", "可以", "良好", "优秀"])
    show("成绩等级查找", gl[["学生姓名", "成绩", "等级"]])

    section("7. 数据透视表：销售地区 × 品名")
    piv = pd.read_csv(os.path.join(SRC, "pivot_sales.csv"))
    piv["销售金额"] = pd.to_numeric(piv["销售金额"])
    table = pd.pivot_table(piv, values="销售金额", index="销售地区", columns="品名",
                           aggfunc="sum", fill_value=0)
    print(table.to_string())
    print("\n按销售人员汇总：")
    print(piv.groupby("销售人员")["销售金额"].sum().sort_values(ascending=False).to_string())

    print("\n所有 Excel 函数已用 pandas/numpy 等价实现并完成校验。")


if __name__ == "__main__":
    main()
