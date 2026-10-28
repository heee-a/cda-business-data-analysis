"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 模板9-流程挖掘
# ============================================================

# ============================================================
# # 1、导入IEEE XES格式数据
# ============================================================

# --- code cell ---
##导入IEEE XES格式数据
#IEEE XES是一种存储日志数据的标准格式。
from pm4py.objects.log.importer.xes import importer as xes_importer

log = xes_importer.apply('./running-example.xes')
print(log)

# --- code cell ---
print('The first trace in the log: \n', log[0])
print('-' * 80)
print('The first event of the first trace: \n', log[0][0])

# --- code cell ---
log[0:2:2][:3]

# --- code cell ---
print(type(log))
print(type(log[0]))
print(type(log[0][0]))

# --- code cell ---
print('Length of the log is %d' %(len(log)))
print('Length of the first trace is %d' %(len(log[0])))
print('Length of the first event is %d' %(len(log[0][0])))

# --- code cell ---
# 导出XES文件
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
xes_exporter.apply(log, "./exportedLog.xes")

# ============================================================
# # 2、流程发现和流程监控
# ============================================================

# ============================================================
# # Directly-Follows Graph和流程监控的演示
# ============================================================

# --- code cell ---
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization

dfg = dfg_discovery.apply(log)
gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)

# --- code cell ---
#添加流程转换耗时信息
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization

dfg = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)
gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.PERFORMANCE)
dfg_visualization.view(gviz)

# --- code cell ---


# --- code cell ---

