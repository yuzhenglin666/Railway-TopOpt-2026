"""
可视化 FEniCSx 计算结果
读取 result.npz 文件并显示位移云图（实体模式）
"""

import numpy as np
import pyvista as pv

# ==================== 1. 加载结果数据 ====================
print("正在加载 result.npz...")
data = np.load('result.npz')
cells = data['cells']        # 单元连接
types = data['types']        # 单元类型
coords = data['coords']      # 节点坐标
disp = data['disp']          # 节点位移

print(f"节点数: {coords.shape[0]}")
print(f"位移数组形状: {disp.shape}")

# ==================== 2. 重建 PyVista 网格 ====================
grid = pv.UnstructuredGrid(cells, types, coords)
grid.point_data['disp'] = disp
grid.point_data['|u|'] = np.linalg.norm(disp, axis=1)
print("网格重建完成")

# ==================== 3. 可视化（实体模式） ====================
pl = pv.Plotter(window_size=[1000, 800])
pl.add_mesh(grid,
            scalars='|u|',
            cmap='viridis',
            show_edges=False,      # 不显示单元边，呈现实心效果
            lighting=True,          # 启用光照，增强立体感
            scalar_bar_args={'title': '位移大小 (m)'})
pl.show_axes()
pl.view_isometric()
print("正在显示图形窗口...")
pl.show()