#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import QApplication
from dolfinx import fem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_utils import generate_cantilever_mesh, load_mesh
from fea_solver import FEASolver
from topology import TopologyOptimizer
from constraints import create_casting_constraint
from visualization import MainWindow


class UltimateMainWindow(MainWindow):
    def __init__(self, domain, facet_tags):
        super().__init__(domain, facet_tags)
        self.last_dc = None

    def update_plot(self, rho_arr, dc_arr):
        self.last_dc = dc_arr.copy()
        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, 0)
        num_cells = self.domain.topology.index_map(tdim).size_local
        cell_Nodes_1d = self.domain.topology.connectivity(tdim, 0).array

        # 完美自适应四面体渲染机制 (根除错位问题)
        nodes_per_cell, vtk_type = 4, pv.CellType.TETRA

        try:
            cell_nodes_2d = cell_Nodes_1d.reshape((num_cells, nodes_per_cell))
            padding = np.full((num_cells, 1), nodes_per_cell, dtype=np.int64)
            cells_vtk = np.hstack([padding, cell_nodes_2d]).flatten()

            # 创建源物理场模型
            mesh_vtk = pv.UnstructuredGrid(cells_vtk, np.full(num_cells, vtk_type, dtype=np.uint8),
                                           self.domain.geometry.x)

            mesh_vtk.cell_data["density"] = rho_arr.astype(np.float64)
            # 分析云图颜色控制：平滑过滤掉奇异畸变极值
            stress_proxy = np.clip(np.abs(dc_arr), 0, np.percentile(np.abs(dc_arr), 98) + 1e-9)
            mesh_vtk.cell_data["Strain Energy"] = stress_proxy

            # ==========================================
            # 🖥️ 左侧：优化结构
            # ==========================================
            self.vtk_plotter.subplot(0, 0)
            self.vtk_plotter.add_mesh(mesh_vtk.outline(), color="gray", name="outline_left")

            # 安全取阈值
            try:
                cutoff_val = max(0.3, self.inputs["体积分数:"].value() * 0.8)
            except:
                cutoff_val = 0.35

            # 挖空显示
            mesh_cut = mesh_vtk.threshold(value=cutoff_val, scalars="density")
            self.vtk_plotter.add_mesh(
                mesh_cut,
                color="#0066cc",
                show_edges=True,  # 稍带网格线看清骨架
                edge_color="#3388cc",
                smooth_shading=True,  # 赋予表面高光
                name="topo_main"
            )

            # ==========================================
            # 🖥️ 右侧：🎯 极其平滑的源长方体云图！
            # ==========================================
            self.vtk_plotter.subplot(0, 1)

            # 🌟 魔法转换：把坚硬的数据块插值为柔和的云彩！
            smooth_mesh = mesh_vtk.cell_data_to_point_data()

            self.vtk_plotter.add_mesh(
                smooth_mesh,
                scalars="Strain Energy",
                cmap="jet",  # 您的图片中使用的彩色主题
                show_edges=False,  # ⛔ 绝对关闭网格线！
                smooth_shading=True,  # ✅ 开启光线平滑着色引擎！
                name="stress_main"
            )

            # 视角同步
            if not getattr(self, "_camera_set", False):
                self.vtk_plotter.link_views()
                self.vtk_plotter.view_isometric()
                self._camera_set = True

            self.vtk_plotter.update()
        except Exception as e:
            print(f"渲染纠错记录: {str(e)}")

    def optimization_done(self):
        super().optimization_done()
        if self.worker and self.last_dc is not None:
            self.update_plot(self.worker.rho_array.copy(), self.last_dc)
            self.log("🎨 一致性云图生成完毕！请查看右侧的完美光滑结构！")


def main():
    # 生成默认底盘网格
    generate_cantilever_mesh(Lx=2.0, Ly=0.5, Lz=0.5, lc=0.15, filename="mesh.msh")
    domain, facet_tags = load_mesh("mesh.msh")

    app = QApplication(sys.argv)
    window = UltimateMainWindow(domain, facet_tags)
    window.show()

    def on_start_optimization():
        if window.running: return
        try:
            params = window.get_params()
            window.log("🌐 启动【多工况综合分析】引流...")

            V_rho = fem.functionspace(domain, ("DG", 0))
            rho_temp = fem.Function(V_rho)
            rho_temp.x.array[:] = params['optimization']['vol_frac']
            fea_eval = FEASolver(domain, facet_tags, rho_temp, params['material'])

            # 环境加载器
            candidates = []
            if params['load'].get('enable_vert'): candidates.append(
                ("垂直变形", (0, 0, -1), params['load']['mag_vert']))
            if params['load'].get('enable_lat'):  candidates.append(("侧向切变", (0, 1, 0), params['load']['mag_lat']))
            if params['load'].get('enable_long'): candidates.append(("轴向拉压", (1, 0, 0), params['load']['mag_long']))
            if not candidates: candidates = [("默认基准", (0, 0, -1), 1.0)]

            active_loads = []
            for name, d, m in candidates:
                fea_eval.clear_loads()
                fea_eval.add_pressure_load(m, d, surface_tag=2)
                c = fea_eval.compute_compliance(fea_eval.solve())
                window.log(f"   ➤ [{name}]: 基础柔度响应 = {c:.4e}")
                active_loads.append({'name': name, 'dir': d, 'mag': m, 'weight': 1.0})

            params['optimization']['active_loads'] = active_loads

            # 最终结算模型
            rho_final = fem.Function(V_rho, name="rho")
            rho_final.x.array[:] = params['optimization']['vol_frac']
            fea_final = FEASolver(domain, facet_tags, rho_final, params['material'])
            if len(active_loads) > 0: fea_final.add_pressure_load(active_loads[0]['mag'], active_loads[0]['dir'],
                                                                  surface_tag=2)

            cast_con = None
            if params['casting']['enable']:
                cast_con = create_casting_constraint(
                    domain, rho_final, params.get('material', {}).get('type', 'steel'),
                    params['casting'].get('draft_dir', [0, 0, 1]))

            optimizer = TopologyOptimizer(rho=rho_final, fea_solver=fea_final,
                                          params=params['optimization'], casting_constraint=cast_con)
            window.start_optimization_with_params(params, optimizer, rho_final)

        except Exception as e:
            window.log(f"💥 运行错误: {str(e)}")

    try:
        window.start_btn.clicked.disconnect()
    except:
        pass
    window.start_btn.clicked.connect(on_start_optimization)
    sys.exit(app.exec_())


if __name__ == "__main__": main()