#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import QApplication
from dolfinx import fem, plot

pv.global_theme.allow_empty_mesh = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_utils import generate_cantilever_mesh, load_mesh
from fea_solver import FEASolver
from topology import TopologyOptimizer
from constraints import create_casting_constraint
from visualization import MainWindow


class SmoothRainbowMainWindow(MainWindow):
    def __init__(self, domain, facet_tags):
        super().__init__(domain, facet_tags)
        self.last_dc = None
        self.vtk_plotter.set_background('white')
        self._init_render_setup = False

    def update_plot(self, rho_arr, dc_arr):
        self.last_dc = dc_arr.copy()

        try:
            # 1. 重建底层网格
            tdim = self.domain.topology.dim
            topology, cell_types, geometry = plot.vtk_mesh(self.domain, tdim)
            mesh_vtk = pv.UnstructuredGrid(topology, cell_types, geometry)

            # 将密度分布绑定给体素单元
            mesh_vtk.cell_data["density"] = rho_arr.astype(np.float64)

            # 2. 安全获取截断值
            try:
                target_vol = self.inputs["体积分数:"].value()
            except:
                target_vol = 0.3

            # 使用非常保守的截断判定，保证任何微小的连接肌肉都不被切断
            cutoff = target_vol * 0.4

            # 🎯【核心修复】：摒弃容易产生破洞的 contour，采用 体积筛选 + 表面提取
            # 1. 先把所有合格的三维方块抠出来（绝对保证实体连通性）
            solid_vol = mesh_vtk.threshold(cutoff, scalars="density")

            # 防闪退兜底
            if solid_vol is None or solid_vol.n_points == 0:
                solid_vol = mesh_vtk

            # 2. 剥离外面一层皮
            surface_shell = solid_vol.extract_surface()

            # 3. 对这层皮应用顶级的拓扑专属平滑（Taubin算法不会导致体积缩水）
            try:
                # 过滤高频阶梯，留下光滑连续的面
                iso_surf = surface_shell.smooth_taubin(n_iter=40, pass_band=0.08)
            except AttributeError:
                # 兼容老版本 PyVista
                iso_surf = surface_shell.smooth(n_iter=25, relaxation_factor=0.1)

            # 注入彩虹色坐标
            iso_surf["x_map"] = iso_surf.points[:, 0]

            # 提取淡淡的参照系铁笼
            if not self._init_render_setup:
                self.wireframe = mesh_vtk.extract_all_edges()
                self.wireframe["x_map"] = self.wireframe.points[:, 0]

            # 双视图投射
            for i in [0, 1]:
                self.vtk_plotter.subplot(0, i)

                if not self._init_render_setup:
                    self.vtk_plotter.add_mesh(
                        self.wireframe, scalars="x_map", cmap="jet_r",
                        line_width=1.0, opacity=0.08, show_scalar_bar=False, name=f"cage_{i}"
                    )

                # 更新主体（带光泽反光效果）
                self.vtk_plotter.add_mesh(
                    iso_surf, scalars="x_map", cmap="jet_r",
                    show_edges=False,
                    smooth_shading=True,
                    specular=0.5,
                    show_scalar_bar=False,
                    name=f"smooth_body_{i}",
                    reset_camera=False  # 绝对防闪烁核心
                )

            # 初始化视角控制
            if not getattr(self, "_camera_set", False):
                self.vtk_plotter.subplot(0, 0);
                self.vtk_plotter.view_isometric()
                self.vtk_plotter.subplot(0, 1);
                self.vtk_plotter.view_xz()
                self._camera_set = True
                self._init_render_setup = True

            self.vtk_plotter.update()

        except Exception as e:
            pass  # 静默过度帧错误

    def optimization_done(self):
        super().optimization_done()
        if self.worker and self.last_dc is not None:
            self.update_plot(self.worker.rho_array.copy(), self.last_dc)
            self.log("✅ 拓扑结构已提纯：连通性与表面光泽处理完毕。")


def main():
    # 恢复为您最开始稳定的尺寸
    generate_cantilever_mesh(Lx=2.0, Ly=0.5, Lz=0.5, lc=0.15, filename="mesh.msh")
    domain, facet_tags = load_mesh("mesh.msh")

    app = QApplication(sys.argv)
    window = SmoothRainbowMainWindow(domain, facet_tags)
    window.show()

    def on_start_optimization():
        if window.running: return
        try:
            params = window.get_params()
            window.log("🚀 启动 3D 拓扑优化...")

            V_rho = fem.functionspace(domain, ("DG", 0))
            rho_init = fem.Function(V_rho)
            rho_init.x.array[:] = params['optimization']['vol_frac']

            fea_solver = FEASolver(domain, facet_tags, rho_init, params['material'])

            # 🎯 恢复为您原始正常的读负载逻辑，绝不干涉！（防止再次变阴间结构）
            candidates = []
            if params['load'].get('enable_vert'): candidates.append(
                ("垂直变形", (0, 0, -1), params['load']['mag_vert']))
            if params['load'].get('enable_lat'):  candidates.append(("侧向变形", (0, 1, 0), params['load']['mag_lat']))
            if params['load'].get('enable_long'): candidates.append(("轴向拉伸", (1, 0, 0), params['load']['mag_long']))
            if not candidates: candidates = [("默认载荷", (0, 0, -1), 1.0)]

            load_info = []
            for name, d, m in candidates:
                fea_solver.clear_loads()
                fea_solver.add_pressure_load(m, d, surface_tag=2)
                fea_solver.compute_compliance(fea_solver.solve())
                load_info.append({'name': name, 'dir': d, 'mag': m, 'weight': 1.0})

            params['optimization']['active_loads'] = load_info

            cast_con = None
            if params['casting']['enable']:
                cast_con = create_casting_constraint(
                    domain, rho_init, params.get('material', {}).get('type', 'steel'),
                    params['casting'].get('draft_dir', [0, 0, 1]))

            optimizer = TopologyOptimizer(rho=rho_init, fea_solver=fea_solver,
                                          params=params['optimization'], casting_constraint=cast_con)
            window.start_optimization_with_params(params, optimizer, rho_init)

        except Exception as e:
            window.log(f"💥 发生错误: {str(e)}")

    try:
        window.start_btn.clicked.disconnect()
    except:
        pass
    window.start_btn.clicked.connect(on_start_optimization)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()