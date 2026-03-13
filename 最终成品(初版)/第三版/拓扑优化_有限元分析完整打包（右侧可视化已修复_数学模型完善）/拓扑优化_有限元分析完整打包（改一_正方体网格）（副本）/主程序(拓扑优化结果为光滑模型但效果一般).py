#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
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


class DualViewMainWindow(MainWindow):
    def __init__(self, domain, facet_tags):
        super().__init__(domain, facet_tags)
        self.vtk_plotter.set_background('white')
        self._cam_init = False
        self.fea_ref = None

    def update_plot(self, rho_arr, dc_arr):
        try:
            tdim = self.domain.topology.dim
            topology, cell_types, geometry = plot.vtk_mesh(self.domain, tdim)

            base_mesh = pv.UnstructuredGrid(topology, cell_types, geometry)
            topo_mesh = base_mesh.copy(deep=True)
            stress_mesh = base_mesh.copy(deep=True)

            # ==========================================
            # ⬅️ [左视口：结构密度场网状剥离渲染]
            # ==========================================
            self.vtk_plotter.subplot(0, 0)
            topo_mesh.cell_data["density"] = rho_arr.astype(np.float64)

            # 由于开启了铸造约束以及体积提升，采用 0.20 可保留构架细节连廊
            solid = topo_mesh.threshold(0.20, scalars="density")

            if solid is not None and solid.n_points > 0:
                try:
                    surf = solid.extract_surface(algorithm='dataset_surface')
                except TypeError:
                    surf = solid.extract_surface()

                try:
                    final_surf = surf.smooth_taubin(n_iter=15, pass_band=0.1)
                except AttributeError:
                    final_surf = surf.smooth(n_iter=15)

                if final_surf.n_points > 0:
                    final_surf["x_coordinate"] = final_surf.points[:, 0]
                    self.vtk_plotter.add_mesh(final_surf, scalars="x_coordinate", cmap="jet_r", smooth_shading=True,
                                              specular=0.6, show_scalar_bar=False, name="left_topo_mesh",
                                              reset_camera=False)
            else:
                self.vtk_plotter.clear_actors()

            # ==========================================
            # ➡️ [右视口：丝滑流畅应力云图]
            # ==========================================
            self.vtk_plotter.subplot(0, 1)
            stress = self.fea_ref.get_von_mises() if self.fea_ref else None

            if stress is not None:
                stress_max = float(np.percentile(stress, 99.5))
                stress_clipped = np.clip(stress, 0, stress_max) if stress_max > 0 else stress
                stress_mesh.cell_data["Strain_Energy"] = stress_clipped.astype(np.float64)

                smooth_stress_mesh = stress_mesh.cell_data_to_point_data()
                self.vtk_plotter.add_mesh(smooth_stress_mesh, scalars="Strain_Energy", cmap="turbo", show_edges=False,
                                          smooth_shading=True, scalar_bar_args={"title": "von Mises Stress"},
                                          name="right_stress_mesh", reset_camera=False)
            else:
                self.vtk_plotter.add_mesh(stress_mesh, color="whitesmoke", opacity=0.3, show_edges=True,
                                          edge_color="lightgray", name="right_stress_mesh_empty", reset_camera=False)

            if not self._cam_init:
                for i in [0, 1]:
                    self.vtk_plotter.subplot(0, i)
                    self.vtk_plotter.reset_camera()
                    self.vtk_plotter.view_isometric()
                self.vtk_plotter.link_views()
                self._cam_init = True

            self.vtk_plotter.update()

        except Exception as e:
            pass


def main():
    generate_cantilever_mesh(Lx=2.0, Ly=0.5, Lz=0.5, lc=0.15, filename="mesh.msh")
    domain, facet_tags = load_mesh("mesh.msh")
    app = QApplication(sys.argv)
    window = DualViewMainWindow(domain, facet_tags)
    window.show()

    def on_run():
        if window.running: return
        try:
            params = window.get_params()
            window.log("⚡ 启动载入单元模型与外设载荷...")

            V_rho = fem.functionspace(domain, ("DG", 0))
            rho = fem.Function(V_rho, name="Density")
            rho.x.array[:] = params['optimization']['vol_frac']

            fea = FEASolver(domain, facet_tags, rho, params['material'])
            window.fea_ref = fea

            load_list = []
            if params['load'].get('enable_vert'): load_list.append(
                {'dir': (0, 0, -1), 'mag': params['load']['mag_vert'], 'weight': 1.0})
            if params['load'].get('enable_lat'): load_list.append(
                {'dir': (0, 1, 0), 'mag': params['load']['mag_lat'], 'weight': 1.0})
            if params['load'].get('enable_long'): load_list.append(
                {'dir': (1, 0, 0), 'mag': params['load']['mag_long'], 'weight': 1.0})

            if not load_list: load_list = [{'dir': (0, 0, -1), 'mag': 1.0, 'weight': 1.0}]

            params['optimization']['active_loads'] = load_list

            cast_con = None
            if params['casting'].get('enable', False):
                cast_con = create_casting_constraint(
                    domain, rho,
                    material_type=params['casting'].get('material', 'steel'),
                    draft_dir=params['casting'].get('draft_direction', [0, 0, 1])
                )
                window.log(f"✅ 成功挂载拔模约束惩罚场！方向: {params['casting'].get('draft_direction')}")

            opt = TopologyOptimizer(rho=rho, fea_solver=fea, params=params['optimization'], casting_constraint=cast_con)
            window.start_optimization_with_params(params, opt, rho)

        except Exception as e:
            window.log(f"💥 系统排查错误: {str(e)}")
            import traceback
            traceback.print_exc()

    if hasattr(window, "start_btn"):
        try:
            window.start_btn.clicked.disconnect()
        except:
            pass
        window.start_btn.clicked.connect(on_run)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()