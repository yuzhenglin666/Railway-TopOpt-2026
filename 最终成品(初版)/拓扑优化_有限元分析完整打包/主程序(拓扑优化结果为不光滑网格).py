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

        if "tetrahedron" in str(self.domain.topology.cell_type):
            nodes_per_cell, vtk_type = 4, pv.CellType.TETRA
        else:
            nodes_per_cell, vtk_type = 8, pv.CellType.HEXAHEDRON

        try:
            cell_nodes_2d = cell_Nodes_1d.reshape((num_cells, nodes_per_cell))
            padding = np.full((num_cells, 1), nodes_per_cell, dtype=np.int64)
            cells_vtk = np.hstack([padding, cell_nodes_2d]).flatten()
            mesh_vtk = pv.UnstructuredGrid(cells_vtk, np.full(num_cells, vtk_type, dtype=np.uint8),
                                           self.domain.geometry.x)

            mesh_vtk.cell_data["density"] = rho_arr.astype(np.float64)
            stress_proxy = np.clip(np.abs(dc_arr), 0, np.percentile(np.abs(dc_arr), 98) + 1e-9)
            mesh_vtk.cell_data["Strain Energy"] = stress_proxy

            self.vtk_plotter.subplot(0, 0)
            mesh_cut = mesh_vtk.threshold(value=max(0.1, self.inputs["体积分数:"].value() * 0.45), scalars="density")
            if mesh_cut.n_cells > 0:
                self.vtk_plotter.add_mesh(mesh_cut, color="#0066cc", show_edges=True, edge_color="#333333",
                                          name="topo_main")
            self.vtk_plotter.add_mesh(mesh_vtk.outline(), color="gray", name="outline")

            self.vtk_plotter.subplot(0, 1)
            smooth_mesh = mesh_vtk.cell_data_to_point_data()
            self.vtk_plotter.add_mesh(smooth_mesh, scalars="Strain Energy", cmap="jet",
                                      show_edges=False, name="stress_main")

            if not getattr(self, "_camera_set", False):
                self.vtk_plotter.link_views();
                self.vtk_plotter.view_isometric();
                self._camera_set = True
            self.vtk_plotter.update()
        except:
            pass

    def optimization_done(self):
        super().optimization_done()
        if self.worker and self.last_dc is not None:
            self.update_plot(self.worker.rho_array.copy(), self.last_dc)
            self.log("🎨 全工况综合拓扑与平滑渲染已全部完成！")


def main():
    generate_cantilever_mesh(Lx=2.0, Ly=0.5, Lz=0.5, lc=0.2, filename="mesh.msh")
    domain, facet_tags = load_mesh("mesh.msh")
    app = QApplication(sys.argv)
    window = UltimateMainWindow(domain, facet_tags)
    window.show()

    def on_start_optimization():
        if window.running: return
        try:
            params = window.get_params()
            window.log("🌐 ===========================================")
            window.log("🌐 启动【多工况综合抗弯扭】探测系统...")

            V_rho = fem.functionspace(domain, ("DG", 0))
            rho_temp = fem.Function(V_rho)
            rho_temp.x.array[:] = params['optimization']['vol_frac']
            fea_eval = FEASolver(domain, facet_tags, rho_temp, params['material'])

            candidates = []
            if params['load'].get('enable_vert'): candidates.append(
                ("垂直变形(Z向)", (0, 0, -1), params['load']['mag_vert']))
            if params['load'].get('enable_lat'):  candidates.append(
                ("侧向切变(Y向)", (0, 1, 0), params['load']['mag_lat']))
            if params['load'].get('enable_long'): candidates.append(
                ("轴向拉压(X向)", (1, 0, 0), params['load']['mag_long']))
            if not candidates: candidates = [("默认基准(Z向)", (0, 0, -1), 1.0)]

            active_loads = []
            for name, d, m in candidates:
                fea_eval.clear_loads()
                fea_eval.add_pressure_load(m, d, surface_tag=2)
                c = fea_eval.compute_compliance(fea_eval.solve())
                window.log(f"   ➤ 记录工况 [{name}]: 基础结构柔度 = {c:.4e}")
                active_loads.append({'name': name, 'dir': d, 'mag': m, 'weight': 1.0})

            window.log(f"⚠️ 结论：已激活 {len(active_loads)} 个工况边界，开始进行全工况联合优化！")
            window.log("🌐 ===========================================")

            params['optimization']['active_loads'] = active_loads

            rho_final = fem.Function(V_rho, name="rho")
            rho_final.x.array[:] = params['optimization']['vol_frac']
            fea_final = FEASolver(domain, facet_tags, rho_final, params['material'])

            # 💡 致命的单行修复：必须装配默认占位受力，避免 Worker 线程启动时引发 form(int(0)) 的 FEniCS 崩溃
            if len(active_loads) > 0:
                fea_final.add_pressure_load(active_loads[0]['mag'], active_loads[0]['dir'], surface_tag=2)

            cast_con = None
            if params['casting']['enable']:
                cast_con = create_casting_constraint(
                    domain, rho_final,
                    material_type=params.get('material', {}).get('type', 'steel'),
                    draft_dir=params['casting'].get('draft_dir', [0, 0, 1])
                )

            optimizer = TopologyOptimizer(rho=rho_final, fea_solver=fea_final,
                                          params=params['optimization'], casting_constraint=cast_con)
            window.start_optimization_with_params(params, optimizer, rho_final)

        except Exception as e:
            window.log(f"💥 运行崩溃: {str(e)}");
            import traceback;
            traceback.print_exc()

    try:
        window.start_btn.clicked.disconnect()
    except:
        pass
    window.start_btn.clicked.connect(on_start_optimization)
    sys.exit(app.exec_())


if __name__ == "__main__": main()