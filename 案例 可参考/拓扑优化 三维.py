#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三维拓扑优化 (Gmsh网格 + FEniCSx计算 + PyQt可视化)
增强版：配色改善（plasma 颜色映射、历史曲线改色等）
"""

import sys
import numpy as np
import gmsh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import ufl
from dolfinx import mesh, fem, plot, io
from dolfinx.fem.petsc import LinearProblem
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QWidget, QPushButton, QLabel, QGridLayout,
                             QSlider)
from PyQt5.QtCore import Qt

# --- 1. 用 Gmsh 生成 网格并定义物理组 (与原文相同) ---
gmsh.initialize()
gmsh.model.add("cantilever_beam")
Lx, Ly, Lz = 2.0, 0.5, 0.5
lc = 0.2
box = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(2)
left_tag = right_tag = None
for surf in surfaces:
    com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
    if np.isclose(com[0], 0.0):
        left_tag = surf[1]
    elif np.isclose(com[0], Lx):
        right_tag = surf[1]
if left_tag is None or right_tag is None:
    raise RuntimeError("无法找到左端面或右端面，请检查几何。")
gmsh.model.addPhysicalGroup(2, [left_tag], tag=1)
gmsh.model.setPhysicalName(2, 1, "fixed")
gmsh.model.addPhysicalGroup(2, [right_tag], tag=2)
gmsh.model.setPhysicalName(2, 2, "load")
volumes = gmsh.model.getEntities(3)
gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=3)
gmsh.model.setPhysicalName(3, 3, "domain")
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
gmsh.model.mesh.generate(3)
gmsh.write("mesh.msh")
gmsh.finalize()

from dolfinx.io import gmsh as gmsh_io
mesh_data = gmsh_io.read_from_msh("mesh.msh", MPI.COMM_WORLD, gdim=3)
domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

# --- 2. FEniCSx 拓扑优化设置 (unchanged) ---
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
V_rho = fem.functionspace(domain, ("DG", 0))
rho = fem.Function(V_rho, name="Density")
vol_frac = 0.3
rho.x.array[:] = vol_frac
E0 = 1.0
Emin = 1e-9
nu = 0.3
p = 3.0
rmin = 0.1
move = 0.2
tol = 1e-2
max_iter = 50
fixed_facets = facet_tags.find(1)
fixed_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, fixed_facets)
zero = fem.Constant(domain, ScalarType((0.0, 0.0, 0.0)))
bc_left = fem.dirichletbc(zero, fixed_dofs, V)
load_facets = facet_tags.find(2)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
area = Ly * Lz
load_mag = 1.0
pressure = -load_mag / area
f_load = fem.Constant(domain, ScalarType((0.0, 0.0, pressure)))
bcs = [bc_left]
def epsilon(u):
    return ufl.sym(ufl.grad(u))
def sigma(u, rho):
    E = Emin + rho**p * (E0 - Emin)
    mu = E / (2*(1+nu))
    lmbda = E*nu / ((1+nu)*(1-2*nu))
    return 2*mu*epsilon(u) + lmbda*ufl.tr(epsilon(u))*ufl.Identity(len(u))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u, rho), epsilon(v)) * ufl.dx
L = ufl.inner(f_load, v) * ds(2)
petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

def compute_compliance(u_h):
    return fem.assemble_scalar(fem.form(ufl.inner(f_load, u_h) * ds(2)))
def compute_energy_density(u_h, rho):
    energy = 0.5 * ufl.inner(sigma(u_h, rho), epsilon(u_h))
    V_dg = fem.functionspace(domain, ("DG", 0))
    expr = fem.Expression(energy, V_dg.element.interpolation_points)
    e_h = fem.Function(V_dg)
    e_h.interpolate(expr)
    return e_h.x.array
def filter_sensitivities(dc, rmin, V_rho):
    coords = V_rho.tabulate_dof_coordinates()
    cell_centers = coords[:, :3]
    N = len(dc)
    dc_f = np.zeros_like(dc)
    for i in range(N):
        dist = np.linalg.norm(cell_centers - cell_centers[i], axis=1)
        mask = dist <= rmin
        w = rmin - dist[mask]
        if w.sum() > 0:
            dc_f[i] = np.sum(w * dc[mask]) / w.sum()
        else:
            dc_f[i] = dc[i]
    return dc_f
def oc_update(rho_arr, dc, vol_frac, move):
    l1, l2 = 0.0, 1e5
    for _ in range(50):
        lmid = 0.5*(l1+l2)
        rho_new = rho_arr * np.sqrt(np.maximum(1e-12, -dc/(lmid+1e-12)))
        rho_new = np.maximum(0.001, np.minimum(0.999, rho_new))
        if np.mean(rho_new) > vol_frac:
            l1 = lmid
        else:
            l2 = lmid
    rho_new = rho_arr * np.sqrt(np.maximum(1e-12, -dc/(lmid+1e-12)))
    rho_new = np.maximum(0.001, np.minimum(0.999, rho_new))
    rho_new = np.maximum(rho_arr - move, np.minimum(rho_arr + move, rho_new))
    return rho_new

# --- 3. PyQt 可视化窗口（配色优化） ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self, rho_func, domain, history):
        super().__init__()
        self.rho = rho_func
        self.domain = domain
        self.history = history
        self.setWindowTitle("拓扑优化实时显示 (配色改进)")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)

        # 三维 plotter
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor, 0, 0, 4, 1)

        # 控件
        self.btn = QPushButton("刷新视图")
        self.btn.clicked.connect(self.update_plot)
        layout.addWidget(self.btn, 0, 1)

        self.label = QLabel("迭代: 初始")
        layout.addWidget(self.label, 1, 1)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(10)
        self.threshold_slider.valueChanged.connect(self.update_plot)
        layout.addWidget(QLabel("密度阈值："), 2, 1)
        layout.addWidget(self.threshold_slider, 3, 1)

        # 历史曲线
        self.fig = Figure(figsize=(5, 3), facecolor="#f0f0f0")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 4, 0, 1, 2)

        self.update_plot()
        self.update_history()

    def update_plot(self):
        self.plotter.clear()
        cells, types, x = plot.vtk_mesh(self.domain, self.domain.topology.dim)
        grid = pv.UnstructuredGrid(cells, types, x)
        grid.cell_data["rho"] = self.rho.x.array

        thresh_val = self.threshold_slider.value() / 100.0
        th = grid.threshold([thresh_val, 1.0], scalars="rho")
        self.plotter.add_mesh(th,
                              scalars="rho",
                              cmap="plasma",          # 改用 plasma 颜色表
                              show_edges=False,
                              opacity="sigmoid_5",
                              scalar_bar_args={"title": "Density",
                                               "title_font_size": 14,
                                               "label_font_size": 10})
        # pyvista's add_axes no longer accepts an 'axis_type' keyword;
        # just add the default axes actor.
        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.canvas.draw()

    def update_history(self):
        self.fig.clf()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        ax1.plot(self.history['compliance'], '-o', color="#003f5c")
        ax1.set_title("Compliance", color="#003f5c")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("C")
        ax1.grid(True, linestyle=":", color="#aaaaaa")
        ax2.plot(self.history['vol'], '-o', color="#bc5090")
        ax2.set_title("Volume fraction", color="#bc5090")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Vol")
        ax2.grid(True, linestyle=":", color="#aaaaaa")
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

# --- 4. 优化主循环 ---
def run_optimization(rho, domain, window):
    history = {'compliance': [], 'vol': [], 'change': []}
    for it in range(max_iter):
        print(f"\n迭代 {it+1}/{max_iter}")
        problem = LinearProblem(
            a, L,
            bcs=bcs,
            petsc_options_prefix="topopt3d_",
            petsc_options=petsc_options
        )
        uh = problem.solve()
        c = compute_compliance(uh)
        history['compliance'].append(c)
        energy = compute_energy_density(uh, rho)
        rho_arr = rho.x.array.copy()
        dc = -p * rho_arr**(p-1) * energy
        if rmin > 0:
            dc = filter_sensitivities(dc, rmin, V_rho)
        rho_new = oc_update(rho_arr, dc, vol_frac, move)
        change = np.max(np.abs(rho_new - rho_arr))
        rho.x.array[:] = rho_new
        current_vol = np.mean(rho.x.array)
        history['vol'].append(current_vol)
        history['change'].append(change)
        print(f"  柔度 = {c:.4e}, 体积 = {current_vol:.4f}, 最大变化 = {change:.4e}")
        window.label.setText(f"迭代: {it+1}")
        if (it+1) % 10 == 0 or it == 0:
            window.update_plot()
            window.update_history()
            QApplication.processEvents()
        if change < tol:
            print("收敛！")
            break
    window.label.setText("优化完成")
    window.update_plot()
    window.update_history()
    return history

# --- 5. 启动应用 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    history = {'compliance': [], 'vol': [], 'change': []}
    window = MainWindow(rho, domain, history)
    window.show()
    history = run_optimization(rho, domain, window)
    with io.XDMFFile(domain.comm, "density_final.xdmf", "w") as file:
        file.write_mesh(domain)
        rho.name = "Density"
        file.write_function(rho)
    np.savez("history.npz",
             compliance=history['compliance'],
             volume=history['vol'],
             change=history['change'])
    print("计算结束，窗口保持打开。关闭窗口退出程序。")
    sys.exit(app.exec_())
