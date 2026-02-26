"""
简化版 FEniCSx + Gmsh + PyQt5 + PyVista 演示程序

特性：
* 3D 长方体悬臂梁
* 使用 gmsh 生成四面体网格（有 gmsh 时）
* PyQt5 界面调整几何/网格/材料/载荷参数
* 求解线弹性问题并在 PyVista 中显示位移云图
"""

import sys
import os
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# 尽量导入 gmsh；如果失败将在运行时退回到内置网格
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    gmsh = None
    GMSH_AVAILABLE = False

from dolfinx import mesh, fem, plot
# newer dolfinx versions expose "gmsh" helper, older ones use "gmshio"
try:
    from dolfinx.io import gmsh as dolfinx_gmsh
except ImportError:  # fallback for older releases
    from dolfinx.io import gmshio as dolfinx_gmsh

from dolfinx.fem.petsc import LinearProblem

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox
)
from pyvistaqt import QtInteractor
import pyvista as pv



class FEMSolver:
    def __init__(self, params):
        self.comm = MPI.COMM_WORLD
        self.params = params
        self.domain = None
        self._prepare_mesh()

    def _prepare_mesh(self):
        L = self.params['length']
        W = self.params['width']
        H = self.params['height']
        nx = self.params['nx']
        ny = self.params['ny']
        nz = self.params['nz']
        use_gmsh = self.params.get('use_gmsh', True) and GMSH_AVAILABLE

        if use_gmsh:
            msh_filename = "beam3d.msh"
            try:
                gmsh.initialize()
                gmsh.model.add("beam")
                # 直接用Box几何
                box_tag = gmsh.model.occ.addBox(0, 0, 0, L, W, H)
                gmsh.model.occ.synchronize()

                # 创建物理组: 体积和两个端面，以便 dolfinx 能识别边界
                vols = gmsh.model.getEntities(3)
                if vols:
                    gmsh.model.addPhysicalGroup(3, [vols[0][1]], tag=1)
                # 端面物理组 (左 2 右)
                surfs = gmsh.model.getEntities(2)
                for surf in surfs:
                    com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
                    if abs(com[0]) < 1e-6:          # 左端面 (x=0)
                        gmsh.model.addPhysicalGroup(2, [surf[1]], tag=2)
                    elif abs(com[0] - L) < 1e-6:     # 右端面 (x=L)
                        gmsh.model.addPhysicalGroup(2, [surf[1]], tag=3)

                mesh_size = self.params.get('mesh_size', min(L, W, H) / max(nx, ny, nz))
                gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
                gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
                gmsh.model.mesh.generate(3)
                gmsh.write(msh_filename)
            finally:
                gmsh.finalize()
            if os.path.exists(msh_filename):
                # read mesh with whichever interface we imported earlier
                res = dolfinx_gmsh.read_from_msh(msh_filename, self.comm, gdim=3)
                # support different dolfinx versions that return variable tuples
                if isinstance(res, (tuple, list)):
                    self.domain = res[0]
                    # optional tags if provided
                    self.cell_tags = res[1] if len(res) > 1 else None
                    self.facet_tags = res[2] if len(res) > 2 else None
                else:
                    self.domain = res
                try:
                    os.remove(msh_filename)
                except OSError:
                    pass
            else:
                use_gmsh = False

        if not use_gmsh:
            # gmsh 不可用时使用 dolfinx 内置网格
            self.domain = mesh.create_box(
                self.comm,
                [np.array([0.0, 0.0, 0.0]), np.array([L, W, H])],
                [nx, ny, nz],
                cell_type=mesh.CellType.tetrahedron,
            )
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

    def solve(self):
        # 向量函数空间（兼容不同 dolfinx/ufl 版本）
        V = fem.functionspace(self.domain, ("Lagrange", 1, (3,)))
        self.V = V

        # 左端固定
        def left(x):
            return np.isclose(x[0], 0.0)

        facets = mesh.locate_entities_boundary(self.domain, self.fdim, left)
        dofs = fem.locate_dofs_topological(V, self.fdim, facets)
        bc = fem.dirichletbc(ScalarType((0.0, 0.0, 0.0)), dofs, V)

        # 右端加载
        def right(x):
            return np.isclose(x[0], self.params['length'])

        facets_r = mesh.locate_entities_boundary(self.domain, self.fdim, right)
        facet_tag = np.full(facets_r.shape, 1, dtype=np.int32)
        self.facet_marker = mesh.meshtags(self.domain, self.fdim, facets_r, facet_tag)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_marker)
        load_vec = ScalarType((0.0, 0.0, -self.params['load']))
        T = fem.Constant(self.domain, load_vec)

        # 弹性常数
        E = self.params['youngs_modulus']
        nu = self.params['poisson_ratio']
        mu = E / (2.0 * (1.0 + nu))
        lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.inner(T, v) * ds(1)

        problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="demo_")
        self.uh = problem.solve()
        return self.uh

    def get_results(self):
        cells, types, x = plot.vtk_mesh(self.V)
        u_values = self.uh.x.array.reshape(-1, 3)
        return x, u_values


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D FEM 可视化演示")
        self.resize(1000, 700)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 控制面板
        params_group = QGroupBox("参数设置")
        p_layout = QVBoxLayout()
        params_group.setLayout(p_layout)

        # 几何
        geom_row = QWidget()
        geom_layout = QHBoxLayout(); geom_layout.setContentsMargins(0,0,0,0)
        geom_row.setLayout(geom_layout)
        geom_layout.addWidget(QLabel("长 L:"))
        self.length_spin = QDoubleSpinBox(); self.length_spin.setRange(0.1,100.0); self.length_spin.setValue(2.0);
        self.length_spin.setSingleStep(0.1); geom_layout.addWidget(self.length_spin)
        geom_layout.addWidget(QLabel("宽 W:"))
        self.width_spin = QDoubleSpinBox(); self.width_spin.setRange(0.1,50.0); self.width_spin.setValue(0.5);
        self.width_spin.setSingleStep(0.1); geom_layout.addWidget(self.width_spin)
        geom_layout.addWidget(QLabel("高 H:"))
        self.height_spin = QDoubleSpinBox(); self.height_spin.setRange(0.1,50.0); self.height_spin.setValue(0.5);
        self.height_spin.setSingleStep(0.1); geom_layout.addWidget(self.height_spin)
        p_layout.addWidget(geom_row)

        # 网格
        mesh_row = QWidget()
        mesh_layout = QHBoxLayout(); mesh_layout.setContentsMargins(0,0,0,0)
        mesh_row.setLayout(mesh_layout)
        mesh_layout.addWidget(QLabel("nx:"))
        self.nx_spin = QSpinBox(); self.nx_spin.setRange(2,100); self.nx_spin.setValue(20);
        mesh_layout.addWidget(self.nx_spin)
        mesh_layout.addWidget(QLabel("ny:"))
        self.ny_spin = QSpinBox(); self.ny_spin.setRange(2,100); self.ny_spin.setValue(5);
        mesh_layout.addWidget(self.ny_spin)
        mesh_layout.addWidget(QLabel("nz:"))
        self.nz_spin = QSpinBox(); self.nz_spin.setRange(2,100); self.nz_spin.setValue(5);
        mesh_layout.addWidget(self.nz_spin)
        p_layout.addWidget(mesh_row)

        self.gmsh_cb = QCheckBox("使用 gmsh 网格"); self.gmsh_cb.setChecked(True)
        p_layout.addWidget(self.gmsh_cb)

        # 材料与载荷
        mat_row = QWidget(); mat_layout = QHBoxLayout(); mat_layout.setContentsMargins(0,0,0,0)
        mat_row.setLayout(mat_layout)
        mat_layout.addWidget(QLabel("Young's E:"))
        self.E_spin = QDoubleSpinBox(); self.E_spin.setRange(1e3,1e12); self.E_spin.setValue(210e9);
        mat_layout.addWidget(self.E_spin)
        mat_layout.addWidget(QLabel("Poisson ν:"))
        self.nu_spin = QDoubleSpinBox(); self.nu_spin.setRange(0.0,0.49); self.nu_spin.setValue(0.3);
        mat_layout.addWidget(self.nu_spin)
        mat_layout.addWidget(QLabel("压力 (N/m²):"))
        self.load_spin = QDoubleSpinBox(); self.load_spin.setRange(0.0,1e8); self.load_spin.setValue(1e6);
        mat_layout.addWidget(self.load_spin)
        p_layout.addWidget(mat_row)

        layout.addWidget(params_group)

        self.run_button = QPushButton("运行求解")
        self.run_button.clicked.connect(self.run_solver)
        layout.addWidget(self.run_button)

        self.status_label = QLabel("等待操作")
        layout.addWidget(self.status_label)

        # PyVista 交互器
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

    def run_solver(self):
        self.status_label.setText("计算中...")
        QApplication.processEvents()
        params = {
            'length': self.length_spin.value(),
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'nx': self.nx_spin.value(),
            'ny': self.ny_spin.value(),
            'nz': self.nz_spin.value(),
            'youngs_modulus': self.E_spin.value(),
            'poisson_ratio': self.nu_spin.value(),
            'load': self.load_spin.value(),
            'use_gmsh': self.gmsh_cb.isChecked(),
        }
        solver = FEMSolver(params)
        uh = solver.solve()
        coords, disp = solver.get_results()
        disp_mag = np.linalg.norm(disp, axis=1)

        cells, types, x = plot.vtk_mesh(solver.V)
        grid = pv.UnstructuredGrid(cells, types, x)
        grid.point_data["Displacement"] = disp
        grid.point_data["|u|"] = disp_mag

        # for visualization we only need the surface of the volume mesh
        surf = grid.extract_surface(algorithm='dataset_surface').clean()
        # map data from original points via vtkOriginalPointIds
        if "vtkOriginalPointIds" in surf.point_data:
            orig_ids = surf.point_data["vtkOriginalPointIds"].astype(int)
            if "Displacement" in grid.point_data:
                surf.point_data["Displacement"] = grid.point_data["Displacement"][orig_ids]
            if "|u|" in grid.point_data:
                surf.point_data["|u|"] = grid.point_data["|u|"][orig_ids]
        else:
            surf.point_data["Displacement"] = grid.point_data.get("Displacement", None)
            surf.point_data["|u|"] = grid.point_data.get("|u|", None)

        self.plotter.clear()
        # plot the surface so that all six faces of the cuboid are visible
        self.plotter.add_mesh(surf,
                              scalars="|u|",
                              cmap="viridis",
                              show_edges=True,
                              backface_culling=False,
                              scalar_bar_args={"title": "|u| (m)"})
        self.plotter.show_axes()
        # place the camera in an isometric view outside the object
        self.plotter.view_isometric()
        # ensure the view bounds are reset to include the whole mesh
        self.plotter.reset_camera()
        self.status_label.setText("完成")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
