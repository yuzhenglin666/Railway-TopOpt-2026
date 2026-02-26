"""
有限元求解器演示程序（PyQt 界面版）
提供参数输入（杨氏模量、泊松比、压力）并实时计算显示位移云图
作者：C组
版本：1.0
"""

import sys
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import LinearProblem
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QLineEdit, QPushButton, QLabel, QMessageBox)
from pyvistaqt import QtInteractor
import pyvista as pv


class FEMSolver:
    """
    简化版求解器：生成长方体网格，施加载荷，计算位移场
    """
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self._create_mesh()
        self._create_function_space()
        self.params = {'E': 210e9, 'nu': 0.3}
        self.bcs = []
        self.load = None

    def _create_mesh(self):
        """创建长方体网格：长2，宽0.5，高0.5"""
        self.domain = mesh.create_box(
            self.comm,
            [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.5, 0.5])],
            [20, 5, 5],
            cell_type=mesh.CellType.tetrahedron
        )
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

    def _create_function_space(self):
        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (3,)))

    def set_material(self, E, nu):
        self.params['E'] = E
        self.params['nu'] = nu

    def set_fixed_boundary(self):
        """固定左端面 (x=0)"""
        def left(x):
            return np.isclose(x[0], 0.0)
        facets = mesh.locate_entities_boundary(self.domain, self.fdim, left)
        dofs = fem.locate_dofs_topological(self.V, self.fdim, facets)
        bc = fem.dirichletbc(ScalarType((0.0, 0.0, 0.0)), dofs, self.V)
        self.bcs.append(bc)

    def set_pressure_load(self, pressure_value):
        """在右端面施加压力 (沿 -z 方向)"""
        def right(x):
            return np.isclose(x[0], 2.0)
        facets = mesh.locate_entities_boundary(self.domain, self.fdim, right)
        facet_tag = np.full(facets.shape, 1, dtype=np.int32)
        self.facet_marker = mesh.meshtags(self.domain, self.fdim, facets, facet_tag)
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_marker)
        self.load = fem.Constant(self.domain, ScalarType((0.0, 0.0, pressure_value)))

    def solve(self):
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            E = self.params['E']
            nu = self.params['nu']
            mu = E / (2 * (1 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            return 2 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3)

        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.inner(self.load, v) * self.ds(1)

        problem = LinearProblem(
            a, L,
            bcs=self.bcs,
            petsc_options_prefix="demo_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        self.uh = problem.solve()
        return self.uh

    def get_results(self):
        """返回节点坐标和节点位移数组"""
        cells, types, x = plot.vtk_mesh(self.V)
        u_values = self.uh.x.array.reshape(-1, 3)
        return x, u_values


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("有限元求解器 - 参数调整与可视化")
        self.resize(1000, 600)

        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ==================== 左侧控制面板 ====================
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QFormLayout(control_panel)

        # 材料参数
        self.E_edit = QLineEdit("210e9")
        self.nu_edit = QLineEdit("0.3")
        self.pressure_edit = QLineEdit("-1e6")

        control_layout.addRow("杨氏模量 E (Pa):", self.E_edit)
        control_layout.addRow("泊松比 ν:", self.nu_edit)
        control_layout.addRow("压力 (Pa):", self.pressure_edit)

        # 运行按钮
        self.run_btn = QPushButton("运行计算")
        control_layout.addRow(self.run_btn)

        # 状态标签
        self.status_label = QLabel("就绪")
        control_layout.addRow(self.status_label)

        main_layout.addWidget(control_panel)

        # ==================== 右侧可视化区域 ====================
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor, stretch=3)

        # 连接按钮信号
        self.run_btn.clicked.connect(self.run_solver)

    def run_solver(self):
        # 读取输入并验证
        try:
            E = float(self.E_edit.text())
            nu = float(self.nu_edit.text())
            pressure = float(self.pressure_edit.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字！")
            return

        # 更新状态
        self.status_label.setText("计算中...")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()  # 刷新界面

        # 创建求解器并设置参数
        solver = FEMSolver()
        solver.set_material(E, nu)
        solver.set_fixed_boundary()
        solver.set_pressure_load(pressure)

        # 执行求解
        try:
            uh = solver.solve()
        except Exception as e:
            self.status_label.setText("计算失败")
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "求解错误", str(e))
            return

        # 获取结果数据
        coords, disp = solver.get_results()
        disp_mag = np.linalg.norm(disp, axis=1)

        # 创建 PyVista 网格
        cells, types, x = plot.vtk_mesh(solver.V)
        grid = pv.UnstructuredGrid(cells, types, x)
        grid.point_data["Displacement"] = disp
        grid.point_data["|u|"] = disp_mag

        # 更新可视化
        self.plotter.clear()
        self.plotter.add_mesh(grid, scalars="|u|", cmap="viridis", show_edges=True,
                              scalar_bar_args={"title": "位移大小 (m)"})
        self.plotter.show_axes()
        self.plotter.view_isometric()
        self.plotter.reset_camera()

        # 恢复状态
        self.status_label.setText("计算完成")
        self.run_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
