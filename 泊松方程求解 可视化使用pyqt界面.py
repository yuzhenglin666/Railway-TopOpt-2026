import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    """PyQt 主窗口，嵌入 PyVista 渲染器显示结果"""
    def __init__(self, uh, V, parent=None):
        super().__init__(parent)
        self.uh = uh
        self.V = V
        self.setWindowTitle("FEniCSx 求解结果 - 泊松方程")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self.visualize()

    def visualize(self):
        cells, types, x = plot.vtk_mesh(self.V)
        grid = pv.UnstructuredGrid(cells, types, x)

        grid.point_data["u"] = self.uh.x.array.real
        grid.set_active_scalars("u")

        self.plotter.add_mesh(grid, show_edges=True, cmap="viridis",
                              scalar_bar_args={'title': 'u(x,y)'})

        warped = grid.warp_by_scalar()
        self.plotter.add_mesh(warped, opacity=0.5, show_edges=False,
                              cmap="viridis")

        self.plotter.view_xy()
        self.plotter.add_axes()


def solve_poisson():
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (2.0, 1.0)),
        n=(32, 16),
        cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(msh, ("Lagrange", 1))

    facets = mesh.locate_entities_boundary(
        msh,
        dim=1,
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
    )
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    g = ufl.sin(5 * x[0])

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

    # 关键修改：petsc_options_prefix 必须是非空字符串
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()

    return uh, V, msh


def main():
    uh, V, msh = solve_poisson()

    out_folder = Path("out_poisson")
    out_folder.mkdir(exist_ok=True)
    with io.XDMFFile(msh.comm, out_folder / "poisson.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_function(uh)

    if MPI.COMM_WORLD.rank == 0:
        app = QApplication(sys.argv)
        window = MainWindow(uh, V)
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()