#!/usr/bin/env python3
# fixed_elasticity_viz.py
import os
import numpy as np
import ufl
from dolfinx import mesh, fem, plot, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

# 容错导入 ScalarType（部分环境 petsc4py 可能不可用）
try:
    from petsc4py.PETSc import ScalarType
except Exception:
    ScalarType = lambda x: x  # 回退：直接使用普通 python tuple/数值

# 可选：pyvista，可视化若不可用则写 XDMF
try:
    import pyvista
except Exception:
    pyvista = None

# ----- 1) 网格 -----
L = 2.0
W = 0.2
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([L, W, W])],
    [50, 5, 5],
    cell_type=mesh.CellType.hexahedron,
)

# ----- 2) 函数空间（位移 3 分量） -----
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

# ----- 3) 材料 -----
E = 1e9
nu = 0.3
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3)

# ----- 4) 边界条件 -----
def left_boundary(x):
    return np.isclose(x[0], 0.0)

fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)

# 根据是否成功导入 ScalarType 选择常量构造方式
try:
    zero_const = fem.Constant(domain, ScalarType((0.0, 0.0, 0.0)))
except Exception:
    zero_const = fem.Constant(domain, (0.0, 0.0, 0.0))

bc_left = fem.dirichletbc(zero_const, left_dofs, V)

def right_boundary(x):
    return np.isclose(x[0], L)

right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)
facet_tag = np.full(right_facets.shape, 1, dtype=np.int32)
facet_marker = mesh.meshtags(domain, fdim, right_facets, facet_tag)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_marker)
try:
    traction_value = fem.Constant(domain, ScalarType((0.0, 0.0, -1e6)))
except Exception:
    traction_value = fem.Constant(domain, (0.0, 0.0, -1e6))

# ----- 5) 变分形式 -----
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(traction_value, v) * ds(1)

# ----- 6) 求解 -----
problem = LinearProblem(
    a, L,
    bcs=[bc_left],
    petsc_options_prefix="elasticity_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
uh = problem.solve()

# 整理位移向量 (n_points, 3)
try:
    u_vec = uh.x.array.reshape((-1, 3))
    u_mag = np.linalg.norm(u_vec, axis=1)
    print("求解完成！最大位移：", np.max(u_mag))
except Exception:
    # 若不能 reshape，则只报告范数
    print("求解完成！位移向量长度：", uh.x.array.size)

# ----- 7) 写 XDMF（始终写出，便于 ParaView 查看） -----
out_xdmf = "elasticity_results.xdmf"
with io.XDMFFile(domain.comm, out_xdmf, "w") as file:
    file.write_mesh(domain)
    uh.name = "Displacement"
    file.write_function(uh)
print(f"结果已保存到 {out_xdmf}")

# ----- 8) PyVista 可视化（如果可用且点数匹配则显示并保存截图） -----
if pyvista is None:
    print("pyvista 未安装：可在本机安装 `pip install pyvista` 或在 ParaView 打开 XDMF 查看结果。")
else:
    cells, types, points = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, points)

    # 尝试把位移作为点数据写入
    wrote_point_data = False
    try:
        if uh.x.array.size == points.shape[0] * 3:
            grid.point_data["u"] = uh.x.array.reshape((-1, 3))
            grid.point_data["u_mag"] = np.linalg.norm(uh.x.array.reshape((-1, 3)), axis=1)
            wrote_point_data = True
        else:
            # 有时 DoF 数与 vtk 点数不一致：跳过点数据写入
            print("警告：DoF 数与 VTK 点数不匹配，跳过将位移写入点数据。直接在 ParaView 中打开 XDMF 查看。")
    except Exception:
        print("写入点数据到 PyVista 网格失败，跳过。")

    # 绘图设置：无头模式下保存截图，有头模式显示交互窗口
    off_screen = pyvista.OFF_SCREEN
    plotter = pyvista.Plotter(off_screen=off_screen)
    plotter.set_background("white")

    if wrote_point_data:
        plotter.add_mesh(grid, scalars="u_mag", cmap="plasma", show_scalar_bar=True)
        try:
            warped = grid.warp_by_vector("u", factor=150.0)
            plotter.add_mesh(warped, scalars="u_mag", cmap="plasma", show_scalar_bar=False)
        except Exception:
            pass
        plotter.add_mesh(grid.extract_geometry(), style="wireframe", color="black", opacity=0.15)
    else:
        # 无位移点数据：只显示网格轮廓
        plotter.add_mesh(grid.extract_geometry(), style="wireframe", color="black", opacity=0.9)

    plotter.add_text("Elasticity result", position="upper_left", font_size=10)
    plotter.camera_position = "iso"

    out_png = "elasticity_viz.png"
    if off_screen:
        plotter.show(screenshot=out_png)
        print(f"截图已保存到 {out_png}")
    else:
        plotter.show()