import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot, io
from dolfinx.fem.petsc import LinearProblem
import pyvista as pv
import matplotlib.pyplot as plt

# ==================== 参数设置 ====================
Lx, Ly = 2.0, 1.0  # 设计域尺寸
nx, ny = 80, 40  # 网格数
E0 = 1.0  # 杨氏模量（归一化）
Emin = 1e-9  # 最小杨氏模量
nu = 0.3  # 泊松比
p = 3.0  # SIMP惩罚因子
vol_frac = 0.4  # 目标体积分数
rmin = 0.06  # 滤波半径（物理单位）
move = 0.2  # OC更新移动限制
tol = 0.01  # 收敛容差
max_iter = 50  # 最大迭代次数

# ==================== 创建网格 ====================
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (Lx, Ly)),
    n=(nx, ny),
    cell_type=mesh.CellType.triangle
)

# ==================== 函数空间 ====================
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))  # 位移空间
V_rho = fem.functionspace(domain, ("DG", 0))  # 密度空间（每个单元一个值）

# ==================== 初始化设计变量 ====================
rho = fem.Function(V_rho, name="Density")
rho.x.array[:] = vol_frac  # 初始均匀分布


# ==================== 边界条件 ====================
def left_boundary(x):
    return np.isclose(x[0], 0.0)


tdim = domain.topology.dim
fdim = tdim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
zero_vector = fem.Constant(domain, ScalarType((0.0, 0.0)))
bc = fem.dirichletbc(zero_vector, left_dofs, V)


# 载荷：整个右边界上的均布压力（向下）
def right_boundary(x):
    return np.isclose(x[0], Lx)


right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)
# 创建一个标记，用于定义边界积分（这里只用一个标记，标记为1）
facet_tag = np.full(right_facets.shape, 1, dtype=np.int32)
facet_tags = mesh.meshtags(domain, fdim, right_facets, facet_tag)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# 压力大小（总力大小 = 压力 × 边界长度，这里直接给压力值，使总力约为1）
pressure = -1.0 / Ly  # 负号表示向下，除以Ly使总力约等于1
f_load = fem.Constant(domain, ScalarType((0.0, pressure)))


# ==================== 变分形式（依赖于密度） ====================
def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, rho):
    E = Emin + rho ** p * (E0 - Emin)
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(sigma(u, rho), epsilon(v)) * ufl.dx
L = ufl.inner(f_load, v) * ds(1)  # 在标记为1的右边界上施加压力

# ==================== 求解器设置 ====================
petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}


# ==================== 辅助函数 ====================
def compute_compliance(u_h):
    """计算柔度（目标函数）"""
    return fem.assemble_scalar(fem.form(ufl.inner(f_load, u_h) * ds(1)))


def compute_energy_density(u_h, rho):
    """计算每个单元的应变能（用于灵敏度）"""
    energy = 0.5 * ufl.inner(sigma(u_h, rho), epsilon(u_h))
    V_dg = fem.functionspace(domain, ("DG", 0))
    energy_expr = fem.Expression(energy, V_dg.element.interpolation_points)
    energy_h = fem.Function(V_dg)
    energy_h.interpolate(energy_expr)
    return energy_h.x.array


def filter_sensitivities(dc, rmin, V_rho):
    """
    灵敏度滤波（基于单元中心的距离平均）
    参数：
        dc : array, 原始灵敏度
        rmin : 滤波半径（物理单位）
        V_rho : 密度函数空间，用于获取单元中心坐标
    返回：滤波后的灵敏度数组
    """
    coords = V_rho.tabulate_dof_coordinates()
    cell_centers = coords[:, :2]
    num_cells = len(dc)

    dc_filtered = np.zeros_like(dc)
    for i in range(num_cells):
        distances = np.linalg.norm(cell_centers - cell_centers[i], axis=1)
        mask = distances < rmin
        weights = rmin - distances[mask]
        if weights.sum() > 0:
            dc_filtered[i] = np.sum(weights * dc[mask]) / (weights.sum() + 1e-12)
        else:
            dc_filtered[i] = dc[i]
    return dc_filtered


def oc_update(rho, dc, vol_frac, move):
    """
    最优准则法更新
    """
    l1, l2 = 0, 1e5
    # 二分法求拉格朗日乘子
    for _ in range(50):
        lmid = 0.5 * (l1 + l2)
        rho_new = rho * np.sqrt(-dc / (lmid + 1e-12))
        rho_new = np.maximum(0.001, np.minimum(0.999, rho_new))
        if np.mean(rho_new) > vol_frac:
            l1 = lmid
        else:
            l2 = lmid
    rho_new = rho * np.sqrt(-dc / (lmid + 1e-12))
    rho_new = np.maximum(0.001, np.minimum(0.999, rho_new))
    # 应用移动限制
    rho_new = np.maximum(rho - move, np.minimum(rho + move, rho_new))
    return rho_new


# ==================== 主优化循环 ====================
history = {'compliance': [], 'vol': [], 'change': []}
for it in range(max_iter):
    print(f"\n迭代 {it + 1}/{max_iter}")

    # 1. 求解位移
    problem = LinearProblem(
        a, L,
        bcs=[bc],
        petsc_options_prefix="topopt_",  # 必须非空
        petsc_options=petsc_options
    )
    uh = problem.solve()

    # 2. 计算柔度
    c = compute_compliance(uh)
    history['compliance'].append(c)
    print(f"  柔度 = {c:.4e}")

    # 3. 计算单元应变能
    energy = compute_energy_density(uh, rho)

    # 4. 计算灵敏度
    rho_arr = rho.x.array.copy()
    dc = -p * rho_arr ** (p - 1) * energy

    # 5. 滤波（传递 V_rho）
    if rmin > 0:
        dc = filter_sensitivities(dc, rmin, V_rho)

    # 6. 更新密度
    rho_new_arr = oc_update(rho_arr, dc, vol_frac, move)
    change = np.max(np.abs(rho_new_arr - rho_arr))
    rho.x.array[:] = rho_new_arr
    history['change'].append(change)
    current_vol = np.mean(rho.x.array)
    history['vol'].append(current_vol)
    print(f"  体积分数 = {current_vol:.4f}, 最大变化 = {change:.4e}")

    # 7. 检查收敛
    if change < tol:
        print("收敛！")
        break

    # 8. 每10步可视化当前密度
    if (it + 1) % 10 == 0:
        # 正确做法：获取网格的 VTK 拓扑，附加单元数据
        cells, types, x = plot.vtk_mesh(domain, tdim)  # 修改：使用 domain 和 tdim
        grid = pv.UnstructuredGrid(cells, types, x)
        grid.cell_data["rho"] = rho.x.array  # 单元数据
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid, show_edges=False, cmap="gray")
        plotter.view_xy()
        plotter.screenshot(f"rho_iter_{it + 1}.png")
        print(f"  已保存密度云图 rho_iter_{it + 1}.png")

print("\n优化完成！")

# ==================== 最终可视化 ====================
cells, types, x = plot.vtk_mesh(domain, tdim)  # 修改：使用 domain 和 tdim
grid = pv.UnstructuredGrid(cells, types, x)
grid.cell_data["rho"] = rho.x.array
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=False, cmap="gray", scalar_bar_args={"title": "Density"})
plotter.view_xy()
plotter.show()

# 保存结果
with io.XDMFFile(domain.comm, "final_density.xdmf", "w") as file:
    file.write_mesh(domain)
    rho.name = "Density"
    file.write_function(rho)

# 绘制收敛历史
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history['compliance'], 'b-')
plt.xlabel('Iteration')
plt.ylabel('Compliance')
plt.title('Compliance History')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['vol'], 'r-', label='Volume')
plt.axhline(y=vol_frac, color='k', linestyle='--', label='Target')
plt.xlabel('Iteration')
plt.ylabel('Volume fraction')
plt.title('Volume History')
plt.legend()
plt.grid(True)
plt.savefig("convergence.png")
print("收敛历史已保存到 convergence.png")