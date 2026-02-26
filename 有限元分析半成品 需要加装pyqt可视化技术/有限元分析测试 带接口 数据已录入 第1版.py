"""
FEniCSx 有限元求解器模块
用于拓扑优化项目中的结构分析
设计为与 PyQt 前端及可视化组对接

作者：C组
版本：1.1
日期：2026-02-26
"""

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, io, plot
from dolfinx.fem.petsc import LinearProblem
import os


class FEMSolver:
    """
    有限元求解器类
    封装了网格加载、材料定义、边界条件施加、求解和结果输出
    """

    def __init__(self, mesh_file):
        """
        初始化求解器，加载网格文件

        参数:
            mesh_file (str): Gmsh生成的 .msh 文件路径（必须包含物理组标记）
        """
        self.comm = MPI.COMM_WORLD
        self._load_mesh(mesh_file)
        self._create_function_space()
        self._init_parameters()

    def _load_mesh(self, mesh_file):
        """
        从 .msh 文件加载网格，并提取边界标记（facet_tags）
        注意：网格必须包含物理组，否则无法正确施加边界条件
        """
        from dolfinx.io import gmsh  # 新版导入路径
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"网格文件不存在: {mesh_file}")
        mesh_data = gmsh.read_from_msh(mesh_file, self.comm, gdim=3)
        self.domain = mesh_data.mesh               # 网格对象
        self.cell_tags = mesh_data.cell_tags       # 单元标记（可能用于材料分区）
        self.facet_tags = mesh_data.facet_tags     # 面标记，用于边界条件
        # 获取维度信息
        self.tdim = self.domain.topology.dim        # 拓扑维度（3）
        self.fdim = self.tdim - 1                    # 面维度（2）

    def _create_function_space(self):
        """创建位移函数空间（一阶拉格朗日向量单元，三维）"""
        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (3,)))

    def _init_parameters(self):
        """初始化材料参数和存储容器"""
        self.params = {
            'E': 210e9,      # 杨氏模量 (Pa)，默认值，可通过 set_material 修改
            'nu': 0.3,       # 泊松比
        }
        self.bcs = []        # 狄利克雷边界条件列表
        self.load_terms = [] # 面力载荷项列表，每个元素为 (traction_vector, marker)
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)  # 边界积分测度

    # ==================== 前端接口（供 UI 调用） ====================

    def set_material(self, E, nu):
        """
        设置材料参数（由 UI 传入）

        参数:
            E (float): 杨氏模量，单位 Pa
            nu (float): 泊松比
        """
        self.params['E'] = E
        self.params['nu'] = nu

    def set_fixed_boundary(self, markers):
        """
        设置固定边界（位移为 0）

        参数:
            markers (list or int): 物理组标记（单个整数或列表），这些边界上的所有位移分量被固定为 0
        """
        if isinstance(markers, int):
            markers = [markers]
        for marker in markers:
            facets = self.facet_tags.find(marker)
            dofs = fem.locate_dofs_topological(self.V, self.fdim, facets)
            bc = fem.dirichletbc(ScalarType((0.0, 0.0, 0.0)), dofs, self.V)
            self.bcs.append(bc)

    def set_prescribed_displacement(self, marker, disp_vector):
        """
        设置强制位移（例如扭曲工况）

        参数:
            marker (int): 物理组标记
            disp_vector (tuple of 3 floats): 位移值 (ux, uy, uz)，单位 m
        """
        facets = self.facet_tags.find(marker)
        dofs = fem.locate_dofs_topological(self.V, self.fdim, facets)
        bc = fem.dirichletbc(ScalarType(disp_vector), dofs, self.V)
        self.bcs.append(bc)

    def set_surface_load(self, marker, traction_vector):
        """
        设置面力载荷（压力）

        参数:
            marker (int): 物理组标记
            traction_vector (tuple of 3 floats): 面力向量 (Tx, Ty, Tz)，单位 Pa（即 N/m²）
                                                 注意：这是压力，不是集中力
        """
        t = fem.Constant(self.domain, ScalarType(traction_vector))
        self.load_terms.append((t, marker))

    def solve(self):
        """
        执行有限元求解

        返回:
            uh (dolfinx.fem.Function): 位移场函数对象
        """
        # 定义试探函数和测试函数
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # 定义应变和应力（各向同性线弹性）
        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            E = self.params['E']
            nu = self.params['nu']
            mu = E / (2 * (1 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            return 2 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3)

        # 双线性形式
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

        # 线性形式：面力贡献
        L = 0
        for (t, marker) in self.load_terms:
            L += ufl.inner(t, v) * self.ds(marker)

        # 求解线性系统
        problem = LinearProblem(
            a, L,
            bcs=self.bcs,
            petsc_options_prefix="fem_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}  # 直接求解器
        )
        self.uh = problem.solve()
        return self.uh

    # ==================== 结果输出接口（供可视化组调用） ====================

    def get_node_coordinates(self):
        """
        获取所有节点的坐标数组

        返回:
            coords (np.ndarray): 形状为 (N, 3) 的数组，N 为节点数
        """
        # 使用 plot.vtk_mesh 获取节点的坐标
        cells, types, x = plot.vtk_mesh(self.V)
        return x

    def get_node_displacements(self):
        """
        获取节点位移数组

        返回:
            u_nodes (np.ndarray): 形状为 (N, 3) 的数组，每一行对应节点的 (ux, uy, uz)
        """
        # 对于一阶拉格朗日单元，自由度的顺序与节点顺序一致
        return self.uh.x.array.reshape(-1, 3)

    def get_element_stress(self):
        """
        计算 von Mises 等效应力，并投影到单元中心（DG0 空间）

        返回:
            stress (np.ndarray): 形状为 (M,) 的数组，M 为单元数，每个单元一个应力值
        """
        # 定义应力表达式
        E = self.params['E']
        nu = self.params['nu']
        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

        def sigma(u):
            return 2 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(3)

        sigma_dev = sigma(self.uh) - (1 / 3) * ufl.tr(sigma(self.uh)) * ufl.Identity(3)
        von_mises = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))

        # 创建 DG0 空间（每个单元一个标量值）
        V_dg = fem.functionspace(self.domain, ("DG", 0))
        stress_expr = fem.Expression(von_mises, V_dg.element.interpolation_points)
        stress_h = fem.Function(V_dg)
        stress_h.interpolate(stress_expr)
        return stress_h.x.array.copy()

    def save_results(self, filename="displacement.xdmf"):
        """
        将位移场保存为 XDMF 文件，可用 ParaView 打开

        参数:
            filename (str): 输出文件名，建议扩展名为 .xdmf
        """
        with io.XDMFFile(self.comm, filename, "w") as file:
            file.write_mesh(self.domain)
            self.uh.name = "Displacement"
            file.write_function(self.uh)


# ==================== 使用示例（集成 C 组 LC-11 载荷） ====================
if __name__ == "__main__":
    # 假设网格文件名为 "bogie_frame.msh"，且包含了以下物理组标记：
    # 1: 固定端（例如弹簧座底部）
    # 2: 弹簧座面（4个面，承受垂向力）
    # 3: 横向止挡面（左右各一）
    # 4: 牵引杆安装座面（前后各一）
    # 5: 左前一系簧座（扭曲位移向上）
    # 6: 右后一系簧座（扭曲位移向下）
    mesh_file = "test_box.msh"

    # 创建求解器实例
    solver = FEMSolver(mesh_file)

    # 设置材料参数（例如钢材）
    solver.set_material(E=210e9, nu=0.3)

    # 设置固定边界（物理组 1）
    solver.set_fixed_boundary(markers=1)

    # 设置扭曲位移（物理组 5 和 6）
    solver.set_prescribed_displacement(marker=5, disp_vector=(0.0, 0.0, 0.025))   # 向上 25 mm
    solver.set_prescribed_displacement(marker=6, disp_vector=(0.0, 0.0, -0.025))  # 向下 25 mm

    # 施加载荷（来自 C 组 LC-11 工况）
    # 注意：以下压力值需要根据实际受力面积计算，面积需从网格或 CAD 测量得到
    # 假设面积值（单位 m²）：
    A_spring = 0.01      # 单个弹簧座面积
    A_lateral = 0.005    # 单个横向止挡面积
    A_traction = 0.008   # 单个牵引杆安装座面积

    # 计算压力（总力除以总面积，负号表示方向）
    p_vert = -313600 / (4 * A_spring)      # 垂向向下
    p_lat = 85750 / (2 * A_lateral)        # 横向（假设沿 Y 轴）
    p_long = 213591 / (2 * A_traction)     # 纵向（假设沿 X 轴）

    # 设置面力载荷
    solver.set_surface_load(marker=2, traction_vector=(0.0, 0.0, p_vert))
    solver.set_surface_load(marker=3, traction_vector=(0.0, p_lat, 0.0))
    solver.set_surface_load(marker=4, traction_vector=(p_long, 0.0, 0.0))

    # 执行求解
    print("正在求解...")
    uh = solver.solve()
    print("求解完成！")

    # 获取结果数据（供可视化组使用）
    coords = solver.get_node_coordinates()               # 节点坐标
    u_nodes = solver.get_node_displacements()             # 节点位移
    stress = solver.get_element_stress()                  # 单元应力

    # 打印结果形状，供可视化组参考
    print(f"节点数: {coords.shape[0]}")
    print(f"位移数组形状: {u_nodes.shape}")
    print(f"应力数组长度: {len(stress)}")

    # 保存结果文件（可选）
    solver.save_results("displacement.xdmf")
    np.savez("results.npz", coords=coords, displacement=u_nodes, stress=stress)
    print("结果已保存为 displacement.xdmf 和 results.npz")

    # 提示可视化组如何使用数据
    print("\n可视化组可使用以下方法读取结果：")
    print("  data = np.load('results.npz')")
    print("  coords = data['coords']          # (N,3) 节点坐标")
    print("  disp = data['displacement']       # (N,3) 节点位移")
    print("  stress = data['stress']           # (M,) 单元应力")
    print("  然后可使用 PyVista、Matplotlib 等绘图。")
