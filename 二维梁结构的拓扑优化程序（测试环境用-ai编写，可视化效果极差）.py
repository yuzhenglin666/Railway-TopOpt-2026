"""
二维梁结构拓扑优化程序 (SIMP方法) - 已适配FEniCSx 0.10.0
基于：Python, FEniCSx, Matplotlib
交通运输科技大赛项目 - 拓扑优化+复合材料
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# 检查环境依赖
print("=" * 60)
print("检查环境依赖...")

try:
    from mpi4py import MPI
    import dolfinx
    from dolfinx import mesh, fem, io
    from dolfinx.fem import (functionspace, FunctionSpace, Function, form, Constant,
                             dirichletbc, locate_dofs_topological, assemble_scalar)
    from dolfinx.fem.petsc import LinearProblem
    import ufl
    import basix.ufl

    FENICSX_AVAILABLE = True
    print("✓ FEniCSx 模块已导入")

    if hasattr(dolfinx, '__version__'):
        print(f"  dolfinx版本: {dolfinx.__version__}")

except ImportError as e:
    print(f"✗ FEniCSx导入失败: {e}")
    FENICSX_AVAILABLE = False

try:
    from petsc4py import PETSc

    PETSC_AVAILABLE = True
    print("✓ PETSc 模块已导入")
except ImportError:
    print("⚠ 未找到petsc4py模块")
    PETSC_AVAILABLE = False

print("=" * 60)

if not all([FENICSX_AVAILABLE, PETSC_AVAILABLE]):
    print("警告: 部分依赖模块未找到，程序可能无法正常运行")
    print("建议: 使用 'conda install -c conda-forge fenics-dolfinx mpi4py petsc4py' 安装")


# ==================== 拓扑优化器类 ====================
class TopologyOptimizer2D:
    """二维结构拓扑优化器 (SIMP方法)"""

    def __init__(self, params):
        """
        初始化拓扑优化器

        参数:
        params: 字典，包含优化参数
            - design_domain: 设计域尺寸 [Lx, Ly]
            - mesh_resolution: 网格分辨率 [nx, ny]
            - youngs_modulus: 杨氏模量
            - poisson_ratio: 泊松比
            - volume_fraction: 目标体积分数
            - penalization: SIMP惩罚因子 (通常3.0)
            - filter_radius: 灵敏度滤波半径
            - load: 载荷大小和方向
            - bc_type: 边界条件类型 ('cantilever'悬臂梁, 'bridge'桥梁, 'michell'迈克轮)
            - optimizer: 优化器 ('oc'最优准则法, 'mma'移动渐近线法)
            - max_iter: 最大迭代次数
            - tol: 收敛容差
        """
        self.params = params
        self.iterations = 0
        self.history = {
            'compliance': [],
            'volume': [],
            'change': [],
            'time': []
        }

        # 设置默认参数
        defaults = {
            'design_domain': [2.0, 1.0],
            'mesh_resolution': [60, 30],
            'youngs_modulus': 1.0,
            'poisson_ratio': 0.3,
            'volume_fraction': 0.4,
            'penalization': 3.0,
            'filter_radius': 1.5,
            'load': (0.0, -1.0),
            'bc_type': 'cantilever',
            'optimizer': 'oc',
            'max_iter': 50,
            'tol': 0.01,
            'move_limit': 0.2,
            'beta': 1.0,  # 投影滤波参数
            'material_model': 'SIMP'  # 'SIMP' 或 'RAMP'
        }

        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

        # 初始化MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # 设计变量和灵敏度
        self.rho = None  # 设计变量（密度）
        self.dc = None  # 灵敏度
        self.rho_filtered = None  # 滤波后的密度
        self.rho_physical = None  # 物理密度（经过投影）

        # FEniCSx对象
        self.domain = None
        self.V = None
        self.V_rho = None

        if self.rank == 0:
            print("拓扑优化器初始化完成")
            print(f"设计域: {self.params['design_domain'][0]}x{self.params['design_domain'][1]}")
            print(f"网格: {self.params['mesh_resolution'][0]}x{self.params['mesh_resolution'][1]}")
            print(f"体积分数: {self.params['volume_fraction']}, 惩罚因子: {self.params['penalization']}")
            print(f"材料模型: {self.params['material_model']}")

    def generate_mesh(self):
        """生成有限元网格"""
        if self.rank == 0:
            print("生成有限元网格...")

        Lx, Ly = self.params['design_domain']
        nx, ny = self.params['mesh_resolution']

        # 使用FEniCSx内置网格生成器
        self.domain = mesh.create_rectangle(
            self.comm,
            [[0.0, 0.0], [Lx, Ly]],
            [nx, ny],
            cell_type=mesh.CellType.triangle
        )

        # 创建函数空间 - 使用正确的API
        # 位移函数空间（向量值，用于二维弹性问题）
        self.V = functionspace(self.domain, ("Lagrange", 1, (2,)))

        # 密度函数空间（标量，每个单元一个值）
        self.V_rho = functionspace(self.domain, ("DG", 0))

        # 初始化设计变量（均匀分布）
        self.rho = Function(self.V_rho)
        self.rho.x.array[:] = self.params['volume_fraction']

        # 初始化滤波后的密度
        self.rho_filtered = Function(self.V_rho)
        self.rho_filtered.x.array[:] = self.params['volume_fraction']

        # 初始化物理密度（用于有限元分析）
        self.rho_physical = Function(self.V_rho)

        if self.rank == 0:
            tdim = self.domain.topology.dim
            print(f"网格生成完成")
            print(f"单元总数: {self.domain.topology.index_map(tdim).size_global}")
            print(f"位移自由度: {self.V.dofmap.index_map.size_global}")
            print(f"密度自由度: {self.V_rho.dofmap.index_map.size_global}")

    def define_boundary_conditions(self):
        """定义边界条件"""
        if self.rank == 0:
            print("定义边界条件和载荷...")

        Lx, Ly = self.params['design_domain']
        load_x, load_y = self.params['load']

        # 根据边界条件类型设置
        if self.params['bc_type'] == 'cantilever':
            # 悬臂梁：左边界固定
            def left_boundary(x):
                return np.isclose(x[0], 0.0)

            tdim = self.domain.topology.dim
            boundary_facets = mesh.locate_entities_boundary(
                self.domain, tdim - 1, left_boundary
            )
            boundary_dofs = locate_dofs_topological(
                self.V, tdim - 1, boundary_facets
            )

            zero_vector = np.zeros(2, dtype=PETSc.ScalarType)
            self.bc = dirichletbc(zero_vector, boundary_dofs, self.V)
            self.bcs = [self.bc]

            # 载荷施加在右边界中点（作为体积力近似）
            self.load_domain = lambda x: x[0] > Lx * 0.9

        elif self.params['bc_type'] == 'bridge':
            # 桥梁：底部固定
            def bottom_boundary(x):
                return np.isclose(x[1], 0.0)

            tdim = self.domain.topology.dim
            boundary_facets = mesh.locate_entities_boundary(
                self.domain, tdim - 1, bottom_boundary
            )
            boundary_dofs = locate_dofs_topological(
                self.V, tdim - 1, boundary_facets
            )

            zero_vector = np.zeros(2, dtype=PETSc.ScalarType)
            self.bc = dirichletbc(zero_vector, boundary_dofs, self.V)
            self.bcs = [self.bc]

            # 载荷施加在上边界中点
            self.load_domain = lambda x: (x[1] > Ly * 0.9) & (np.abs(x[0] - Lx / 2) < Lx * 0.1)

        elif self.params['bc_type'] == 'michell':
            # 迈克轮：左下角固定，简化处理为固定左边界
            def left_boundary(x):
                return np.isclose(x[0], 0.0)

            tdim = self.domain.topology.dim
            boundary_facets = mesh.locate_entities_boundary(
                self.domain, tdim - 1, left_boundary
            )
            boundary_dofs = locate_dofs_topological(
                self.V, tdim - 1, boundary_facets
            )

            zero_vector = np.zeros(2, dtype=PETSc.ScalarType)
            self.bc = dirichletbc(zero_vector, boundary_dofs, self.V)
            self.bcs = [self.bc]

            # 载荷施加在右上角
            self.load_domain = lambda x: (x[0] > Lx * 0.9) & (x[1] > Ly * 0.9)

        if self.rank == 0:
            print(f"边界条件类型: {self.params['bc_type']}")

    def material_interpolation(self, rho):
        """材料插值"""
        E_min = 1e-9 * self.params['youngs_modulus']  # 防止奇异矩阵
        E_0 = self.params['youngs_modulus']
        p = self.params['penalization']

        if self.params['material_model'] == 'SIMP':
            # SIMP插值: E = E_min + (E_0 - E_min) * rho^p
            return E_min + (E_0 - E_min) * rho ** p
        elif self.params['material_model'] == 'RAMP':
            # RAMP插值: E = E_min + (E_0 - E_min) * rho/(1 + p*(1 - rho))
            return E_min + (E_0 - E_min) * rho / (1 + p * (1 - rho))
        else:
            # 默认使用SIMP
            return E_min + (E_0 - E_min) * rho ** p

    def solve_elasticity(self):
        """求解线弹性问题"""
        # 更新物理密度（经过投影滤波）
        self.apply_projection_filter()

        # 材料属性
        E = self.material_interpolation(self.rho_physical)
        nu = self.params['poisson_ratio']

        # 拉梅常数（平面应力假设）
        mu = E / (2.0 * (1.0 + nu))
        lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # 平面应力的修正lambda
        lmbda = 2 * mu * lmbda / (lmbda + 2 * mu)

        # 定义应变和应力
        def epsilon(u):
            return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

        def sigma(u):
            return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2)

        # 定义变分问题
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # 载荷
        load_x, load_y = self.params['load']
        f = Constant(self.domain, PETSc.ScalarType((load_x, load_y)))

        # 变分形式
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.dot(f, v) * ufl.dx

        # 创建并求解线性系统 - 修复：添加petsc_options_prefix参数
        problem = LinearProblem(a, L, bcs=self.bcs, petsc_options_prefix="beam_")

        # 配置求解器选项
        opts = PETSc.Options()
        opts.prefixPush("beam_")
        opts["ksp_type"] = "preonly"
        opts["pc_type"] = "lu"
        opts["pc_factor_mat_solver_type"] = "mumps"
        opts.prefixPop()

        problem.solver.setFromOptions()

        # 求解位移场
        u_h = problem.solve()

        # 计算柔度（目标函数）
        compliance_form = form(ufl.dot(f, u_h) * ufl.dx)
        compliance = assemble_scalar(compliance_form)

        # 计算灵敏度 (∂C/∂ρ)
        # 对于SIMP方法: ∂C/∂ρ = -p * ρ^(p-1) * u^T * K0 * u
        # 简化计算：使用应变能密度
        p = self.params['penalization']
        rho_array = self.rho_physical.x.array

        # 计算单元应变能
        strain_energy_density = 0.5 * ufl.inner(sigma(u_h), epsilon(u_h))

        # 创建灵敏度函数
        self.dc = Function(self.V_rho)

        # 简化计算灵敏度
        # 注意：这是一个近似，精确计算需要遍历每个单元
        if self.params['material_model'] == 'SIMP':
            dc_array = -p * rho_array ** (p - 1) * float(compliance) / len(rho_array)
        elif self.params['material_model'] == 'RAMP':
            # RAMP模型的灵敏度
            q = p
            dc_array = - (1 + q) / (1 + q * (1 - rho_array)) ** 2 * float(compliance) / len(rho_array)

        self.dc.x.array[:] = dc_array

        return u_h, compliance

    def apply_density_filter(self):
        """应用密度滤波 (灵敏度滤波)"""
        filter_radius = self.params['filter_radius']

        # 简化滤波：均值滤波
        rho_array = self.rho.x.array
        rho_filtered_array = np.zeros_like(rho_array)

        # 简化的均值滤波
        # 在实际应用中，应该使用基于距离的卷积滤波
        n = len(rho_array)
        for i in range(n):
            # 取相邻单元的均值（简化处理）
            indices = [max(0, i - 2), max(0, i - 1), i, min(n - 1, i + 1), min(n - 1, i + 2)]
            rho_filtered_array[i] = np.mean(rho_array[indices])

        self.rho_filtered.x.array[:] = rho_filtered_array

        if self.rank == 0 and self.iterations % 10 == 0:
            print(f"  密度滤波完成 (半径: {filter_radius})")

    def apply_projection_filter(self):
        """应用投影滤波 (将密度转换为物理密度)"""
        beta = self.params.get('beta', 1.0)
        eta = 0.5  # 投影阈值

        # Heaviside投影
        rho_f = self.rho_filtered.x.array
        if beta < 1e-6:
            # 当beta很小时，近似为阶跃函数
            rho_p = np.where(rho_f > eta, 1.0, 0.0)
        else:
            # 平滑Heaviside投影
            numerator = np.tanh(beta * eta) + np.tanh(beta * (rho_f - eta))
            denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
            rho_p = numerator / denominator

        self.rho_physical.x.array[:] = rho_p

    def update_design_variables_oc(self, compliance):
        """使用最优准则法(OC)更新设计变量"""
        move_limit = self.params['move_limit']
        volume_fraction = self.params['volume_fraction']

        # 当前体积
        current_volume = np.mean(self.rho.x.array)

        # 二分法求解拉格朗日乘子
        l1 = 1e-6
        l2 = 1e6
        lmbda = 0.5 * (l1 + l2)

        for i in range(50):  # 二分法迭代
            # 计算更新因子
            B = -self.dc.x.array / (lmbda + 1e-12)
            B = np.maximum(1e-6, B)  # 防止除零

            # OC更新公式
            rho_new = self.rho.x.array * np.sqrt(B)

            # 应用移动限制
            rho_lower = np.maximum(0.001, self.rho.x.array - move_limit)
            rho_upper = np.minimum(0.999, self.rho.x.array + move_limit)
            rho_new = np.maximum(rho_lower, np.minimum(rho_upper, rho_new))

            # 检查体积
            new_volume = np.mean(rho_new)

            if new_volume > volume_fraction:
                l1 = lmbda
            else:
                l2 = lmbda

            lmbda = 0.5 * (l1 + l2)

            # 提前终止
            if abs(new_volume - volume_fraction) < 0.001:
                break

        # 计算设计变量变化
        change = np.max(np.abs(rho_new - self.rho.x.array))

        # 更新设计变量
        self.rho.x.array[:] = rho_new

        return change, new_volume

    def check_convergence(self, change, iteration):
        """检查收敛条件"""
        # 检查变化量
        if change < self.params['tol']:
            if self.rank == 0:
                print(f"  收敛: 设计变量变化 {change:.6f} < 容差 {self.params['tol']}")
            return True

        # 检查最大迭代次数
        if iteration >= self.params['max_iter']:
            if self.rank == 0:
                print(f"  达到最大迭代次数 {self.params['max_iter']}")
            return True

        # 检查柔度变化（如果历史记录足够长）
        if len(self.history['compliance']) >= 10:
            recent_compliance = self.history['compliance'][-10:]
            if max(recent_compliance) - min(recent_compliance) < 0.001 * abs(np.mean(recent_compliance)):
                if self.rank == 0:
                    print(f"  收敛: 柔度在最近10次迭代中变化很小")
                return True

        return False

    def optimize(self):
        """执行拓扑优化主循环"""
        if self.rank == 0:
            print("\n" + "=" * 60)
            print("开始拓扑优化")
            print("=" * 60)

        start_time = time.time()

        # 迭代优化
        for iteration in range(self.params['max_iter']):
            iter_start_time = time.time()
            self.iterations = iteration + 1

            if self.rank == 0:
                print(f"\n迭代 {self.iterations}/{self.params['max_iter']}")

            # 1. 求解弹性问题，计算柔度和灵敏度
            u_h, compliance = self.solve_elasticity()

            # 2. 应用灵敏度滤波
            self.apply_density_filter()

            # 3. 更新设计变量
            change, volume = self.update_design_variables_oc(compliance)

            # 4. 记录历史
            self.history['compliance'].append(float(compliance))
            self.history['volume'].append(float(volume))
            self.history['change'].append(float(change))
            self.history['time'].append(time.time() - iter_start_time)

            # 5. 打印进度
            if self.rank == 0:
                print(f"  柔度: {compliance:.6e}")
                print(f"  体积分数: {volume:.4f} (目标: {self.params['volume_fraction']})")
                print(f"  最大变化: {change:.6f}")
                print(f"  迭代时间: {self.history['time'][-1]:.3f}秒")

            # 6. 检查收敛
            if self.check_convergence(change, iteration):
                break

            # 7. 每10次迭代保存一次中间结果
            if self.iterations % 10 == 0:
                if self.rank == 0:
                    self.save_results(iteration=self.iterations)
                    self.visualize_density(iteration=self.iterations, show=False)

        total_time = time.time() - start_time

        if self.rank == 0:
            print(f"\n优化完成!")
            print(f"总迭代次数: {self.iterations}")
            print(f"总时间: {total_time:.2f}秒")
            print(f"平均每次迭代: {total_time / self.iterations:.3f}秒")

        # 保存最终结果
        if self.rank == 0:
            self.save_results(iteration='final')

        return self.rho, self.history

    def save_results(self, iteration='final'):
        """保存优化结果"""
        # 创建结果目录
        os.makedirs("topopt_results", exist_ok=True)

        # 保存密度场
        filename = f"topopt_results/density_iter_{iteration}.xdmf"
        with io.XDMFFile(self.comm, filename, "w") as xdmf:
            xdmf.write_mesh(self.domain)
            self.rho.name = "Density"
            xdmf.write_function(self.rho)

        # 保存优化历史
        history_file = f"topopt_results/optimization_history.npz"
        np.savez(history_file,
                 compliance=self.history['compliance'],
                 volume=self.history['volume'],
                 change=self.history['change'],
                 time=self.history['time'],
                 params=self.params)

        # 保存参数
        params_file = f"topopt_results/parameters.txt"
        with open(params_file, 'w') as f:
            for key, value in self.params.items():
                f.write(f"{key}: {value}\n")

        print(f"结果已保存到: {filename}")

    def visualize_density(self, iteration='final', show=True):
        """可视化密度分布"""
        # 获取密度数据
        rho_array = self.rho.x.array

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 密度分布直方图
        ax1 = axes[0, 0]
        ax1.hist(rho_array, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('密度')
        ax1.set_ylabel('单元数量')
        ax1.set_title(f'密度分布直方图 (迭代 {iteration})')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=self.params['volume_fraction'], color='red', linestyle='--',
                    label=f'目标体积分数: {self.params["volume_fraction"]}')
        ax1.legend()

        # 2. 优化历史：柔度
        ax2 = axes[0, 1]
        if len(self.history['compliance']) > 0:
            iterations = range(1, len(self.history['compliance']) + 1)
            ax2.plot(iterations, self.history['compliance'], 'b-o', linewidth=2, markersize=4)
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('柔度')
            ax2.set_title('柔度收敛历史')
            ax2.grid(True, alpha=0.3)

            # 添加收敛指示器
            if len(self.history['compliance']) > 1:
                improvement = 100 * (self.history['compliance'][0] - self.history['compliance'][-1]) / \
                              self.history['compliance'][0]
                ax2.text(0.05, 0.95, f'改善: {improvement:.1f}%',
                         transform=ax2.transAxes, fontsize=10,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 3. 优化历史：体积分数
        ax3 = axes[1, 0]
        if len(self.history['volume']) > 0:
            iterations = range(1, len(self.history['volume']) + 1)
            ax3.plot(iterations, self.history['volume'], 'g-s', linewidth=2, markersize=4)
            ax3.axhline(y=self.params['volume_fraction'], color='r',
                        linestyle='--', label='目标体积分数')
            ax3.set_xlabel('迭代次数')
            ax3.set_ylabel('体积分数')
            ax3.set_title('体积分数历史')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. 优化历史：设计变量变化
        ax4 = axes[1, 1]
        if len(self.history['change']) > 0:
            iterations = range(1, len(self.history['change']) + 1)
            ax4.plot(iterations, self.history['change'], 'r-^', linewidth=2, markersize=4)
            ax4.axhline(y=self.params['tol'], color='k',
                        linestyle='--', label='收敛容差')
            ax4.set_xlabel('迭代次数')
            ax4.set_ylabel('最大变化')
            ax4.set_title('设计变量变化历史')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.suptitle(f'拓扑优化结果 - {self.params["bc_type"]}梁', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图形
        fig.savefig(f'topopt_results/visualization_iter_{iteration}.png', dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

    def compare_with_commercial_software(self):
        """与商业软件结果对比（概念性）"""
        if self.rank != 0:
            return

        print("\n" + "=" * 60)
        print("拓扑优化结果分析")
        print("=" * 60)

        # 生成报告
        report_file = "topopt_results/optimization_report.txt"
        with open(report_file, 'w') as f:
            f.write("二维梁结构拓扑优化结果报告\n")
            f.write("=" * 50 + "\n\n")
            f.write("1. 优化参数\n")
            f.write("-" * 30 + "\n")
            for key, value in self.params.items():
                f.write(f"{key}: {value}\n")

            f.write("\n2. 优化结果\n")
            f.write("-" * 30 + "\n")
            if self.history['compliance']:
                f.write(f"初始柔度: {self.history['compliance'][0]:.6e}\n")
                f.write(f"最终柔度: {self.history['compliance'][-1]:.6e}\n")
                improvement = 100 * (self.history['compliance'][0] - self.history['compliance'][-1]) / \
                              self.history['compliance'][0]
                f.write(f"改善比例: {improvement:.2f}%\n")
                f.write(f"最终体积分数: {self.history['volume'][-1]:.4f}\n")
                f.write(f"目标体积分数: {self.params['volume_fraction']}\n")
                f.write(f"总迭代次数: {self.iterations}\n")
                f.write(f"总计算时间: {sum(self.history['time']):.2f}秒\n")

            f.write("\n3. 验证建议\n")
            f.write("-" * 30 + "\n")
            f.write("3.1 有限元验证\n")
            f.write("   - 在Abaqus/ANSYS中建立相同模型\n")
            f.write("   - 施加相同边界条件和载荷\n")
            f.write("   - 比较位移场和应力分布\n")
            f.write("   - 计算柔度并进行对比\n\n")

            f.write("3.2 物理实验验证\n")
            f.write("   - 将优化结构导出为STL格式\n")
            f.write("   - 使用3D打印制造优化结构\n")
            f.write("   - 进行静态载荷测试\n")
            f.write("   - 测量位移并与模拟结果对比\n\n")

            f.write("3.3 性能指标对比\n")
            f.write("   - 刚度/重量比\n")
            f.write("   - 最大应力水平\n")
            f.write("   - 制造可行性\n")

        print(f"优化报告已保存到: {report_file}")

        # 打印建议
        print("\n下一步验证建议:")
        print("1. 将优化结构导出为STL格式进行3D打印")
        print("2. 在Abaqus/ANSYS中验证结构性能")
        print("3. 与传统设计方案进行对比")


# ==================== 简化测试版本 ====================
def test_simple_cantilever():
    """简化的悬臂梁优化测试"""
    print("\n" + "=" * 60)
    print("简化的悬臂梁拓扑优化测试")
    print("=" * 60)

    # 简化参数 - 减少网格和迭代次数以快速测试
    params = {
        'design_domain': [2.0, 1.0],
        'mesh_resolution': [40, 20],  # 减少网格密度
        'youngs_modulus': 1.0,
        'poisson_ratio': 0.3,
        'volume_fraction': 0.4,
        'penalization': 3.0,
        'filter_radius': 1.5,
        'load': (0.0, -1.0),
        'bc_type': 'cantilever',
        'optimizer': 'oc',
        'max_iter': 20,  # 减少迭代次数
        'tol': 0.02,  # 增大容差
        'move_limit': 0.2,
        'material_model': 'SIMP',
        'beta': 1.0,
    }

    try:
        # 创建优化器
        optimizer = TopologyOptimizer2D(params)

        # 生成网格
        optimizer.generate_mesh()

        # 定义边界条件
        optimizer.define_boundary_conditions()

        print("\n开始简化优化...")

        # 执行少量迭代
        start_time = time.time()
        for iteration in range(params['max_iter']):
            print(f"\n迭代 {iteration + 1}/{params['max_iter']}")

            # 求解弹性问题
            u_h, compliance = optimizer.solve_elasticity()

            # 记录历史
            optimizer.history['compliance'].append(float(compliance))
            optimizer.history['volume'].append(float(np.mean(optimizer.rho.x.array)))

            # 简单更新设计变量（简化）
            # 这里使用一个简化的更新规则，而不是完整的OC方法
            if iteration < params['max_iter'] - 1:
                # 基于灵敏度简单更新
                dc_norm = np.linalg.norm(optimizer.dc.x.array)
                if dc_norm > 0:
                    update = -0.1 * optimizer.dc.x.array / dc_norm
                    new_rho = optimizer.rho.x.array + update
                    new_rho = np.clip(new_rho, 0.001, 0.999)
                    optimizer.rho.x.array[:] = new_rho

            print(f"  柔度: {compliance:.6e}")
            print(f"  体积分数: {np.mean(optimizer.rho.x.array):.4f}")

        total_time = time.time() - start_time
        print(f"\n简化优化完成! 总时间: {total_time:.2f}秒")

        # 保存结果
        optimizer.save_results(iteration='test')

        # 可视化
        optimizer.visualize_density(iteration='test', show=True)

        return optimizer

    except Exception as e:
        print(f"简化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_minimal_fem():
    """最小有限元测试 - 验证FEniCSx环境"""
    print("\n" + "=" * 60)
    print("最小有限元测试")
    print("=" * 60)

    try:
        from mpi4py import MPI
        from petsc4py import PETSc

        # 创建简单网格
        comm = MPI.COMM_WORLD
        domain = mesh.create_rectangle(comm, [[0.0, 0.0], [2.0, 1.0]], [20, 10])

        # 创建函数空间
        V = functionspace(domain, ("Lagrange", 1, (2,)))

        # 定义边界条件（左边界固定）
        def left_boundary(x):
            return np.isclose(x[0], 0.0)

        tdim = domain.topology.dim
        boundary_facets = mesh.locate_entities_boundary(domain, tdim - 1, left_boundary)
        boundary_dofs = locate_dofs_topological(V, tdim - 1, boundary_facets)

        zero_vector = np.zeros(2, dtype=PETSc.ScalarType)
        bc = dirichletbc(zero_vector, boundary_dofs, V)

        # 材料参数
        E = 1.0
        nu = 0.3
        mu = E / (2.0 * (1.0 + nu))
        lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # 平面应力修正
        lmbda = 2 * mu * lmbda / (lmbda + 2 * mu)

        # 定义变分问题
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        def epsilon(u):
            return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

        def sigma(u):
            return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2)

        # 载荷
        f = Constant(domain, PETSc.ScalarType((0.0, -1.0)))

        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.dot(f, v) * ufl.dx

        # 求解
        problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="test_")

        opts = PETSc.Options()
        opts.prefixPush("test_")
        opts["ksp_type"] = "preonly"
        opts["pc_type"] = "lu"
        opts["pc_factor_mat_solver_type"] = "mumps"
        opts.prefixPop()

        problem.solver.setFromOptions()

        u_h = problem.solve()

        # 计算柔度
        compliance_form = form(ufl.dot(f, u_h) * ufl.dx)
        compliance = assemble_scalar(compliance_form)

        print(f"有限元测试成功!")
        print(f"位移场求解完成")
        print(f"柔度: {compliance:.6e}")
        print(f"最大位移: {np.max(np.abs(u_h.x.array)):.6e}")

        return True

    except Exception as e:
        print(f"有限元测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""

    # 检查必要模块
    if not FENICSX_AVAILABLE:
        print("错误: FEniCSx未正确安装!")
        print("请运行: conda install -c conda-forge fenics-dolfinx")
        return

    print("\n" + "=" * 60)
    print("二维梁结构拓扑优化程序")
    print("交通运输科技大赛项目: 拓扑优化+复合材料")
    print("=" * 60)

    # 运行最小有限元测试
    print("\n第一步: 运行最小有限元测试...")
    fem_success = test_minimal_fem()

    if not fem_success:
        print("有限元测试失败，无法继续拓扑优化")
        return

    # 运行简化优化测试
    print("\n第二步: 运行简化拓扑优化测试...")
    optimizer = test_simple_cantilever()

    if optimizer:
        print("\n" + "=" * 60)
        print("测试成功完成!")
        print("=" * 60)
        print("\n下一步:")
        print("1. 查看 'topopt_results' 目录中的结果")
        print("2. 使用ParaView打开XDMF文件查看密度分布")
        print("3. 查看PNG文件中的收敛历史")

        # 询问是否运行完整优化
        run_full = input("\n是否运行完整优化? (y/n): ").strip().lower()
        if run_full == 'y':
            try:
                # 完整优化参数
                full_params = {
                    'design_domain': [2.0, 1.0],
                    'mesh_resolution': [80, 40],
                    'youngs_modulus': 1.0,
                    'poisson_ratio': 0.3,
                    'volume_fraction': 0.4,
                    'penalization': 3.0,
                    'filter_radius': 1.5,
                    'load': (0.0, -1.0),
                    'bc_type': 'cantilever',
                    'optimizer': 'oc',
                    'max_iter': 50,
                    'tol': 0.01,
                    'move_limit': 0.2,
                    'material_model': 'SIMP',
                    'beta': 1.0,
                }

                print("\n" + "=" * 60)
                print("运行完整拓扑优化")
                print("=" * 60)

                full_optimizer = TopologyOptimizer2D(full_params)
                full_optimizer.generate_mesh()
                full_optimizer.define_boundary_conditions()
                rho_opt, history = full_optimizer.optimize()

                full_optimizer.visualize_density(iteration='full', show=True)
                full_optimizer.compare_with_commercial_software()

            except Exception as e:
                print(f"完整优化失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("简化测试失败，请检查错误信息")


# ==================== 运行程序 ====================
if __name__ == "__main__":
    main()