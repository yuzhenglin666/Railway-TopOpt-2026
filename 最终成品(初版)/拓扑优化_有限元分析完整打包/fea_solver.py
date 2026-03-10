# fea_solver.py
import ufl
import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType


class FEASolver:
    def __init__(self, domain, facet_tags, rho, material_params):
        """
        domain: 网格
        facet_tags: 边界标记
        rho: 密度函数 (DG0)
        material_params: 材料参数字典
        """
        self.domain = domain
        self.facet_tags = facet_tags
        self.rho = rho
        self.E0 = material_params['E0']
        self.Emin = material_params.get('Emin', 1e-9)
        self.nu = material_params['nu']
        self.p = material_params.get('p', 3.0)

        self.V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        self.u_trial = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self._setup_bcs()
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        self.load_terms = []  # 存储多个载荷项 (f_load, tag)

    def _setup_bcs(self):
        fixed_facets = self.facet_tags.find(1)
        fixed_dofs = fem.locate_dofs_topological(
            self.V, self.domain.topology.dim - 1, fixed_facets
        )
        zero = fem.Constant(self.domain, ScalarType((0.0, 0.0, 0.0)))
        self.bc_left = fem.dirichletbc(zero, fixed_dofs, self.V)
        self.bcs = [self.bc_left]

    def add_pressure_load(self, magnitude, direction, surface_tag=2):
        """
        添加一个压力载荷
        magnitude: 总力大小 (N)
        direction: 方向向量 (dx, dy, dz)
        surface_tag: 载荷作用面的物理组标记 (默认 2 为右端面)
        """
        # 计算作用面面积（假设为矩形，可根据 surface_tag 调整）
        coords = self.domain.geometry.x
        Ly = np.max(coords[:, 1]) - np.min(coords[:, 1])
        Lz = np.max(coords[:, 2]) - np.min(coords[:, 2])
        area = Ly * Lz  # 右端面面积
        pressure = magnitude / area
        load_vector = (direction[0] * pressure, direction[1] * pressure, direction[2] * pressure)
        f_load = fem.Constant(self.domain, ScalarType(load_vector))
        self.load_terms.append((f_load, surface_tag))

    def clear_loads(self):
        """清除所有载荷，用于重新设置"""
        self.load_terms = []

    def get_forms(self):
        """返回 a 和 L 形式，用于组装"""

        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            E = self.Emin + self.rho ** self.p * (self.E0 - self.Emin)
            mu = E / (2 * (1 + self.nu))
            lmbda = E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            return 2 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))

        a = ufl.inner(sigma(self.u_trial), epsilon(self.v)) * ufl.dx
        # 组合所有载荷项
        L = 0
        for f_load, tag in self.load_terms:
            L += ufl.inner(f_load, self.v) * self.ds(tag)
        return a, L

    def solve(self):
        a, L = self.get_forms()
        problem = LinearProblem(a, L, bcs=self.bcs,
                                petsc_options_prefix="topopt_",
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()
        return uh

    def compute_compliance(self, uh):
        """计算柔度（目标函数）"""
        a, L = self.get_forms()

        # 修复：防止测试刚开始无载荷时 L 报错为 int
        if L == 0:
            return 0.0

        # 注意：L 作为 1-form 函数内含有供外力内积的虚拟位移 self.v，
        # 要进行求解积分必须把外力内的 self.v 彻底平铺并替换成刚解得的实际 uh 进行积分！
        compliance_expr = ufl.replace(L, {self.v: uh})
        compliance_form = fem.form(compliance_expr)
        return fem.assemble_scalar(compliance_form)

    def compute_energy_density(self, uh):
        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            E = self.Emin + self.rho ** self.p * (self.E0 - self.Emin)
            mu = E / (2 * (1 + self.nu))
            lmbda = E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            return 2 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))

        energy = 0.5 * ufl.inner(sigma(uh), epsilon(uh))
        V_dg = fem.functionspace(self.domain, ("DG", 0))
        expr = fem.Expression(energy, V_dg.element.interpolation_points)
        e_h = fem.Function(V_dg)
        e_h.interpolate(expr)
        return e_h.x.array

    def compute_sensitivity(self, uh):
        """计算目标函数（柔度）对密度场 rho 的梯度（灵敏度）"""

        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        dE_drho = self.p * (self.rho ** (self.p - 1)) * (self.E0 - self.Emin)

        mu_prime = dE_drho / (2 * (1 + self.nu))
        lmbda_prime = dE_drho * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        def dsigma_drho(u):
            return 2 * mu_prime * epsilon(u) + lmbda_prime * ufl.tr(epsilon(u)) * ufl.Identity(len(u))

            # 巧妙复用 Expression 方法免除 assemble_vector 组装时的底层版本阻挠和多进程数据错误

        # 对每一个单元乘上其对应的 dx 体积权重 (ufl.CellVolume)
        # 求导公式本质 dc_i = \int - inner(dsigma_drho(u), epsilon(u)) dx
        h_vol = ufl.CellVolume(self.domain)
        sens_expr = -ufl.inner(dsigma_drho(uh), epsilon(uh)) * h_vol

        V_dg = fem.functionspace(self.domain, ("DG", 0))
        expr = fem.Expression(sens_expr, V_dg.element.interpolation_points)
        dc_h = fem.Function(V_dg)
        dc_h.interpolate(expr)

        return dc_h.x.array[:]