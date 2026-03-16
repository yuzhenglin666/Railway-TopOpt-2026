# fea_solver.py
import ufl
import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType


class FEASolver:
    def __init__(self, domain, facet_tags, rho, material_params):
        self.domain = domain
        self.facet_tags = facet_tags
        self.rho = rho
        self.E0 = float(material_params['E0'])
        self.Emin = float(material_params.get('Emin', 1e-9))
        self.nu = float(material_params['nu'])
        self.p = float(material_params.get('p', 3.0))

        self.V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        self.u_trial = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # 极硬核投影空间
        self.V_dg = fem.functionspace(self.domain, ("DG", 0))
        self.u_dg = ufl.TrialFunction(self.V_dg)
        self.v_dg = ufl.TestFunction(self.V_dg)
        self.a_proj = ufl.inner(self.u_dg, self.v_dg) * ufl.dx

        self.u = fem.Function(self.V)
        self._setup_bcs()
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        self.load_terms = []

    def _setup_bcs(self):
        fixed_facets = self.facet_tags.find(1)
        fixed_dofs = fem.locate_dofs_topological(self.V, self.domain.topology.dim - 1, fixed_facets)
        zero = fem.Constant(self.domain, ScalarType((0.0, 0.0, 0.0)))
        self.bcs = [fem.dirichletbc(zero, fixed_dofs, self.V)]

    def add_pressure_load(self, magnitude, direction, surface_tag=2):
        mag = float(magnitude)
        coords = self.domain.geometry.x
        Ly = np.max(coords[:, 1]) - np.min(coords[:, 1])
        Lz = np.max(coords[:, 2]) - np.min(coords[:, 2])
        area = max(1e-8, Ly * Lz)
        pressure = mag / area
        f_load = fem.Constant(self.domain,
                              ScalarType((direction[0] * pressure, direction[1] * pressure, direction[2] * pressure)))
        self.load_terms.append((f_load, surface_tag))

    def clear_loads(self):
        self.load_terms = []

    def get_forms(self):
        def epsilon(u): return ufl.sym(ufl.grad(u))

        def sigma(u):
            E = self.Emin + self.rho ** self.p * (self.E0 - self.Emin)
            mu = E / (2 * (1 + self.nu))
            lmbda = E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            return 2 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3)

        a = ufl.inner(sigma(self.u_trial), epsilon(self.v)) * ufl.dx
        L = ufl.inner(fem.Constant(self.domain, ScalarType((0.0, 0.0, 0.0))), self.v) * ufl.dx
        for f_load, tag in self.load_terms:
            L += ufl.inner(f_load, self.v) * self.ds(tag)
        return a, L

    def solve(self):
        a, L = self.get_forms()
        problem = LinearProblem(a, L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                petsc_options_prefix="opts_2_")
        uh_temp = problem.solve()
        self.u.x.array[:] = uh_temp.x.array
        return self.u

    def compute_compliance(self, uh):
        _, L = self.get_forms()
        return float(fem.assemble_scalar(fem.form(ufl.action(L, uh))))

    def compute_sensitivity(self, uh):
        def epsilon(u): return ufl.sym(ufl.grad(u))

        dE_drho = self.p * (self.rho ** (self.p - 1)) * (self.E0 - self.Emin)
        mu_p = dE_drho / (2 * (1 + self.nu))
        lmbda_p = dE_drho * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        def dsigma(u): return 2 * mu_p * epsilon(u) + lmbda_p * ufl.tr(epsilon(u)) * ufl.Identity(3)

        sens_expr = -ufl.inner(dsigma(uh), epsilon(uh))

        L_proj = ufl.inner(sens_expr, self.v_dg) * ufl.dx
        problem = LinearProblem(self.a_proj, L_proj, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
                                petsc_options_prefix="p_sen_")
        dc_h = problem.solve()
        return dc_h.x.array[:]

    def get_von_mises(self):
        if float(np.max(np.abs(self.u.x.array))) < 1e-12: return None
        E, nu = self.E0, self.nu
        mu = fem.Constant(self.domain, ScalarType(E / (2 * (1 + nu))))
        lmbda = fem.Constant(self.domain, ScalarType(E * nu / ((1 + nu) * (1 - 2 * nu))))

        def epsilon(u): return ufl.sym(ufl.grad(u))

        def sigma(u): return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2 * mu * epsilon(u)

        s = sigma(self.u) - (1. / 3) * ufl.tr(sigma(self.u)) * ufl.Identity(3)
        von_mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))

        L_proj = ufl.inner(von_mises, self.v_dg) * ufl.dx
        problem = LinearProblem(self.a_proj, L_proj, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
                                petsc_options_prefix="p_smis_")
        stress_f = problem.solve()
        return stress_f.x.array.copy()