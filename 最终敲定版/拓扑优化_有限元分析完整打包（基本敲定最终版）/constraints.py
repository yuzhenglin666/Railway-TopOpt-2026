# constraints.py
import numpy as np
from scipy.spatial import cKDTree


class CastingConstraint:
    def __init__(self, domain, rho, params):
        self.domain = domain
        self.rho = rho
        self.params = params
        self.draft_dir = np.array(params.get('draft_direction', [0, 0, 1]))
        self.min_thickness = params.get('min_thickness', 2.0)
        self.draft_angle = params.get('draft_angle', 3.0)
        self.p_norm = params.get('p_norm', 6)
        self.draft_angle_rad = np.radians(self.draft_angle)

        norm_d = np.linalg.norm(self.draft_dir)
        self.d = self.draft_dir / norm_d if norm_d > 1e-6 else np.array([0, 0, 1])

        self.V_rho = rho.function_space
        self.cell_centers = self._get_cell_centers()
        self.n_cells = len(rho.x.array)
        self.tree = cKDTree(self.cell_centers)
        self.cell_size = self._estimate_cell_size()

    def _get_cell_centers(self):
        coords = self.V_rho.tabulate_dof_coordinates()
        return coords[:, :3]

    def _estimate_cell_size(self):
        if len(self.cell_centers) < 2: return 1.0
        dists, _ = self.tree.query(self.cell_centers, k=2)
        return np.median(dists[:, 1])

    def _ramp_function(self, x):
        return np.maximum(0, x)

    def compute_boundary_gradient(self, rho_arr):
        grad = np.zeros((self.n_cells, 3))
        centers = self.cell_centers
        dists, idxs = self.tree.query(centers, k=5)
        for i in range(self.n_cells):
            for neighbor_idx in range(1, 5):
                j = idxs[i, neighbor_idx]
                dist_j = dists[i, neighbor_idx]
                if dist_j < 1e-6: continue
                vec = centers[j] - centers[i]
                drho = rho_arr[j] - rho_arr[i]
                for k in range(3):
                    if abs(vec[k]) > 1e-6:
                        grad[i, k] += drho / vec[k] * np.exp(-dist_j ** 2 / self.cell_size ** 2)
        grad_norm = np.linalg.norm(grad, axis=1)
        grad_unit = np.zeros_like(grad)
        mask = grad_norm > 1e-6
        grad_unit[mask] = grad[mask] / grad_norm[mask, np.newaxis]
        return grad_norm, grad_unit

    def compute_casting_constraint(self, rho_arr):
        grad_norm, grad_unit = self.compute_boundary_gradient(rho_arr)
        n_dot_d = -np.dot(grad_unit, self.d)
        penalty = self._ramp_function(-n_dot_d)
        boundary_mask = ((rho_arr > 0.2) & (rho_arr < 0.8)).astype(float)
        g_pnorm = (np.sum((penalty ** 2 * boundary_mask) ** self.p_norm)) ** (1 / self.p_norm)
        g_pnorm = g_pnorm / self.n_cells
        tol = np.sin(self.draft_angle_rad) * 0.1
        g_val = g_pnorm - tol
        dgdx = np.zeros((1, self.n_cells))
        mask_idx = np.where(boundary_mask > 1e-6)[0]
        dgdx[0, mask_idx] = -2 * penalty[mask_idx] * (rho_arr[mask_idx] - 0.5) / self.n_cells
        return np.array([g_val]), dgdx

    def enforce_min_thickness(self, rho_arr):
        filter_radius = self.min_thickness / (2 * self.cell_size)
        if filter_radius < 1.0: return rho_arr
        rho_filtered = np.zeros_like(rho_arr)
        neighbors_list = self.tree.query_ball_point(self.cell_centers, filter_radius)
        for i, neighbors in enumerate(neighbors_list):
            if not neighbors: rho_filtered[i] = rho_arr[i]; continue
            dists = np.linalg.norm(self.cell_centers[neighbors] - self.cell_centers[i], axis=1)
            weights = np.maximum(0, filter_radius - dists)
            w_sum = np.sum(weights)
            rho_filtered[i] = np.sum(weights * rho_arr[neighbors]) / w_sum if w_sum > 0 else rho_arr[i]
        return rho_filtered

    def compute_thickness_sensitivity(self, rho_arr):
        dgdx = np.zeros((1, self.n_cells))
        filter_radius = self.min_thickness / (2 * self.cell_size)
        if filter_radius < 1.0: return dgdx
        neighbors_list = self.tree.query_ball_point(self.cell_centers, filter_radius)
        for i, neighbors in enumerate(neighbors_list):
            if len(neighbors) > 0:
                local_mean = np.mean(rho_arr[neighbors])
                if np.std(rho_arr[neighbors]) > 0.2: dgdx[0, i] = -0.1 * (rho_arr[i] - local_mean)
        return dgdx

    def get_constraint(self, rho_arr):
        rho_filtered = self.enforce_min_thickness(rho_arr)
        g_casting, dg_casting = self.compute_casting_constraint(rho_filtered)
        dg_thickness = self.compute_thickness_sensitivity(rho_filtered)
        return g_casting, dg_casting + dg_thickness * 0.1


def create_casting_constraint(domain, rho, material_type='steel', draft_dir=[0, 0, 1]):
    thickness_map = {'steel': 2.0, 'cast_iron': 3.0, 'aluminum': 1.5, 'stainless': 1.5}
    # 转换为字典供 CastingConstraint 使用
    params_dict = {
        'draft_direction': draft_dir,
        'min_thickness': thickness_map.get(material_type, 2.0),
        'draft_angle': 3.0,
        'p_norm': 6
    }
    return CastingConstraint(domain, rho, params_dict)