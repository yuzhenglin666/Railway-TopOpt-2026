# topology.py
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix


class TopologyOptimizer:
    def __init__(self, rho, fea_solver, params, casting_constraint=None):
        self.rho = rho
        self.fea = fea_solver
        self.params = params  # 🔴 关键：保存整个 params 字典以便提取 active_loads
        self.casting_constraint = casting_constraint
        self.vol_frac = params.get('vol_frac', 0.3)
        self.p = params.get('p', 3.0)
        self.rmin = params.get('rmin', 0.1)
        self.move = params.get('move', 0.2)
        self.tol = params.get('tol', 1e-2)
        self.max_iter = params.get('max_iter', 50)

        self.n = len(rho.x.array)
        self.history = {'compliance': [], 'vol': [], 'change': []}

        # [核心提速] 初始化时直接构建滤波权重稀疏矩阵
        self._build_filter_matrix()

    def _build_filter_matrix(self):
        """利用 cKDTree 和稀疏矩阵将滤波速度提升百倍"""
        coords = self.rho.function_space.tabulate_dof_coordinates()
        cell_centers = coords[:, :3]
        tree = cKDTree(cell_centers)

        row, col, data = [], [], []
        neighbors_list = tree.query_ball_point(cell_centers, self.rmin)
        for i, neighbors in enumerate(neighbors_list):
            for j in neighbors:
                dist = np.linalg.norm(cell_centers[i] - cell_centers[j])
                weight = self.rmin - dist
                if weight > 0:
                    row.append(i)
                    col.append(j)
                    data.append(weight)

        self.H = coo_matrix((data, (row, col)), shape=(self.n, self.n)).tocsr()
        self.Hs = self.H.sum(axis=1).A1

    def filter_sensitivities(self, dc):
        """极速灵敏度滤波（基于稀疏矩阵乘法）"""
        dc_f = (self.H.dot(dc * self.rho.x.array)) / self.Hs / np.maximum(1e-3, self.rho.x.array)
        return dc_f

    def _oc_update(self, x, dc):
        """最优准则法 (Optimality Criteria) 迭代更新材料密度"""
        l1, l2 = 0.0, 1e9
        x_new = np.zeros_like(x)

        # 二分法寻找拉格朗日乘子 (满足体积约束)
        while (l2 - l1) / (l1 + l2 + 1e-3) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            # OC法的启发式更新公式
            B_k = np.sqrt(-dc / lmid)

            x_new[:] = np.maximum(0.001, np.maximum(x - self.move,
                                                    np.minimum(1.0, np.minimum(x + self.move, x * B_k))))

            # 判断体积是否超标
            if np.mean(x_new) - self.vol_frac > 0:
                l1 = lmid
            else:
                l2 = lmid
        return x_new

    def optimize_step(self, iteration):
        """
        🚀 核心旗舰功能：多工况综合分析与迭代
        """
        # 1. 获取主程序传递过来的所有【已激活工况】
        active_loads = self.params.get('active_loads', [{'dir': (0, 0, -1), 'mag': 1.0, 'weight': 1.0}])

        total_compliance = 0.0
        total_dc = np.zeros(self.n)

        # 2. 🌟 遍历全部工况，求解并累加灵敏度 (Multi-Loadcase 核心)
        for load in active_loads:
            # 清理旧载荷，加载当前遍历到的新载荷
            self.fea.clear_loads()
            self.fea.add_pressure_load(load['mag'], load['dir'], surface_tag=2)

            # 求解有限元位移
            u = self.fea.solve()

            # 提取单工况下的柔度与灵敏度
            c_i = self.fea.compute_compliance(u)
            dc_i = self.fea.compute_sensitivity(u)

            # 按权重将破坏力叠加（默认全工况等权综合）
            total_compliance += c_i * load['weight']
            total_dc += dc_i * load['weight']

        # 3. 对叠加后的总灵敏度进行敏度滤波（防止棋盘格现象）
        dc_f = self.filter_sensitivities(total_dc)

        # 4. 融合铸造约束 (如果开启)
        if self.casting_constraint is not None:
            g_cast, dg_cast = self.casting_constraint.get_constraint(self.rho.x.array)
            # 将铸造灵敏度惩罚叠加到结构灵敏度上
            dc_f += dg_cast.flatten() * 0.5

            # 5. 调用 OC 法更新密度场
        rho_new = self._oc_update(self.rho.x.array, dc_f)

        # 6. 计算收敛变化量
        change = np.max(np.abs(rho_new - self.rho.x.array))
        self.rho.x.array[:] = rho_new

        # 记录历史数据
        self.history['compliance'].append(total_compliance)
        self.history['vol'].append(np.mean(rho_new))
        self.history['change'].append(change)

        return total_compliance, change, total_dc

    def optimize(self):
        """为 PyQt 界面提供的迭代生成器"""
        for i in range(self.max_iter):
            c, change, dc = self.optimize_step(i)
            # 将当前状态 yield 传递给 GUI 线程进行实时平滑渲染
            yield self.rho.x.array, dc, c, change

            if change < self.tol:
                break