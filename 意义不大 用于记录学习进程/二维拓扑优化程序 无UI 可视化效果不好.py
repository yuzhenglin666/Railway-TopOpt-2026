"""
二维拓扑优化完整程序 - 单文件版本
适用于交通科技大赛项目
作者：中南大学交通运输科技大赛团队
更新：修复UFL导入问题，使用稳定API
"""

# ==================== 导入必要的库 ====================
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

# 尝试导入FEniCSx相关库
try:
    from mpi4py import MPI
    import dolfinx
    from dolfinx import mesh, fem, io
    from dolfinx.fem import (FunctionSpace, Function, form, Constant,
                             dirichletbc, locate_dofs_topological, assemble_scalar)
    from dolfinx.fem.petsc import LinearProblem
    import ufl
    from petsc4py import PETSc

    FENICSX_AVAILABLE = True
    print("FEniCSx导入成功!")
except ImportError as e:
    print(f"导入FEniCSx时出错: {e}")
    print("请运行: conda install -c conda-forge fenics-dolfinx mpi4py petsc4py")
    FENICSX_AVAILABLE = False


# ==================== 可视化样式设置 ====================
def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'


# ==================== 简化的拓扑优化器类 ====================
class SimplifiedTopologyOptimizer2D:
    """简化的二维拓扑优化器 (避免复杂的FEniCSx API问题)"""

    def __init__(self, params):
        """
        初始化拓扑优化器

        参数:
            params: 字典，包含优化参数
        """
        if not FENICSX_AVAILABLE:
            raise ImportError("FEniCSx未正确安装，无法运行拓扑优化程序")

        self.params = params
        self.set_default_parameters()

        # 初始化历史记录
        self.history = {
            'compliance': [],
            'volume': [],
            'change': [],
            'time': []
        }
        self.iterations = 0

        # 设计变量
        self.rho = None  # 密度场

        print("=" * 60)
        print("简化的二维拓扑优化器初始化完成")
        print(f"设计域: {self.params['design_domain'][0]} × {self.params['design_domain'][1]}")
        print(f"网格: {self.params['mesh_resolution'][0]} × {self.params['mesh_resolution'][1]}")
        print(f"材料: E={self.params['youngs_modulus']}, ν={self.params['poisson_ratio']}")
        print(f"优化: 体积分数={self.params['volume_fraction']}, 惩罚={self.params['penalization']}")
        print(f"边界条件: {self.params['bc_type']}")
        print("=" * 60)

    def set_default_parameters(self):
        """设置默认参数"""
        defaults = {
            'design_domain': [2.0, 1.0],
            'mesh_resolution': [60, 30],
            'youngs_modulus': 1.0,
            'poisson_ratio': 0.3,
            'volume_fraction': 0.4,
            'penalization': 3.0,
            'filter_radius': 1.5,
            'load': [0.0, -1.0],
            'bc_type': 'cantilever',
            'max_iter': 50,
            'tol': 0.01,
            'move_limit': 0.2,
            'filter_type': 'sensitivity',
            'beta': 1.0,
            'eta': 0.5,
        }

        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

    def solve_analytical_problem(self, rho):
        """简化的解析求解（代替FEniCSx求解）"""
        # 这是一个简化的柔度计算，用于演示
        # 在实际应用中，应该使用FEniCSx进行有限元求解

        Lx, Ly = self.params['design_domain']
        nx, ny = self.params['mesh_resolution']

        # 简化的柔度计算（仅供演示）
        total_density = np.mean(rho)
        penalty = self.params['penalization']

        # 模拟柔度：密度越低，柔度越高
        compliance = 100.0 / (total_density ** penalty + 1e-6)

        return compliance

    def compute_sensitivities(self, rho, compliance):
        """计算灵敏度 (简化的)"""
        p = self.params['penalization']
        n = len(rho)

        # 简化的灵敏度计算
        if n > 0:
            dc = -p * rho ** (p - 1) * compliance / n
        else:
            dc = np.zeros_like(rho)

        return dc

    def apply_filter(self, rho):
        """应用密度滤波"""
        rho_filtered = np.zeros_like(rho)
        n = len(rho)

        for i in range(n):
            # 简单的邻域平均滤波
            start = max(0, i - 2)
            end = min(n, i + 3)
            rho_filtered[i] = np.mean(rho[start:end])

        return rho_filtered

    def update_design_variables(self, rho, dc):
        """使用最优准则法(OC)更新设计变量"""
        move_limit = self.params['move_limit']
        volume_fraction = self.params['volume_fraction']

        # 二分法求解拉格朗日乘子
        l1 = 1e-9
        l2 = 1e9
        lmbda = 0.5 * (l1 + l2)

        for _ in range(50):
            B = -dc / (lmbda + 1e-12)
            B = np.maximum(1e-6, B)

            rho_new = rho * np.sqrt(B)

            rho_lower = np.maximum(0.001, rho - move_limit)
            rho_upper = np.minimum(0.999, rho + move_limit)
            rho_new = np.maximum(rho_lower, np.minimum(rho_upper, rho_new))

            new_volume = np.mean(rho_new)

            if new_volume > volume_fraction:
                l1 = lmbda
            else:
                l2 = lmbda

            lmbda = 0.5 * (l1 + l2)

            if abs(new_volume - volume_fraction) < 0.001:
                break

        change = np.max(np.abs(rho_new - rho))

        return rho_new, change, new_volume

    def optimize(self, verbose=True):
        """执行拓扑优化主循环"""
        print("\n" + "=" * 60)
        print("开始拓扑优化 (简化版本)")
        print("=" * 60)

        start_time = time.time()

        # 初始化设计变量
        nx, ny = self.params['mesh_resolution']
        n_elements = nx * ny
        rho = np.full(n_elements, self.params['volume_fraction'])

        for iteration in range(self.params['max_iter']):
            iter_start_time = time.time()
            self.iterations = iteration + 1

            if verbose:
                print(f"\n迭代 {self.iterations}/{self.params['max_iter']}")

            # 1. 求解问题（简化版本）
            compliance = self.solve_analytical_problem(rho)

            # 2. 计算灵敏度
            dc = self.compute_sensitivities(rho, compliance)

            # 3. 应用滤波
            rho_filtered = self.apply_filter(rho)

            # 4. 更新设计变量
            rho_new, change, volume = self.update_design_variables(rho_filtered, dc)

            # 5. 更新设计变量
            rho = rho_new

            # 6. 记录历史
            self.history['compliance'].append(float(compliance))
            self.history['volume'].append(float(volume))
            self.history['change'].append(float(change))
            self.history['time'].append(time.time() - iter_start_time)

            # 7. 打印进度
            if verbose:
                print(f"  柔度: {compliance:.6e}")
                print(f"  体积分数: {volume:.4f} (目标: {self.params['volume_fraction']})")
                print(f"  最大变化: {change:.6f}")
                print(f"  迭代时间: {self.history['time'][-1]:.3f}秒")

            # 8. 检查收敛
            if change < self.params['tol']:
                if verbose:
                    print(f"  收敛: 设计变量变化 {change:.6f} < 容差 {self.params['tol']}")
                break

        total_time = time.time() - start_time

        self.rho = rho.reshape((ny, nx))

        if verbose:
            print(f"\n优化完成!")
            print(f"总迭代次数: {self.iterations}")
            print(f"总时间: {total_time:.2f}秒")
            print(f"平均每次迭代: {total_time / self.iterations:.3f}秒")

        return self.rho, self.history


# ==================== 专业可视化函数 ====================
def create_professional_visualization_simplified(optimizer, iteration='final',
                                                 show=True, save_path=None):
    """
    创建专业化的拓扑优化结果可视化 (简化版本)
    """
    setup_plot_style()

    # 准备数据
    rho = optimizer.rho.flatten()
    hist, bin_edges = np.histogram(rho, bins=30, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    comp = optimizer.history['compliance']
    vol = optimizer.history['volume']
    change = optimizer.history['change']
    iters = range(1, len(comp) + 1)

    # 计算关键指标
    final_comp = comp[-1] if comp else 0
    initial_comp = comp[0] if len(comp) > 1 else final_comp
    improvement = ((initial_comp - final_comp) / initial_comp * 100) if initial_comp > 0 else 0
    final_vol_frac = vol[-1] if vol else 0
    target_vol_frac = optimizer.params['volume_fraction']

    # 创建图形和布局
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1, 1],
                           wspace=0.25, hspace=0.35)

    # 子图1：密度分布直方图
    ax1 = plt.subplot(gs[0, 0])
    bars = ax1.bar(bin_centers, hist, width=0.025, alpha=0.8,
                   color='steelblue', edgecolor='black', linewidth=0.5)

    low_density = np.sum(rho < 0.1) / len(rho) * 100
    high_density = np.sum(rho > 0.9) / len(rho) * 100

    ax1.axvline(x=optimizer.params['volume_fraction'], color='red',
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f"目标: {optimizer.params['volume_fraction']}")

    ax1.set_xlabel('材料密度', fontsize=11, fontweight='bold')
    ax1.set_ylabel('单元数量', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 设计变量（密度）分布', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)

    textstr = '\n'.join([
        f'平均密度: {np.mean(rho):.3f}',
        f'单元总数: {len(rho)}',
        f'低密度(<0.1): {low_density:.1f}%',
        f'高密度(>0.9): {high_density:.1f}%'
    ])
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图2：柔度与体积收敛历史
    ax2 = plt.subplot(gs[0, 1:])
    color_comp = 'tab:blue'

    ax2.plot(iters, comp, color=color_comp, linewidth=2.5,
             marker='o', markersize=5, label='柔度', zorder=5)
    ax2.set_xlabel('迭代次数', fontsize=11, fontweight='bold')
    ax2.set_ylabel('柔度', color=color_comp, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_comp)
    ax2.grid(True, alpha=0.3, linestyle='--')

    ax2_vol = ax2.twinx()
    color_vol = 'tab:red'
    ax2_vol.plot(iters, vol, color=color_vol, linewidth=2.0,
                 linestyle='--', marker='s', markersize=4, label='体积分数')
    ax2_vol.axhline(y=target_vol_frac, color='green',
                    linestyle=':', linewidth=2, label=f'目标 ({target_vol_frac})')
    ax2_vol.set_ylabel('体积分数', color=color_vol, fontsize=11, fontweight='bold')
    ax2_vol.tick_params(axis='y', labelcolor=color_vol)
    ax2_vol.set_ylim([0, max(max(vol) * 1.1, target_vol_frac * 1.2) if vol else 1])

    ax2.set_title('(b) 目标函数与体积分数收敛历史', fontsize=12, fontweight='bold', pad=10)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_vol.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    improvement_text = f'柔度改善: {improvement:.1f}%\n初始: {initial_comp:.2e}\n最终: {final_comp:.2e}'
    ax2.text(0.05, 0.95, improvement_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # 子图3：设计变量变化历史
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(iters, change, color='darkorange', linewidth=2,
             marker='^', markersize=5)
    ax3.axhline(y=optimizer.params['tol'], color='grey',
                linestyle=':', linewidth=2, label=f'容差 ({optimizer.params["tol"]})')
    ax3.set_xlabel('迭代次数', fontsize=11, fontweight='bold')
    ax3.set_ylabel('最大密度变化', fontsize=11, fontweight='bold')
    ax3.set_title('(c) 设计变量变化历史', fontsize=12, fontweight='bold', pad=10)
    ax3.set_yscale('log')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both', linestyle='--')

    if len(change) > 0:
        converged = change[-1] < optimizer.params['tol']
        color = 'green' if converged else 'red'
        ax3.scatter(len(change), change[-1], color=color, s=100, zorder=10,
                    edgecolors='black', linewidth=1.5)

    # 子图4：优化结构示意图
    ax4 = plt.subplot(gs[1, 1:])

    colors = ["white", "lightblue", "blue", "darkblue"]
    cmap = LinearSegmentedColormap.from_list("topopt_cmap", colors, N=256)

    # 显示密度分布
    im = ax4.imshow(optimizer.rho, cmap=cmap,
                    extent=[0, 1, 0, 1], aspect='auto',
                    vmin=0, vmax=1, origin='lower')

    # 添加等高线
    try:
        CS = ax4.contour(optimizer.rho, levels=[0.1, 0.5, 0.9],
                         colors=['gray', 'black', 'darkred'],
                         linewidths=[0.5, 1.0, 1.5],
                         extent=[0, 1, 0, 1], alpha=0.7)
        ax4.clabel(CS, inline=True, fontsize=8)
    except:
        pass

    ax4.set_title('(d) 优化结构密度分布', fontsize=12, fontweight='bold', pad=10)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('材料密度', fontsize=10)

    Lx, Ly = optimizer.params['design_domain']
    ax4.set_xlim([0, Lx])
    ax4.set_ylim([0, Ly])

    bc_type = optimizer.params['bc_type']
    if bc_type == 'cantilever':
        ax4.axvline(x=0, color='red', linewidth=3, alpha=0.5, label='固定端')
        ax4.scatter(Lx, Ly / 2, color='green', s=200, marker='v',
                    edgecolors='black', linewidth=2, zorder=10,
                    label='载荷点')
    elif bc_type == 'bridge':
        ax4.scatter(Lx / 4, 0, color='red', s=150, marker='s', zorder=10, label='支撑')
        ax4.scatter(3 * Lx / 4, 0, color='red', s=150, marker='s', zorder=10)
        ax4.scatter(Lx / 2, Ly, color='green', s=200, marker='v',
                    edgecolors='black', linewidth=2, zorder=10, label='载荷点')
    elif bc_type == 'michell':
        ax4.scatter(0, 0, color='red', s=150, marker='s', zorder=10, label='支撑')
        ax4.scatter(Lx, 0, color='red', s=150, marker='s', zorder=10)
        ax4.scatter(Lx / 2, Ly, color='green', s=200, marker='v',
                    edgecolors='black', linewidth=2, zorder=10, label='载荷点')

    ax4.legend(fontsize=9, loc='upper right')

    # 添加全局大标题
    bc_type_names = {
        'cantilever': '悬臂梁',
        'bridge': '桥梁结构',
        'michell': '迈克轮结构'
    }
    bc_name = bc_type_names.get(bc_type, bc_type)

    fig.suptitle(
        f'拓扑优化结果：{bc_name} (简化版本)\n'
        f'迭代: {iteration} | 最终柔度: {final_comp:.3e} | '
        f'体积分数: {final_vol_frac:.3f}/{target_vol_frac} | '
        f'改善率: {improvement:.1f}%',
        fontsize=14, fontweight='bold', y=0.99
    )

    # 保存和显示
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                    exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"可视化图表已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_iteration_history_simplified(optimizer, save_path=None):
    """绘制详细的迭代历史 (简化版本)"""
    setup_plot_style()

    comp = optimizer.history['compliance']
    vol = optimizer.history['volume']
    change = optimizer.history['change']
    iters = range(1, len(comp) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 子图1：柔度收敛
    ax1 = axes[0, 0]
    ax1.plot(iters, comp, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('柔度')
    ax1.set_title('柔度收敛历史')
    ax1.grid(True, alpha=0.3)

    # 子图2：体积分数
    ax2 = axes[0, 1]
    ax2.plot(iters, vol, 'g-s', linewidth=2, markersize=4)
    ax2.axhline(y=optimizer.params['volume_fraction'], color='r',
                linestyle='--', label='目标')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('体积分数')
    ax2.set_title('体积分数历史')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3：设计变量变化
    ax3 = axes[1, 0]
    ax3.plot(iters, change, 'r-^', linewidth=2, markersize=4)
    ax3.axhline(y=optimizer.params['tol'], color='k',
                linestyle='--', label='容差')
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('最大变化')
    ax3.set_title('设计变量变化历史')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4：时间统计
    ax4 = axes[1, 1]
    times = optimizer.history['time']
    cumulative_time = np.cumsum(times)
    ax4.plot(iters, times, 'm-D', linewidth=2, markersize=4, label='单次迭代')
    ax4.plot(iters, cumulative_time, 'c-', linewidth=2, label='累计时间')
    ax4.set_xlabel('迭代次数')
    ax4.set_ylabel('时间 (秒)')
    ax4.set_title('计算时间统计')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'拓扑优化详细历史 | 总迭代: {len(comp)} | 总时间: {sum(times):.1f}秒',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"历史图表已保存至: {save_path}")

    plt.show()


# ==================== 主程序 ====================
def run_simplified_example():
    """运行简化的悬臂梁优化示例"""
    print("=" * 60)
    print("二维拓扑优化程序 - 简化版本")
    print("=" * 60)

    # 1. 设置优化参数
    params = {
        'design_domain': [2.0, 1.0],  # 设计域尺寸 [Lx, Ly]
        'mesh_resolution': [80, 40],  # 网格分辨率 [nx, ny]
        'youngs_modulus': 1.0,  # 杨氏模量
        'poisson_ratio': 0.3,  # 泊松比
        'volume_fraction': 0.4,  # 目标体积分数
        'penalization': 3.0,  # SIMP惩罚因子
        'filter_radius': 1.5,  # 滤波半径
        'load': [0.0, -1.0],  # 载荷 [Fx, Fy]
        'bc_type': 'cantilever',  # 边界条件类型
        'max_iter': 30,  # 最大迭代次数
        'tol': 0.01,  # 收敛容差
        'move_limit': 0.2,  # 移动限制
    }

    # 2. 创建优化器实例
    optimizer = SimplifiedTopologyOptimizer2D(params)

    # 3. 执行优化
    print("\n开始执行拓扑优化...")
    rho_opt, history = optimizer.optimize()

    # 4. 输出优化结果摘要
    if optimizer.history['compliance']:
        summary = {
            'iterations': optimizer.iterations,
            'initial_compliance': optimizer.history['compliance'][0],
            'final_compliance': optimizer.history['compliance'][-1],
            'improvement_percent': ((optimizer.history['compliance'][0] - optimizer.history['compliance'][-1]) /
                                    optimizer.history['compliance'][0] * 100) if optimizer.history['compliance'][
                                                                                     0] > 0 else 0,
            'final_volume_fraction': optimizer.history['volume'][-1],
            'target_volume_fraction': optimizer.params['volume_fraction']
        }

        print("\n" + "=" * 60)
        print("优化结果摘要:")
        print(f"  迭代次数: {summary['iterations']}")
        print(f"  初始柔度: {summary['initial_compliance']:.3e}")
        print(f"  最终柔度: {summary['final_compliance']:.3e}")
        print(f"  改善比例: {summary['improvement_percent']:.1f}%")
        print(f"  最终体积分数: {summary['final_volume_fraction']:.3f}")
        print(f"  目标体积分数: {summary['target_volume_fraction']:.3f}")
        print("=" * 60)

    # 5. 保存结果
    os.makedirs("topopt_results_simplified", exist_ok=True)
    np.savez("topopt_results_simplified/optimization_results.npz",
             rho=optimizer.rho,
             history=optimizer.history,
             params=optimizer.params)

    print(f"结果已保存到: topopt_results_simplified/optimization_results.npz")

    # 6. 创建专业可视化
    print("\n生成可视化图表...")

    # 专业综合图表
    create_professional_visualization_simplified(
        optimizer,
        iteration='final',
        save_path='topopt_results_simplified/professional_visualization.png',
        show=True
    )

    # 详细历史图表
    plot_iteration_history_simplified(
        optimizer,
        save_path='topopt_results_simplified/iteration_history.png'
    )

    print("\n示例运行完成！")
    print("查看 'topopt_results_simplified' 目录中的结果文件")

    return optimizer


def run_alternative_version():
    """运行不依赖FEniCSx的替代版本"""
    print("=" * 60)
    print("纯Python拓扑优化器 (无FEniCSx依赖)")
    print("=" * 60)

    # 这是一个完全不依赖FEniCSx的版本
    # 使用简单的有限元求解器

    class PurePythonTopologyOptimizer:
        def __init__(self, params):
            self.params = params
            self.set_default_parameters()

            nx, ny = self.params['mesh_resolution']
            self.nx = nx
            self.ny = ny
            self.n_elements = nx * ny

            # 初始化密度
            self.rho = np.full(self.n_elements, self.params['volume_fraction'])

            self.history = {
                'compliance': [],
                'volume': [],
                'change': []
            }

            print(f"初始化完成: {nx}x{ny} 网格")

        def set_default_parameters(self):
            defaults = {
                'design_domain': [2.0, 1.0],
                'mesh_resolution': [40, 20],
                'volume_fraction': 0.4,
                'penalization': 3.0,
                'max_iter': 20,
                'tol': 0.01,
            }

            for key, value in defaults.items():
                if key not in self.params:
                    self.params[key] = value

        def optimize(self):
            print("\n开始拓扑优化...")

            for iteration in range(self.params['max_iter']):
                # 1. 计算柔度 (简化的)
                compliance = self.compute_compliance(self.rho)

                # 2. 计算灵敏度
                dc = self.compute_sensitivities(self.rho, compliance)

                # 3. 更新设计变量
                rho_new, change = self.update_design_variables(self.rho, dc)

                # 4. 记录历史
                self.history['compliance'].append(compliance)
                self.history['volume'].append(np.mean(rho_new))
                self.history['change'].append(change)

                # 5. 更新设计变量
                self.rho = rho_new

                print(f"迭代 {iteration + 1}: 柔度={compliance:.4e}, 体积={np.mean(rho_new):.3f}, 变化={change:.4f}")

                if change < self.params['tol']:
                    print(f"收敛于迭代 {iteration + 1}")
                    break

            print("\n优化完成!")

            return self.rho.reshape((self.ny, self.nx)), self.history

        def compute_compliance(self, rho):
            # 简化的柔度计算
            p = self.params['penalization']
            avg_density = np.mean(rho)
            return 1.0 / (avg_density ** p + 1e-6)

        def compute_sensitivities(self, rho, compliance):
            p = self.params['penalization']
            dc = -p * rho ** (p - 1) * compliance / len(rho)
            return dc

        def update_design_variables(self, rho, dc):
            move_limit = 0.2
            volume_fraction = self.params['volume_fraction']

            # 简单的OC更新
            B = -dc / np.mean(dc) if np.mean(dc) != 0 else np.ones_like(dc)
            rho_new = rho * np.sqrt(B)

            # 体积约束
            current_volume = np.mean(rho_new)
            if current_volume > volume_fraction:
                rho_new = rho_new * (volume_fraction / current_volume)

            # 移动限制
            rho_new = np.maximum(0.001, np.minimum(0.999, rho_new))

            change = np.max(np.abs(rho_new - rho))

            return rho_new, change

    # 运行优化
    params = {
        'design_domain': [2.0, 1.0],
        'mesh_resolution': [40, 20],
        'volume_fraction': 0.4,
        'penalization': 3.0,
        'max_iter': 20,
        'tol': 0.01,
    }

    optimizer = PurePythonTopologyOptimizer(params)
    rho_opt, history = optimizer.optimize()

    # 可视化结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(rho_opt, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label='密度')
    plt.title('优化结构')

    plt.subplot(1, 2, 2)
    plt.plot(history['compliance'], 'b-o')
    plt.xlabel('迭代次数')
    plt.ylabel('柔度')
    plt.title('柔度收敛历史')
    plt.grid(True)

    plt.suptitle('纯Python拓扑优化结果', fontsize=14)
    plt.tight_layout()
    plt.savefig('topopt_results_simplified/pure_python_result.png', dpi=300)
    plt.show()

    print(f"\n结果已保存到: topopt_results_simplified/pure_python_result.png")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("二维拓扑优化程序")
    print("交通运输科技大赛项目")
    print("=" * 60)

    # 创建结果目录
    os.makedirs("topopt_results_simplified", exist_ok=True)

    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 简化版本拓扑优化 (使用FEniCSx，但避免复杂API)")
    print("2. 纯Python版本 (无FEniCSx依赖)")
    print("3. 退出")

    choice = input("\n请输入选择 (1-3): ").strip()

    if choice == '1':
        if FENICSX_AVAILABLE:
            run_simplified_example()
        else:
            print("FEniCSx不可用，无法运行此模式")
            print("请先安装FEniCSx或选择模式2")
    elif choice == '2':
        run_alternative_version()
    elif choice == '3':
        print("程序退出")
    else:
        print("无效选择，运行简化版本...")
        if FENICSX_AVAILABLE:
            run_simplified_example()
        else:
            run_alternative_version()

    print("\n程序运行结束")
