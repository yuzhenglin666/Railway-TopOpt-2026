#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拓扑优化系统 - 多载荷组合版
前端：ttkbootstrap 美化界面
后端：FEniCSx + nlopt MMA + 铸造约束
可视化：matplotlib 显示密度切片和收敛曲线
支持用户自由组合垂向、横向、纵向载荷，自动选择最危险工况
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
import threading
import queue
import time
from pathlib import Path

# 添加当前目录到模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入后端模块
from mesh_utils import generate_cantilever_mesh, load_mesh
from fea_solver import FEASolver
from topology import TopologyOptimizer
from constraints import create_casting_constraint

# ========== 优化线程类 ==========
class OptimizationThread(threading.Thread):
    def __init__(self, params, queue):
        super().__init__()
        self.params = params
        self.queue = queue
        self.stop_flag = False
        self.daemon = True

    def run(self):
        try:
            # 加载网格
            self.queue.put(('log', "加载网格..."))
            domain, facet_tags = load_mesh(self.params['mesh']['filename'])

            # 初始化密度场
            from dolfinx import fem
            V_rho = fem.functionspace(domain, ("DG", 0))
            rho = fem.Function(V_rho, name="Density")
            np.random.seed(42)
            rho.x.array[:] = self.params['optimization']['vol_frac'] + 0.1 * (np.random.rand(len(rho.x.array)) - 0.5)
            rho.x.array[:] = np.clip(rho.x.array, 0.001, 0.999)

            # 创建有限元求解器（不含载荷）
            fea = FEASolver(domain, facet_tags, rho, self.params['material'])

            # 构建载荷组合列表
            load_combos = []
            if self.params['load']['enable_vert']:
                load_combos.append(((0,0,-1), self.params['load']['mag_vert']))
            if self.params['load']['enable_lat']:
                load_combos.append(((0,1,0), self.params['load']['mag_lat']))
            if self.params['load']['enable_long']:
                load_combos.append(((1,0,0), self.params['load']['mag_long']))

            # 如果没有选择任何载荷，默认垂向
            if not load_combos:
                load_combos = [((0,0,-1), 1.0)]

            # 计算每种组合的柔度，找出最危险工况（柔度最大）
            self.queue.put(('log', f"正在评估 {len(load_combos)} 种载荷组合..."))
            worst_dir = None
            worst_mag = None
            worst_c = -np.inf

            for dir, mag in load_combos:
                fea.clear_loads()
                fea.add_pressure_load(mag, dir, surface_tag=2)  # 假设都作用在右端面
                uh = fea.solve()
                c = fea.compute_compliance(uh)
                self.queue.put(('log', f"  组合 {dir} 柔度 = {c:.2e}"))
                if c > worst_c:
                    worst_c = c
                    worst_dir = dir
                    worst_mag = mag

            self.queue.put(('log', f"最危险工况: 方向={worst_dir}, 大小={worst_mag}, 柔度={worst_c:.2e}"))

            # 用最危险工况重新设置载荷
            fea.clear_loads()
            fea.add_pressure_load(worst_mag, worst_dir, surface_tag=2)

            # 创建铸造约束
            casting_constraint = None
            if self.params['casting']['enabled']:
                casting_constraint = create_casting_constraint(
                    domain, rho,
                    material_type=self.params['casting']['material_type'],
                    draft_dir=self.params['casting']['draft_direction']
                )
                casting_constraint.min_thickness = self.params['casting']['min_thickness']
                casting_constraint.draft_angle = self.params['casting']['draft_angle']

            # 创建优化器
            opt = TopologyOptimizer(rho, fea, self.params['optimization'],
                                    casting_constraint=casting_constraint)

            # 优化循环
            n = len(rho.x.array)
            vol_frac = self.params['optimization']['vol_frac']
            rmin = self.params['optimization']['rmin']
            max_iter = self.params['optimization']['max_iter']
            p = self.params['material']['p']
            tol = 1e-2

            history = {'compliance': [], 'vol': [], 'change': []}

            for it in range(max_iter):
                if self.stop_flag:
                    self.queue.put(('log', "优化被用户停止"))
                    break

                uh = fea.solve()
                c = fea.compute_compliance(uh)
                energy = fea.compute_energy_density(uh)

                rho_arr = rho.x.array.copy()
                dc_raw = -p * rho_arr**(p-1) * energy
                if rmin > 0:
                    dc = opt.filter_sensitivities(dc_raw)
                else:
                    dc = dc_raw

                current_vol = np.mean(rho_arr)
                g_vol = current_vol - vol_frac
                dg_vol = (1.0 / n) * np.ones(n)

                import nlopt
                obj_data = {'c': c, 'dc': dc}
                def obj_func(x, grad):
                    if grad.size > 0:
                        grad[:] = obj_data['dc']
                    return obj_data['c']
                vol_data = {'g': g_vol, 'dg': dg_vol}
                def vol_constraint(x, grad):
                    if grad.size > 0:
                        grad[:] = vol_data['dg']
                    return vol_data['g']

                opt_nl = nlopt.opt(nlopt.LD_MMA, n)
                opt_nl.set_lower_bounds(0.001)
                opt_nl.set_upper_bounds(1.0)
                opt_nl.set_min_objective(obj_func)
                opt_nl.add_inequality_constraint(vol_constraint, 1e-8)

                if casting_constraint is not None:
                    g_cast, dg_cast = casting_constraint.get_constraint(rho_arr)
                    cast_data = {'g': g_cast[0], 'dg': dg_cast[0, :]}
                    def cast_constraint(x, grad):
                        if grad.size > 0:
                            grad[:] = cast_data['dg']
                        return cast_data['g']
                    opt_nl.add_inequality_constraint(cast_constraint, 1e-8)

                opt_nl.set_xtol_rel(1e-4)

                try:
                    x_new = opt_nl.optimize(rho_arr)
                except Exception as e:
                    self.queue.put(('log', f"MMA 优化出错: {e}"))
                    break

                change = np.max(np.abs(x_new - rho_arr))
                rho.x.array[:] = x_new

                history['compliance'].append(c)
                history['vol'].append(np.mean(x_new))
                history['change'].append(change)

                self.queue.put(('progress', (it+1, max_iter, c, np.mean(x_new), change)))

                if change < tol:
                    self.queue.put(('log', f"收敛于迭代 {it+1}"))
                    break

            self.queue.put(('done', (rho, history, domain)))

        except Exception as e:
            self.queue.put(('log', f"错误: {str(e)}"))
            import traceback
            traceback.print_exc()


# ========== 主 GUI 类 ==========
class OptimizedGUI:
    def __init__(self):
        self.root = tb.Window(themename="superhero")
        self.root.title("拓扑优化系统 - 多载荷版")
        self.root.geometry("1300x800")

        # 设置字体
        self.default_font = ('Helvetica', 10)
        self.title_font = ('Helvetica', 12, 'bold')
        self.status_font = ('Helvetica', 11, 'bold')
        self.log_font = ('Consolas', 9)

        self.optimizer = None
        self.thread = None
        self.running = False
        self.queue = queue.Queue()
        self.domain = None          # 保存网格对象，用于绘图
        self.history = {'compliance': [], 'vol': [], 'change': []}
        self.rho_array = None       # 当前密度数组（一维）

        self.create_menu()
        self.create_main_layout()
        self.check_queue()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开结果", command=self.load_results)
        file_menu.add_command(label="保存结果", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menubar)

    def create_main_layout(self):
        left_frame = tb.Frame(self.root, width=400, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        right_frame = tb.Frame(self.root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_control_panel(left_frame)

        self.notebook = tb.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.frame_2d = tb.Frame(self.notebook)
        self.notebook.add(self.frame_2d, text="密度切片")
        self.create_2d_view()

        self.frame_3d = tb.Frame(self.notebook)
        self.notebook.add(self.frame_3d, text="3D 预览")
        self.create_3d_view()

        self.frame_plot = tb.Frame(self.notebook)
        self.notebook.add(self.frame_plot, text="收敛曲线")
        self.create_plot_view()

    def create_control_panel(self, parent):
        # 几何参数
        geo_group = tb.LabelFrame(parent, text="几何与网格", padx=10, pady=10, font=self.title_font)
        geo_group.pack(fill=tk.X, pady=5)

        row = 0
        label_kw = {'font': self.default_font}
        entry_kw = {'font': self.default_font, 'width': 10}

        tb.Label(geo_group, text="Lx (m):", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.design_x = tb.Entry(geo_group, **entry_kw)
        self.design_x.insert(0, "2.0")
        self.design_x.grid(row=row, column=1, padx=5)

        tb.Label(geo_group, text="Ly (m):", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.design_y = tb.Entry(geo_group, **entry_kw)
        self.design_y.insert(0, "0.5")
        self.design_y.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(geo_group, text="Lz (m):", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.design_z = tb.Entry(geo_group, **entry_kw)
        self.design_z.insert(0, "0.5")
        self.design_z.grid(row=row, column=1, padx=5)

        tb.Label(geo_group, text="网格尺寸 lc:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.lc = tb.Entry(geo_group, **entry_kw)
        self.lc.insert(0, "0.2")
        self.lc.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(geo_group, text="nx:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.mesh_x = tb.Entry(geo_group, **entry_kw)
        self.mesh_x.insert(0, "40")
        self.mesh_x.grid(row=row, column=1, padx=5)

        tb.Label(geo_group, text="ny:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.mesh_y = tb.Entry(geo_group, **entry_kw)
        self.mesh_y.insert(0, "10")
        self.mesh_y.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(geo_group, text="nz:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.mesh_z = tb.Entry(geo_group, **entry_kw)
        self.mesh_z.insert(0, "10")
        self.mesh_z.grid(row=row, column=1, padx=5)

        # 材料参数
        mat_group = tb.LabelFrame(parent, text="材料", padx=10, pady=10, font=self.title_font)
        mat_group.pack(fill=tk.X, pady=5)

        row = 0
        tb.Label(mat_group, text="杨氏模量 (GPa):", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.E = tb.Entry(mat_group, **entry_kw)
        self.E.insert(0, "1.0")
        self.E.grid(row=row, column=1, padx=5)

        tb.Label(mat_group, text="泊松比:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.nu = tb.Entry(mat_group, **entry_kw)
        self.nu.insert(0, "0.3")
        self.nu.grid(row=row, column=3, padx=5)

        # 载荷组合区域
        load_group = tb.LabelFrame(parent, text="载荷组合", padx=10, pady=10, font=self.title_font)
        load_group.pack(fill=tk.X, pady=5)

        self.load_vert_var = tk.BooleanVar(value=True)
        self.load_lat_var = tk.BooleanVar(value=False)
        self.load_long_var = tk.BooleanVar(value=False)

        cb_vert = tb.Checkbutton(load_group, text="垂向力 (Z)", variable=self.load_vert_var, bootstyle="primary")
        cb_vert.grid(row=0, column=0, sticky=tk.W, pady=2)
        self.mag_vert = tb.Entry(load_group, width=10)
        self.mag_vert.insert(0, "1.0")
        self.mag_vert.grid(row=0, column=1, padx=5)

        cb_lat = tb.Checkbutton(load_group, text="横向力 (Y)", variable=self.load_lat_var, bootstyle="primary")
        cb_lat.grid(row=1, column=0, sticky=tk.W, pady=2)
        self.mag_lat = tb.Entry(load_group, width=10)
        self.mag_lat.insert(0, "0.5")
        self.mag_lat.grid(row=1, column=1, padx=5)

        cb_long = tb.Checkbutton(load_group, text="纵向力 (X)", variable=self.load_long_var, bootstyle="primary")
        cb_long.grid(row=2, column=0, sticky=tk.W, pady=2)
        self.mag_long = tb.Entry(load_group, width=10)
        self.mag_long.insert(0, "0.5")
        self.mag_long.grid(row=2, column=1, padx=5)

        # 优化参数
        opt_group = tb.LabelFrame(parent, text="优化参数", padx=10, pady=10, font=self.title_font)
        opt_group.pack(fill=tk.X, pady=5)

        row = 0
        tb.Label(opt_group, text="体积分数:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.vol_frac = tb.Entry(opt_group, **entry_kw)
        self.vol_frac.insert(0, "0.35")
        self.vol_frac.grid(row=row, column=1, padx=5)

        tb.Label(opt_group, text="惩罚因子 p:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.penal = tb.Entry(opt_group, **entry_kw)
        self.penal.insert(0, "3.0")
        self.penal.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(opt_group, text="滤波半径:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.rmin = tb.Entry(opt_group, **entry_kw)
        self.rmin.insert(0, "0.1")
        self.rmin.grid(row=row, column=1, padx=5)

        tb.Label(opt_group, text="最大迭代:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.max_iter = tb.Entry(opt_group, **entry_kw)
        self.max_iter.insert(0, "50")
        self.max_iter.grid(row=row, column=3, padx=5)

        # 铸造约束
        cast_group = tb.LabelFrame(parent, text="铸造约束", padx=10, pady=10, font=self.title_font)
        cast_group.pack(fill=tk.X, pady=5)

        row = 0
        self.enable_cast_var = tk.BooleanVar(value=False)
        self.enable_cast_cb = tb.Checkbutton(cast_group, text="启用铸造约束", variable=self.enable_cast_var,
                                             bootstyle="primary", command=self.toggle_cast)
        self.enable_cast_cb.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=5)

        row += 1
        tb.Label(cast_group, text="材料类型:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.material_type = tb.Combobox(cast_group, values=["steel", "cast_iron", "aluminum", "stainless"],
                                         width=10, font=self.default_font, state="disabled")
        self.material_type.set("steel")
        self.material_type.grid(row=row, column=1, padx=5)

        tb.Label(cast_group, text="最小壁厚 (mm):", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.min_thick = tb.Entry(cast_group, **entry_kw, state="disabled")
        self.min_thick.insert(0, "2.0")
        self.min_thick.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(cast_group, text="拔模斜度 (°):", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.draft_angle = tb.Entry(cast_group, **entry_kw, state="disabled")
        self.draft_angle.insert(0, "3.0")
        self.draft_angle.grid(row=row, column=1, padx=5)

        tb.Label(cast_group, text="拔模方向:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.draft_dir = tb.Combobox(cast_group, values=["X", "Y", "Z"], width=5, font=self.default_font, state="disabled")
        self.draft_dir.set("Z")
        self.draft_dir.grid(row=row, column=3, padx=5)

        # 按钮
        btn_frame = tb.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = tb.Button(btn_frame, text="开始优化", command=self.start_optimization, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tb.Button(btn_frame, text="停止", command=self.stop_optimization, state=tk.DISABLED, width=10)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        tb.Button(btn_frame, text="重置参数", command=self.reset_params, width=10).pack(side=tk.LEFT, padx=5)

        # 进度区域
        progress_frame = tb.LabelFrame(parent, text="优化进度", padx=10, pady=10, font=self.title_font)
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.progress_bar = tb.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.status_text = tk.Text(progress_frame, height=10, wrap=tk.WORD, font=self.log_font)
        self.status_text.pack(fill=tk.BOTH, expand=True)

        status_line = tb.Frame(progress_frame)
        status_line.pack(fill=tk.X, pady=2)
        self.lbl_iter = tb.Label(status_line, text="迭代: 0/0", font=self.status_font)
        self.lbl_iter.pack(side=tk.LEFT, padx=5)
        self.lbl_comp = tb.Label(status_line, text="柔度: --", font=self.status_font)
        self.lbl_comp.pack(side=tk.LEFT, padx=5)
        self.lbl_vol = tb.Label(status_line, text="体积: --", font=self.status_font)
        self.lbl_vol.pack(side=tk.LEFT, padx=5)

    def toggle_cast(self):
        state = tk.NORMAL if self.enable_cast_var.get() else tk.DISABLED
        self.material_type.config(state=state)
        self.min_thick.config(state=state)
        self.draft_angle.config(state=state)
        self.draft_dir.config(state=state)

    def create_2d_view(self):
        self.fig_2d = Figure(figsize=(5,4), dpi=100)
        self.ax_2d = self.fig_2d.add_subplot(111)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.frame_2d)
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_3d_view(self):
        self.fig_3d = Figure(figsize=(5,4), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.frame_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_plot_view(self):
        self.fig_plot = Figure(figsize=(5,4), dpi=100)
        self.ax_plot = self.fig_plot.add_subplot(111)
        self.canvas_plot = FigureCanvasTkAgg(self.fig_plot, master=self.frame_plot)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.status_text.insert(tk.END, msg + "\n")
        self.status_text.see(tk.END)

    def get_params(self):
        # 解析载荷方向（已通过复选框和大小输入）
        return {
            'mesh': {
                'Lx': float(self.design_x.get()),
                'Ly': float(self.design_y.get()),
                'Lz': float(self.design_z.get()),
                'lc': float(self.lc.get()),
                'filename': 'mesh.msh'
            },
            'material': {
                'E0': float(self.E.get()) * 1e9,
                'Emin': 1e-9,
                'nu': float(self.nu.get()),
                'p': float(self.penal.get())
            },
            'load': {
                'enable_vert': self.load_vert_var.get(),
                'enable_lat': self.load_lat_var.get(),
                'enable_long': self.load_long_var.get(),
                'mag_vert': float(self.mag_vert.get()),
                'mag_lat': float(self.mag_lat.get()),
                'mag_long': float(self.mag_long.get())
            },
            'optimization': {
                'vol_frac': float(self.vol_frac.get()),
                'rmin': float(self.rmin.get()),
                'move': 0.2,
                'tol': 1e-2,
                'max_iter': int(self.max_iter.get())
            },
            'casting': {
                'enabled': self.enable_cast_var.get(),
                'material_type': self.material_type.get(),
                'draft_direction': [1,0,0] if self.draft_dir.get()=="X" else ([0,1,0] if self.draft_dir.get()=="Y" else [0,0,1]),
                'min_thickness': float(self.min_thick.get()) if self.enable_cast_var.get() else 2.0,
                'draft_angle': float(self.draft_angle.get()) if self.enable_cast_var.get() else 3.0
            }
        }

    def start_optimization(self):
        if self.running:
            messagebox.showwarning("提示", "优化正在进行中")
            return
        try:
            params = self.get_params()
            # 先生成网格（主线程）
            self.log("正在生成网格...")
            generate_cantilever_mesh(
                Lx=params['mesh']['Lx'],
                Ly=params['mesh']['Ly'],
                Lz=params['mesh']['Lz'],
                lc=params['mesh']['lc'],
                filename=params['mesh']['filename']
            )
            self.log("网格生成完成")

            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.log("="*50)
            self.log("开始拓扑优化")
            self.log(f"参数: {params}")
            self.log("="*50)

            self.thread = OptimizationThread(params, self.queue)
            self.thread.start()
        except Exception as e:
            messagebox.showerror("错误", f"参数错误: {str(e)}")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def stop_optimization(self):
        if self.running and self.thread:
            self.thread.stop_flag = True
            self.log("正在停止优化...")

    def reset_params(self):
        self.design_x.delete(0, tk.END); self.design_x.insert(0, "2.0")
        self.design_y.delete(0, tk.END); self.design_y.insert(0, "0.5")
        self.design_z.delete(0, tk.END); self.design_z.insert(0, "0.5")
        self.lc.delete(0, tk.END); self.lc.insert(0, "0.2")
        self.mesh_x.delete(0, tk.END); self.mesh_x.insert(0, "40")
        self.mesh_y.delete(0, tk.END); self.mesh_y.insert(0, "10")
        self.mesh_z.delete(0, tk.END); self.mesh_z.insert(0, "10")
        self.E.delete(0, tk.END); self.E.insert(0, "1.0")
        self.nu.delete(0, tk.END); self.nu.insert(0, "0.3")
        self.load_vert_var.set(True)
        self.load_lat_var.set(False)
        self.load_long_var.set(False)
        self.mag_vert.delete(0, tk.END); self.mag_vert.insert(0, "1.0")
        self.mag_lat.delete(0, tk.END); self.mag_lat.insert(0, "0.5")
        self.mag_long.delete(0, tk.END); self.mag_long.insert(0, "0.5")
        self.vol_frac.delete(0, tk.END); self.vol_frac.insert(0, "0.35")
        self.penal.delete(0, tk.END); self.penal.insert(0, "3.0")
        self.rmin.delete(0, tk.END); self.rmin.insert(0, "0.1")
        self.max_iter.delete(0, tk.END); self.max_iter.insert(0, "50")
        self.enable_cast_var.set(False)
        self.toggle_cast()
        self.material_type.set("steel")
        self.min_thick.delete(0, tk.END); self.min_thick.insert(0, "2.0")
        self.draft_angle.delete(0, tk.END); self.draft_angle.insert(0, "3.0")
        self.draft_dir.set("Z")
        self.log("参数已重置")

    def check_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == 'progress':
                    it, total, comp, vol, chg = msg[1]
                    self.progress_bar['value'] = (it / total) * 100
                    self.lbl_iter.config(text=f"迭代: {it}/{total}")
                    self.lbl_comp.config(text=f"柔度: {comp:.2e}")
                    self.lbl_vol.config(text=f"体积: {vol:.3f}")
                    if it % 5 == 0:
                        self.log(f"迭代 {it}: 柔度={comp:.2e}, 体积={vol:.3f}, 变化={chg:.4f}")
                elif msg[0] == 'log':
                    self.log(msg[1])
                elif msg[0] == 'done':
                    rho, history, domain = msg[1]
                    self.rho_array = rho.x.array
                    self.history = history
                    self.domain = domain
                    self.running = False
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.log("优化完成！")
                    self.update_plots()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def update_plots(self):
        if self.rho_array is None or self.domain is None:
            return

        # 从 domain 获取网格划分信息（简化：假设为规则网格）
        nx = int(self.mesh_x.get())
        ny = int(self.mesh_y.get())
        nz = int(self.mesh_z.get())
        expected_cells = nx * ny * nz
        if len(self.rho_array) != expected_cells:
            self.log(f"警告：密度数组大小 {len(self.rho_array)} 与网格 {expected_cells} 不匹配，无法显示切片")
            return

        rho_3d = self.rho_array.reshape((nz, ny, nx))

        # 2D 切片：取中间层
        slice_z = nz // 2
        self.ax_2d.clear()
        im = self.ax_2d.imshow(rho_3d[slice_z, :, :], cmap='viridis', aspect='auto', origin='lower')
        self.ax_2d.set_title(f"密度切片 Z={slice_z} (层)")
        plt.colorbar(im, ax=self.ax_2d, label='密度')
        self.canvas_2d.draw()

        # 3D 预览：只显示密度大于0.5的单元
        self.ax_3d.clear()
        X, Y, Z = np.meshgrid(range(nx), range(ny), range(nz), indexing='ij')
        mask = rho_3d > 0.5
        self.ax_3d.scatter(X[mask], Y[mask], Z[mask], c=rho_3d[mask], cmap='viridis', s=5, alpha=0.5)
        self.ax_3d.set_xlabel('X'); self.ax_3d.set_ylabel('Y'); self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title("3D 密度分布 (>0.5)")
        self.canvas_3d.draw()

        # 收敛曲线
        self.ax_plot.clear()
        if self.history['compliance']:
            self.ax_plot.plot(self.history['compliance'], 'b-')
            self.ax_plot.set_xlabel('迭代次数')
            self.ax_plot.set_ylabel('柔度')
            self.ax_plot.set_title('收敛历史')
            self.ax_plot.grid(True)
        self.canvas_plot.draw()

    def save_results(self):
        if self.rho_array is None:
            messagebox.showwarning("警告", "没有可保存的结果")
            return
        from tkinter import filedialog
        folder = filedialog.askdirectory()
        if folder:
            path = Path(folder)
            np.savez(path/"results.npz", rho=self.rho_array, history=self.history)
            self.log(f"结果保存至 {path}")

    def load_results(self):
        from tkinter import filedialog
        file = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
        if file:
            data = np.load(file, allow_pickle=True)
            self.log(f"加载结果: {file}")

    def show_about(self):
        messagebox.showinfo("关于", "拓扑优化系统 v3.0\n基于 FEniCSx + nlopt MMA\n界面 by ttkbootstrap")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = OptimizedGUI()
    app.run()