import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 设置中文字体（根据系统已安装字体调整）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

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


class TopologyOptimizer3D:
    def __init__(self, params):
        self.params = params
        self.nx, self.ny = params['mesh_resolution']
        self.rho = np.full((self.ny, self.nx), params['volume_fraction'])
        self.history = {'compliance': [], 'volume': [], 'change': []}
        self.iterations = 0
        self.stop_flag = False

    def compute_compliance(self, rho):
        ny, nx = rho.shape
        p = self.params['penalization']
        K = np.zeros((ny, nx))
        for i in range(nx):
            for j in range(ny):
                E = 1e-9 + (1 - 1e-9) * rho[j, i] ** p
                K[j, i] = E
        avg_stiffness = np.mean(K)
        if self.params['bc_type'] == 'cantilever':
            compliance = 100.0 / (avg_stiffness + 1e-6)
        else:
            compliance = 90.0 / (avg_stiffness + 1e-6)
        return compliance

    def compute_sensitivities(self, rho, compliance):
        p = self.params['penalization']
        return -p * rho ** (p - 1) * compliance / (self.nx * self.ny)

    def apply_filter(self, rho):
        rho_filtered = rho.copy()
        ny, nx = rho.shape
        for i in range(ny):
            for j in range(nx):
                i_min = max(0, i-2); i_max = min(ny, i+3)
                j_min = max(0, j-2); j_max = min(nx, j+3)
                rho_filtered[i,j] = np.mean(rho[i_min:i_max, j_min:j_max])
        return rho_filtered

    def update_oc(self, rho, dc):
        move = self.params['move_limit']
        vol_frac = self.params['volume_fraction']
        l1, l2 = 0, 1e9
        for _ in range(50):
            lmid = (l1 + l2) / 2
            B = -dc / (lmid + 1e-10)
            B = np.maximum(0.1, B)
            rho_new = rho * np.sqrt(B)
            rho_new = np.clip(rho_new, 0.01, 0.99)
            rho_new = np.clip(rho_new, rho - move, rho + move)
            if np.mean(rho_new) > vol_frac:
                l1 = lmid
            else:
                l2 = lmid
            if l2 - l1 < 1e-6:
                break
        change = np.max(np.abs(rho_new - rho))
        return rho_new, change, np.mean(rho_new)

    def optimize(self, progress_queue):
        for it in range(self.params['max_iter']):
            if self.stop_flag:
                progress_queue.put(('log', "优化被用户停止"))
                break
            compliance = self.compute_compliance(self.rho)
            dc = self.compute_sensitivities(self.rho, compliance)
            rho_f = self.apply_filter(self.rho)
            rho_new, change, vol = self.update_oc(rho_f, dc)
            self.rho = rho_new
            self.history['compliance'].append(compliance)
            self.history['volume'].append(vol)
            self.history['change'].append(change)
            self.iterations = it + 1
            progress_queue.put(('progress', (it+1, self.params['max_iter'], compliance, vol, change)))
            if change < self.params['tol'] and it > 10:
                progress_queue.put(('log', f"收敛于迭代 {it+1}"))
                break
        progress_queue.put(('done', self))


# ---------- GUI类（字体美化，支持中文）----------
class OptimizedGUI:
    def __init__(self):
        self.root = tb.Window(themename="superhero")
        self.root.title("拓扑优化系统 - 专业版")
        self.root.geometry("1300x800")

        # 设置字体（仅对支持字体的控件有效）
        self.default_font = ('Helvetica', 10)
        self.title_font = ('Helvetica', 12, 'bold')
        self.status_font = ('Helvetica', 11, 'bold')
        self.log_font = ('Consolas', 9)

        self.optimizer = None
        self.thread = None
        self.running = False
        self.queue = queue.Queue()

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
        self.notebook.add(self.frame_2d, text="2D视图")
        self.create_2d_view()

        self.frame_3d = tb.Frame(self.notebook)
        self.notebook.add(self.frame_3d, text="3D视图")
        self.create_3d_view()

        self.frame_plot = tb.Frame(self.notebook)
        self.notebook.add(self.frame_plot, text="收敛曲线")
        self.create_plot_view()

    def create_control_panel(self, parent):
        # 优化参数框
        param_frame = tb.LabelFrame(parent, text="优化参数", padx=10, pady=10, font=self.title_font)
        param_frame.pack(fill=tk.X, pady=5)

        row = 0
        # 标签字体
        label_kw = {'font': self.default_font}
        entry_kw = {'font': self.default_font, 'width': 10}

        tb.Label(param_frame, text="设计域宽度 (m):", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.design_x = tb.Entry(param_frame, **entry_kw)
        self.design_x.insert(0, "2.0")
        self.design_x.grid(row=row, column=1, padx=5)

        tb.Label(param_frame, text="高度 (m):", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.design_y = tb.Entry(param_frame, **entry_kw)
        self.design_y.insert(0, "1.0")
        self.design_y.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(param_frame, text="网格 X:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.mesh_x = tb.Entry(param_frame, **entry_kw)
        self.mesh_x.insert(0, "60")
        self.mesh_x.grid(row=row, column=1, padx=5)

        tb.Label(param_frame, text="网格 Y:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.mesh_y = tb.Entry(param_frame, **entry_kw)
        self.mesh_y.insert(0, "30")
        self.mesh_y.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(param_frame, text="体积分数:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.vol_frac = tb.Entry(param_frame, **entry_kw)
        self.vol_frac.insert(0, "0.4")
        self.vol_frac.grid(row=row, column=1, padx=5)

        tb.Label(param_frame, text="惩罚因子:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.penal = tb.Entry(param_frame, **entry_kw)
        self.penal.insert(0, "3.0")
        self.penal.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(param_frame, text="滤波半径:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.filter_r = tb.Entry(param_frame, **entry_kw)
        self.filter_r.insert(0, "1.5")
        self.filter_r.grid(row=row, column=1, padx=5)

        tb.Label(param_frame, text="最大迭代:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.max_iter = tb.Entry(param_frame, **entry_kw)
        self.max_iter.insert(0, "50")
        self.max_iter.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(param_frame, text="收敛容差:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.tol = tb.Entry(param_frame, **entry_kw)
        self.tol.insert(0, "0.001")
        self.tol.grid(row=row, column=1, padx=5)

        tb.Label(param_frame, text="移动限制:", **label_kw).grid(row=row, column=2, sticky=tk.W, pady=2)
        self.move = tb.Entry(param_frame, **entry_kw)
        self.move.insert(0, "0.05")
        self.move.grid(row=row, column=3, padx=5)

        row += 1
        tb.Label(param_frame, text="边界条件:", **label_kw).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.bc_type = tb.Combobox(param_frame, values=["cantilever", "bridge", "michell"], 
                                    width=12, font=self.default_font)
        self.bc_type.set("cantilever")
        self.bc_type.grid(row=row, column=1, columnspan=3, sticky=tk.W, padx=5)

        # 按钮区域（移除font参数）
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
        return {
            'design_domain': [float(self.design_x.get()), float(self.design_y.get())],
            'mesh_resolution': [int(self.mesh_x.get()), int(self.mesh_y.get())],
            'volume_fraction': float(self.vol_frac.get()),
            'penalization': float(self.penal.get()),
            'filter_radius': float(self.filter_r.get()),
            'max_iter': int(self.max_iter.get()),
            'tol': float(self.tol.get()),
            'move_limit': float(self.move.get()),
            'bc_type': self.bc_type.get(),
        }

    def start_optimization(self):
        if self.running:
            messagebox.showwarning("提示", "优化正在进行中")
            return
        try:
            params = self.get_params()
            self.optimizer = TopologyOptimizer3D(params)
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.log("="*50)
            self.log("开始拓扑优化")
            self.log(f"参数: {params}")
            self.log("="*50)

            self.thread = threading.Thread(target=self.optimizer.optimize, args=(self.queue,))
            self.thread.daemon = True
            self.thread.start()
        except Exception as e:
            messagebox.showerror("错误", f"参数错误: {str(e)}")

    def stop_optimization(self):
        if self.running and self.optimizer:
            self.optimizer.stop_flag = True
            self.log("正在停止优化...")

    def reset_params(self):
        self.design_x.delete(0, tk.END); self.design_x.insert(0, "2.0")
        self.design_y.delete(0, tk.END); self.design_y.insert(0, "1.0")
        self.mesh_x.delete(0, tk.END); self.mesh_x.insert(0, "60")
        self.mesh_y.delete(0, tk.END); self.mesh_y.insert(0, "30")
        self.vol_frac.delete(0, tk.END); self.vol_frac.insert(0, "0.4")
        self.penal.delete(0, tk.END); self.penal.insert(0, "3.0")
        self.filter_r.delete(0, tk.END); self.filter_r.insert(0, "1.5")
        self.max_iter.delete(0, tk.END); self.max_iter.insert(0, "50")
        self.tol.delete(0, tk.END); self.tol.insert(0, "0.001")
        self.move.delete(0, tk.END); self.move.insert(0, "0.05")
        self.bc_type.set("cantilever")
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
                    self.update_plots()
                elif msg[0] == 'log':
                    self.log(msg[1])
                elif msg[0] == 'done':
                    self.optimizer = msg[1]
                    self.running = False
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.log("优化完成！")
                    self.update_plots()
                    self.show_3d()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def update_plots(self):
        if not self.optimizer:
            return
        self.ax_2d.clear()
        self.ax_2d.imshow(self.optimizer.rho, cmap='viridis', aspect='auto', origin='lower')
        self.ax_2d.set_title("2D密度分布")
        self.canvas_2d.draw()

        self.ax_3d.clear()
        ny, nx = self.optimizer.rho.shape
        X, Y = np.meshgrid(range(nx), range(ny))
        self.ax_3d.plot_surface(X, Y, self.optimizer.rho, cmap='viridis', linewidth=0, antialiased=True)
        self.ax_3d.set_xlabel('X'); self.ax_3d.set_ylabel('Y'); self.ax_3d.set_zlabel('密度')
        self.ax_3d.set_title("3D密度分布")
        self.canvas_3d.draw()

        self.ax_plot.clear()
        if self.optimizer.history['compliance']:
            self.ax_plot.plot(self.optimizer.history['compliance'], 'b-')
            self.ax_plot.set_xlabel('迭代次数')
            self.ax_plot.set_ylabel('柔度')
            self.ax_plot.set_title('收敛历史')
            self.ax_plot.grid(True)
        self.canvas_plot.draw()

    def show_3d(self):
        self.notebook.select(self.frame_3d)

    def save_results(self):
        if not self.optimizer:
            messagebox.showwarning("警告", "没有可保存的结果")
            return
        from tkinter import filedialog
        folder = filedialog.askdirectory()
        if folder:
            path = Path(folder)
            np.savez(path/"results.npz", rho=self.optimizer.rho, history=self.optimizer.history)
            self.log(f"结果保存至 {path}")

    def load_results(self):
        from tkinter import filedialog
        file = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
        if file:
            data = np.load(file, allow_pickle=True)
            self.log(f"加载结果: {file}")

    def show_about(self):
        messagebox.showinfo("关于", "拓扑优化系统 v2.0\n支持交互式3D可视化")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = OptimizedGUI()
    app.run()