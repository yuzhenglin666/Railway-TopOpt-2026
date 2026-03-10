# visualization.py
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QComboBox, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QPushButton, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QThread

import pyvista as pv
from pyvistaqt import QtInteractor


class OptimizationWorker(QThread):
    # ✅ 信号升级：额外传出灵敏度 dc 作为应变能/应力的数据
    progress_signal = pyqtSignal(int, float, float)
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(np.ndarray, np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, optimizer, bounds, max_iter, rho_array):
        super().__init__()
        self.optimizer = optimizer
        self.bounds = bounds
        self.max_iter = max_iter
        self.rho_array = rho_array
        self.is_running = True

    def run(self):
        import nlopt
        opt = nlopt.opt(nlopt.LD_MMA, len(self.rho_array))
        opt.set_lower_bounds(self.bounds[0])
        opt.set_upper_bounds(self.bounds[1])

        # 回调函数
        def objective(x, grad):
            if not self.is_running:
                raise Exception("被用户手动停止")
            self.rho_array[:] = x

            uh = self.optimizer.fea.solve()
            c = self.optimizer.fea.compute_compliance(uh)

            if hasattr(self.optimizer.fea, 'compute_sensitivity'):
                dc = self.optimizer.fea.compute_sensitivity(uh)
            elif hasattr(self.optimizer.fea, 'compute_sensitivities'):
                dc = self.optimizer.fea.compute_sensitivities(uh)
            else:
                raise Exception("无法计算梯度")

            dc_f = self.optimizer.filter_sensitivities(dc)

            if grad.size > 0:
                grad[:] = dc_f

            vol = np.mean(x)
            it = len(self.optimizer.history['compliance'])
            self.log_signal.emit(f"迭代 {it}: 柔度 = {c:.2e}, 体积占比 = {vol:.3f}")
            self.progress_signal.emit(it, c, vol)

            # ✅ 新增：把 dc（原始未滤波的应变能分布）传出去画应力图
            self.plot_signal.emit(x.copy(), np.array(dc).copy())

            self.optimizer.history['compliance'].append(c)
            self.optimizer.history['vol'].append(vol)
            return c

        opt.set_min_objective(objective)

        def vol_constraint(x, grad):
            vol = np.mean(x)
            if grad.size > 0:
                grad[:] = np.ones_like(x) / len(x)
            return vol - self.optimizer.vol_frac

        opt.add_inequality_constraint(vol_constraint, 1e-4)

        if hasattr(self.optimizer, 'casting_constraint') and self.optimizer.casting_constraint:
            def cast_con(x, grad):
                g, dgdx = self.optimizer.casting_constraint.get_constraint(x)
                if grad.size > 0:
                    grad[:] = dgdx[0]
                return g[0]

            opt.add_inequality_constraint(cast_con, 1e-4)

        opt.set_maxeval(self.max_iter)

        try:
            self.log_signal.emit("MMA 优化算法启动...")
            opt.optimize(self.rho_array)
            self.log_signal.emit("优化顺利完成或达到最大迭代次数！")
        except Exception as e:
            self.log_signal.emit(f"优化终止/提示: {str(e)}")
        finally:
            self.finished_signal.emit()


class MainWindow(QMainWindow):
    def __init__(self, domain, facet_tags):
        super().__init__()
        self.domain = domain
        self.facet_tags = facet_tags
        self.running = False
        self.worker = None

        # ✅ 将画布划分为 1行 2列 (shape=(1, 2))
        self.vtk_plotter = QtInteractor(self, shape=(1, 2))

        # 左右两侧底色统一初始化
        for i in range(2):
            self.vtk_plotter.subplot(0, i)
            self.vtk_plotter.hide_axes()
            self.vtk_plotter.set_background("white", top="lightgray")

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("拓扑优化双屏系统 - 实体与应变能流图")
        # ✅ 加宽了窗口保证双屏显示不拥挤
        self.resize(1300, 750)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        h_layout = QHBoxLayout(main_widget)

        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(0, 0, 10, 0)

        group_geo = QGroupBox("几何与网格")
        grid_geo = QVBoxLayout()
        self.inputs = {}
        vars_geo = [("Lx (m):", 2.0), ("Ly (m):", 0.5), ("Lz (m):", 0.5), ("网格尺寸 lc:", 0.2)]
        for label, val in vars_geo:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            qds = QDoubleSpinBox()
            qds.setRange(0.01, 100);
            qds.setValue(val);
            qds.setSingleStep(0.1)
            self.inputs[label] = qds
            row.addWidget(qds)
            grid_geo.addLayout(row)

        vars_mesh = [("nx:", 40), ("ny:", 10), ("nz:", 10)]
        for label, val in vars_mesh:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            qs = QSpinBox()
            qs.setRange(2, 200);
            qs.setValue(val)
            self.inputs[label] = qs
            row.addWidget(qs)
            grid_geo.addLayout(row)
        group_geo.setLayout(grid_geo)
        left_panel.addWidget(group_geo)

        group_mat = QGroupBox("材料")
        grid_mat = QVBoxLayout()
        vars_mat = [("杨氏模量 (GPa):", 1.0), ("泊松比:", 0.3)]
        for label, val in vars_mat:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            qds = QDoubleSpinBox()
            qds.setRange(0.01, 300);
            qds.setValue(val);
            qds.setDecimals(3)
            self.inputs[label] = qds
            row.addWidget(qds)
            grid_mat.addLayout(row)
        group_mat.setLayout(grid_mat)
        left_panel.addWidget(group_mat)

        group_load = QGroupBox("载荷组合")
        grid_load = QVBoxLayout()
        self.loads = {}
        for load_name, val in [("垂向力 (Z)", 1.0), ("横向力 (Y)", 0.5), ("纵向力 (X)", 0.5)]:
            row = QHBoxLayout()
            cb = QCheckBox(load_name)
            if "Z" in load_name: cb.setChecked(True)
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 100);
            spin.setValue(val)
            self.loads[load_name] = (cb, spin)
            row.addWidget(cb)
            row.addWidget(spin)
            grid_load.addLayout(row)
        group_load.setLayout(grid_load)
        left_panel.addWidget(group_load)

        group_opt = QGroupBox("优化参数")
        grid_opt = QVBoxLayout()
        vars_opt = [("体积分数:", 0.35, QDoubleSpinBox), ("惩罚因子 p:", 3.0, QDoubleSpinBox),
                    ("滤波半径:", 0.1, QDoubleSpinBox), ("最大迭代:", 50, QSpinBox)]
        for label, val, WidgetType in vars_opt:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            w = WidgetType()
            if isinstance(w, QSpinBox):
                w.setRange(1, 1000)
            else:
                w.setRange(0.01, 10.0); w.setSingleStep(0.05)
            w.setValue(val)
            self.inputs[label] = w
            row.addWidget(w)
            grid_opt.addLayout(row)
        group_opt.setLayout(grid_opt)
        left_panel.addWidget(group_opt)

        group_cast = QGroupBox("铸造约束")
        grid_cast = QVBoxLayout()
        self.cb_cast = QCheckBox("启用")
        grid_cast.addWidget(self.cb_cast)

        row = QHBoxLayout();
        row.addWidget(QLabel("材料类型:"))
        self.combo_mat = QComboBox();
        self.combo_mat.addItems(["steel", "cast_iron", "aluminum"])
        row.addWidget(self.combo_mat);
        grid_cast.addLayout(row)

        row = QHBoxLayout();
        row.addWidget(QLabel("最小壁厚 (mm):"))
        self.spin_thick = QDoubleSpinBox();
        self.spin_thick.setValue(2.0);
        self.spin_thick.setRange(0.5, 10)
        row.addWidget(self.spin_thick);
        grid_cast.addLayout(row)

        row = QHBoxLayout();
        row.addWidget(QLabel("拔模斜度 (°):"))
        self.spin_ang = QDoubleSpinBox();
        self.spin_ang.setValue(3.0);
        self.spin_ang.setRange(0, 45)
        row.addWidget(self.spin_ang);
        grid_cast.addLayout(row)

        row = QHBoxLayout();
        row.addWidget(QLabel("拔模方向:"))
        self.combo_dir = QComboBox();
        self.combo_dir.addItems(["Z", "Y", "X"])
        row.addWidget(self.combo_dir);
        grid_cast.addLayout(row)
        group_cast.setLayout(grid_cast)
        left_panel.addWidget(group_cast)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始优化")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.reset_btn = QPushButton("重置参数")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.reset_btn)
        left_panel.addLayout(btn_layout)

        self.log_txt = QTextEdit()
        self.log_txt.setReadOnly(True)
        self.log_txt.setMaximumHeight(100)
        left_panel.addWidget(self.log_txt)

        self.status_bar = QProgressBar()
        self.status_bar.setValue(0)
        left_panel.addWidget(self.status_bar)

        h_layout.addLayout(left_panel, 1)

        # ✅ 加大右侧画板比重
        h_layout.addWidget(self.vtk_plotter.interactor, 4)

        self.stop_btn.clicked.connect(self.stop_optimization)

    def log(self, text):
        self.log_txt.append(text)
        self.log_txt.verticalScrollBar().setValue(self.log_txt.verticalScrollBar().maximum())

    def update_plot(self, rho_arr, dc_arr):
        # --- 步骤 1: 构建拓扑基网格 ---
        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, 0)
        num_cells = self.domain.topology.index_map(tdim).size_local

        cell_Nodes_1d = self.domain.topology.connectivity(tdim, 0).array
        cell_type_str = str(self.domain.topology.cell_type)

        if "tetrahedron" in cell_type_str:
            nodes_per_cell = 4
            vtk_type = pv.CellType.TETRA
        elif "hexahedron" in cell_type_str:
            nodes_per_cell = 8
            vtk_type = pv.CellType.HEXAHEDRON
        else:
            raise NotImplementedError(f"不支持的网格类型: {cell_type_str}")

        cell_nodes_2d = cell_Nodes_1d.reshape((num_cells, nodes_per_cell))
        padding = np.full((num_cells, 1), nodes_per_cell, dtype=np.int64)
        cells_vtk = np.hstack([padding, cell_nodes_2d]).flatten()
        celltypes = np.full(num_cells, vtk_type, dtype=np.uint8)
        coords = self.domain.geometry.x

        mesh_vtk = pv.UnstructuredGrid(cells_vtk, celltypes, coords)

        # --- 步骤 2: 绑定两种数据 ---
        mesh_vtk.cell_data["density"] = rho_arr.astype(np.float64)

        # 处理作为“应力云图”代理的应变能 (绝对值，切掉极值点防止颜色失效)
        stress_proxy = np.abs(np.array(dc_arr, dtype=np.float64))
        p99 = np.percentile(stress_proxy, 99)
        if p99 > 0:
            stress_proxy = np.clip(stress_proxy, a_min=0.0, a_max=p99)
        mesh_vtk.cell_data["Strain Energy"] = stress_proxy

        # --- 步骤 3: 阈值处理，准备左侧的切割体 ---
        threshold_val = 0.4
        mesh_cut = mesh_vtk.threshold(value=threshold_val, scalars="density", invert=False)

        # ========================================================
        # ✅ 左侧视角 (Subplot 0)：挖空的 3D 拓扑骨架
        # ========================================================
        self.vtk_plotter.subplot(0, 0)
        self.vtk_plotter.clear_actors()  # 清除模型但保留相机
        if mesh_cut.n_cells > 0:
            self.vtk_plotter.add_mesh(mesh_cut, color="#0066cc", opacity=1.0,
                                      show_edges=True, edge_color="#000000", smooth_shading=True)

            # 绘制透明边界壳，方便观察它在多大空间内
            shell = mesh_vtk.threshold(value=0.0, scalars="density", invert=False)
            self.vtk_plotter.add_mesh(shell, color="lightgray", style="wireframe", opacity=0.08)

        self.vtk_plotter.add_text("1. 3D Topology (Density > 0.4)", font_size=12, name="title_left")

        # ========================================================
        # ✅ 右侧视角 (Subplot 1)：原长方体的应力/应变能流传递云图
        # ========================================================
        self.vtk_plotter.subplot(0, 1)
        self.vtk_plotter.clear_actors()

        # 用 "jet" 配色 (经典有限元彩虹色) 渲染整个原始长方体的应力/阻力路径
        self.vtk_plotter.add_mesh(
            mesh_vtk,
            scalars="Strain Energy",
            cmap="jet",  # 从蓝到红过渡的经典应力云图配色
            show_edges=True,
            edge_color="darkgray",
            line_width=0.2,
            scalar_bar_args={"title": "Strain Energy\n(Stress Proxy)"}  # 颜色刻度条
        )
        self.vtk_plotter.add_text("2. Load Path / Stress Cloud", font_size=12, name="title_right")

        # --- 步骤 4: 只在开头锁定视角，且将左右屏幕连动 ---
        if not hasattr(self, "_camera_set"):
            self.vtk_plotter.link_views()  # 神奇开关：鼠标移动左边，右边画面也会同步跟着动！
            self.vtk_plotter.subplot(0, 0)
            self.vtk_plotter.camera_position = 'iso'
            self.vtk_plotter.view_isometric()
            self._camera_set = True

        self.vtk_plotter.update()

    def update_progress(self, it, c, vol):
        max_iter = self.inputs["最大迭代:"].value()
        percent = int(it / max_iter * 100)
        self.status_bar.setValue(percent if percent <= 100 else 100)

    def stop_optimization(self):
        if self.worker:
            self.worker.is_running = False
            self.log("收到停止指令，正在安全退出...")

    def optimization_done(self):
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.inputs["最大迭代:"].setEnabled(True)
        self.status_bar.setValue(100)
        self.log("执行进程结束！")

    def get_params(self):
        p = {}
        p['material'] = {
            'E0': self.inputs["杨氏模量 (GPa):"].value(),
            'nu': self.inputs["泊松比:"].value()
        }
        p['optimization'] = {
            'vol_frac': self.inputs["体积分数:"].value(),
            'p': self.inputs["惩罚因子 p:"].value(),
            'rmin': self.inputs["滤波半径:"].value(),
            'max_iter': self.inputs["最大迭代:"].value()
        }
        p['load'] = {
            'enable_vert': self.loads["垂向力 (Z)"][0].isChecked(), 'mag_vert': self.loads["垂向力 (Z)"][1].value(),
            'enable_lat': self.loads["横向力 (Y)"][0].isChecked(), 'mag_lat': self.loads["横向力 (Y)"][1].value(),
            'enable_long': self.loads["纵向力 (X)"][0].isChecked(), 'mag_long': self.loads["纵向力 (X)"][1].value(),
        }
        dir_map = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
        p['casting'] = {
            'enable': self.cb_cast.isChecked(), 'material': self.combo_mat.currentText(),
            'min_thickness': self.spin_thick.value(), 'draft_angle': self.spin_ang.value(),
            'draft_direction': dir_map[self.combo_dir.currentText()]
        }
        return p

    def start_optimization_with_params(self, params, optimizer, rho_func):
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.inputs["最大迭代:"].setEnabled(False)
        self.status_bar.setValue(0)

        bounds = (np.zeros_like(rho_func.x.array), np.ones_like(rho_func.x.array))

        self.worker = OptimizationWorker(optimizer, bounds, params['optimization']['max_iter'], rho_func.x.array)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log)
        self.worker.plot_signal.connect(self.update_plot)
        self.worker.finished_signal.connect(self.optimization_done)

        self.worker.start()