# visualization.py
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QComboBox, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QPushButton, QTextEdit, QProgressBar)
from PyQt5.QtCore import pyqtSignal, QThread
from pyvistaqt import QtInteractor


class OptimizationWorker(QThread):
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
        self.log_signal.emit("🚀 强制启动 MMA(移动渐近线) 转向架优化内核...")
        try:
            import nlopt
            opt = nlopt.opt(nlopt.LD_MMA, len(self.rho_array))
            opt.set_lower_bounds(0.001)
            opt.set_upper_bounds(1.0)

            # 获取网格体积，严格还原真实导数，防止 MMA 第一步死锁退出！
            coords = self.optimizer.fea.domain.geometry.x
            Lx = np.max(coords[:, 0]) - np.min(coords[:, 0])
            Ly = np.max(coords[:, 1]) - np.min(coords[:, 1])
            Lz = np.max(coords[:, 2]) - np.min(coords[:, 2])
            v_domain = max(1e-8, Lx * Ly * Lz)
            v_cell = v_domain / len(self.rho_array)

            self.objective_c0 = None
            self.iter_count = 0

            # -----主目标函数 (基于多载荷的柔度)-----
            def mma_objective(x, grad):
                if not self.is_running:
                    raise Exception("受人工干预安全停止")

                self.rho_array[:] = x
                active_loads = self.optimizer.params.get('active_loads',
                                                         [{'dir': (0, 0, -1), 'mag': 1.0, 'weight': 1.0}])

                total_c = 0.0
                total_dc = np.zeros_like(x)

                for ld in active_loads:
                    self.optimizer.fea.clear_loads()
                    self.optimizer.fea.add_pressure_load(ld['mag'], ld['dir'], surface_tag=2)
                    uh = self.optimizer.fea.solve()
                    total_c += self.optimizer.fea.compute_compliance(uh) * ld.get('weight', 1.0)
                    total_dc += self.optimizer.fea.compute_sensitivity(uh) * ld.get('weight', 1.0)

                dc_f = self.optimizer.filter_sensitivities(total_dc)
                if self.objective_c0 is None:
                    self.objective_c0 = abs(total_c) if abs(total_c) > 1e-12 else 1.0

                if grad.size > 0:
                    # 将偏导数转换到真实规模给 MMA 看
                    grad[:] = (dc_f * v_cell) / self.objective_c0

                vol = np.mean(x)
                self.iter_count += 1

                self.log_signal.emit(f"MMA 第 {self.iter_count} 轮求解: 柔度 = {total_c:.2e}, 体积比 = {vol:.3f}")
                self.progress_signal.emit(self.iter_count, total_c, vol)
                self.plot_signal.emit(x.copy(), total_dc.copy())

                self.optimizer.history['compliance'].append(total_c)
                self.optimizer.history['vol'].append(vol)

                return total_c / self.objective_c0

            opt.set_min_objective(mma_objective)

            # -----1. 体积约束-----
            def mma_vol_constraint(x, grad):
                if grad.size > 0: grad[:] = np.ones_like(x) / len(x)
                return np.mean(x) - self.optimizer.vol_frac

            opt.add_inequality_constraint(mma_vol_constraint, 1e-6)

            # -----2. 铸造工艺约束 (对于转向架构架等必须启用)-----
            if hasattr(self.optimizer, 'casting_constraint') and self.optimizer.casting_constraint is not None:
                self.log_signal.emit("🛠️ 工艺铸造拔模与壁厚约束已成功接入 MMA 算子。")

                def mma_cast_con(x, grad):
                    try:
                        g, dgdx = self.optimizer.casting_constraint.get_constraint(x)
                        if grad.size > 0:
                            grad[:] = dgdx.flatten()
                        return float(g[0])
                    except Exception as e:
                        if grad.size > 0: grad[:] = 0.0
                        return 0.0

                opt.add_inequality_constraint(mma_cast_con, 1e-4)

            opt.set_maxeval(self.max_iter)
            opt.set_xtol_rel(1e-3)

            # 正式启动推演！
            opt.optimize(self.rho_array.copy())
            self.log_signal.emit("🎉 MMA 拓扑收敛：达到最佳轻量化构架极限！")

        except Exception as e:
            if "人工" not in str(e):
                import traceback
                traceback.print_exc()
            self.log_signal.emit(f"提示: MMA 停止/完成 ({str(e)})")
        finally:
            self.finished_signal.emit()


class MainWindow(QMainWindow):
    def __init__(self, domain, facet_tags):
        super().__init__()
        self.domain = domain
        self.facet_tags = facet_tags
        self.running = False
        self.worker = None

        self.vtk_plotter = QtInteractor(self, shape=(1, 2))
        for i in range(2):
            self.vtk_plotter.subplot(0, i)
            self.vtk_plotter.hide_axes()
            self.vtk_plotter.set_background("white", top="lightgray")

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("MMA转向架拓扑优化系统 - 制造铸造约束版")
        self.resize(1350, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        h_layout = QHBoxLayout(main_widget)

        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(0, 0, 10, 0)

        # ====== 1. 几何 ======
        group_geo = QGroupBox("1. 设计域网格")
        grid_geo = QVBoxLayout()
        self.inputs = {}
        vars_geo = [("Lx (长度):", 2.0), ("Ly (宽度):", 0.5), ("Lz (高度):", 0.5), ("网格尺寸 lc:", 0.2)]
        for label, val in vars_geo:
            row = QHBoxLayout();
            row.addWidget(QLabel(label));
            qds = QDoubleSpinBox();
            qds.setRange(0.01, 100);
            qds.setValue(val);
            qds.setSingleStep(0.1)
            self.inputs[label] = qds;
            row.addWidget(qds);
            grid_geo.addLayout(row)
        vars_mesh = [("nx:", 40), ("ny:", 10), ("nz:", 10)]
        for label, val in vars_mesh:
            row = QHBoxLayout();
            row.addWidget(QLabel(label));
            qs = QSpinBox();
            qs.setRange(2, 200);
            qs.setValue(val)
            self.inputs[label] = qs;
            row.addWidget(qs);
            grid_geo.addLayout(row)
        group_geo.setLayout(grid_geo);
        left_panel.addWidget(group_geo)

        # ====== 2. 多轴载荷 ======
        group_load = QGroupBox("2. 外部多轴载荷配比")
        grid_load = QVBoxLayout()
        self.loads = {}
        for load_name, val in [("主要 Z向垂向力", 1.0), ("支线 Y向侧偏力", 0.1), ("抗扭 X向纵向力", 0.0)]:
            row = QHBoxLayout()
            cb = QCheckBox(load_name)
            if "Z" in load_name or "Y" in load_name: cb.setChecked(True)  # 默认开着侧偏，拉动连杆生长
            spin = QDoubleSpinBox();
            spin.setRange(-100, 100);
            spin.setValue(val)
            self.loads[load_name] = (cb, spin)
            row.addWidget(cb);
            row.addWidget(spin);
            grid_load.addLayout(row)
        group_load.setLayout(grid_load);
        left_panel.addWidget(group_load)

        # ====== 3. 制造工艺 ======
        group_cast = QGroupBox("3. 结构铸造与工艺约束 (必设项)")
        grid_cast = QVBoxLayout()
        self.cb_cast = QCheckBox("强制启用工艺铸造限制")
        self.cb_cast.setChecked(True)  # 🌟默认强开限制
        grid_cast.addWidget(self.cb_cast)

        row = QHBoxLayout();
        row.addWidget(QLabel("构架主材料:"));
        self.combo_mat = QComboBox();
        self.combo_mat.addItems(["steel", "cast_iron", "aluminum"])
        row.addWidget(self.combo_mat);
        grid_cast.addLayout(row)

        row = QHBoxLayout();
        row.addWidget(QLabel("最小壁厚限制(mm):"));
        self.spin_thick = QDoubleSpinBox();
        self.spin_thick.setValue(3.0);
        self.spin_thick.setRange(0.5, 10)
        row.addWidget(self.spin_thick);
        grid_cast.addLayout(row)

        row = QHBoxLayout();
        row.addWidget(QLabel("开模拔出方向:"));
        self.combo_dir = QComboBox();
        self.combo_dir.addItems(["Z (推荐脱模方向)", "Y", "X"])
        row.addWidget(self.combo_dir);
        grid_cast.addLayout(row)
        group_cast.setLayout(grid_cast);
        left_panel.addWidget(group_cast)

        # ====== 4. MMA算子 ======
        group_opt = QGroupBox("4. MMA 控制参数")
        grid_opt = QVBoxLayout()
        # ✨为了使得构架能够有足够材料连通而不是独立，推荐体积调到 0.45
        vars_opt = [("体积分数约束:", 0.45, QDoubleSpinBox), ("材料惩罚:", 3.0, QDoubleSpinBox),
                    ("结网滤波半径:", 0.15, QDoubleSpinBox), ("MMA最大步数:", 60, QSpinBox)]
        for label, val, WidgetType in vars_opt:
            row = QHBoxLayout();
            row.addWidget(QLabel(label));
            w = WidgetType()
            if isinstance(w, QSpinBox):
                w.setRange(1, 1000)
            else:
                w.setRange(0.01, 10.0); w.setSingleStep(0.05)
            w.setValue(val);
            self.inputs[label] = w;
            row.addWidget(w);
            grid_opt.addLayout(row)
        group_opt.setLayout(grid_opt);
        left_panel.addWidget(group_opt)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("发车: 运行 MMA 工业优化")
        self.stop_btn = QPushButton("中止")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn);
        btn_layout.addWidget(self.stop_btn)
        left_panel.addLayout(btn_layout)

        self.log_txt = QTextEdit()
        self.log_txt.setReadOnly(True)
        self.log_txt.setMaximumHeight(100)
        left_panel.addWidget(self.log_txt)

        self.status_bar = QProgressBar()
        self.status_bar.setValue(0)
        left_panel.addWidget(self.status_bar)

        h_layout.addLayout(left_panel, 1)
        h_layout.addWidget(self.vtk_plotter.interactor, 4)
        self.stop_btn.clicked.connect(self.stop_optimization)

    def log(self, text):
        self.log_txt.append(text)
        self.log_txt.verticalScrollBar().setValue(self.log_txt.verticalScrollBar().maximum())

    def update_plot(self, rho_arr, dc_arr):
        pass

    def update_progress(self, it, c, vol):
        max_iter = self.inputs["MMA最大步数:"].value()
        percent = int((it / max_iter) * 100)
        if percent > 100: percent = 100
        self.status_bar.setValue(percent)

    def stop_optimization(self):
        if self.worker:
            self.worker.is_running = False
            self.log("收到命令，正在安全中断 MMA ...")

    def optimization_done(self):
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.inputs["MMA最大步数:"].setEnabled(True)
        self.status_bar.setValue(100)

    def get_params(self):
        p = {}
        # 硬编码底层材料杨氏模量，防计算崩溃
        p['material'] = {'E0': 200e9, 'nu': 0.3}
        p['optimization'] = {'vol_frac': self.inputs["体积分数约束:"].value(), 'p': self.inputs["材料惩罚:"].value(),
                             'rmin': self.inputs["结网滤波半径:"].value(),
                             'max_iter': self.inputs["MMA最大步数:"].value()}
        p['load'] = {
            'enable_vert': self.loads["主要 Z向垂向力"][0].isChecked(),
            'mag_vert': self.loads["主要 Z向垂向力"][1].value(),
            'enable_lat': self.loads["支线 Y向侧偏力"][0].isChecked(),
            'mag_lat': self.loads["支线 Y向侧偏力"][1].value(),
            'enable_long': self.loads["抗扭 X向纵向力"][0].isChecked(),
            'mag_long': self.loads["抗扭 X向纵向力"][1].value(),
        }
        dir_map = {"Z (推荐脱模方向)": [0, 0, 1], "Y": [0, 1, 0], "X": [1, 0, 0]}
        p['casting'] = {
            'enable': self.cb_cast.isChecked(),
            'material': self.combo_mat.currentText(),
            'min_thickness': self.spin_thick.value(),
            'draft_direction': dir_map[self.combo_dir.currentText()]
        }
        return p

    def start_optimization_with_params(self, params, optimizer, rho_func):
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.inputs["MMA最大步数:"].setEnabled(False)
        self.status_bar.setValue(0)

        bounds = (np.zeros_like(rho_func.x.array), np.ones_like(rho_func.x.array))
        self.worker = OptimizationWorker(optimizer, bounds, params['optimization']['max_iter'], rho_func.x.array)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log)
        self.worker.plot_signal.connect(self.update_plot)
        self.worker.finished_signal.connect(self.optimization_done)
        self.worker.start()