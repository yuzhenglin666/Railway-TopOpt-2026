# visualization.py
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QComboBox, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QPushButton, QTextEdit, QProgressBar,
                             QTreeWidget, QTreeWidgetItem, QSplitter)
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from pyvistaqt import QtInteractor
import pyvista as pv
from dolfinx import plot


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

            # 评估体积
            coords = self.optimizer.fea.domain.geometry.x
            Lx = np.max(coords[:, 0]) - np.min(coords[:, 0])
            Ly = np.max(coords[:, 1]) - np.min(coords[:, 1])
            Lz = np.max(coords[:, 2]) - np.min(coords[:, 2])
            v_domain = max(1e-8, Lx * Ly * Lz)
            v_cell = v_domain / len(self.rho_array)

            self.objective_c0 = None
            self.iter_count = 0

            def mma_objective(x, grad):
                if not self.is_running:
                    raise Exception("受人工干预安全停止")

                self.rho_array[:] = x

                # 🔥 终极防空指针保护：不管字典嵌套在第一层还是第二层，或为空，必定安全取值！
                raw_params = getattr(self.optimizer, 'params', {})
                if 'load' in raw_params and 'active_loads' in raw_params['load']:
                    active_loads = raw_params['load']['active_loads']
                else:
                    active_loads = raw_params.get('active_loads', [{'dir': (0, 0, -1), 'mag': 1.0, 'weight': 1.0}])

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
                    grad[:] = (dc_f * v_cell) / self.objective_c0

                vol = np.mean(x)
                self.iter_count += 1

                self.log_signal.emit(f"🔄 第 {self.iter_count} 轮: 合计响应柔度 = {total_c:.2e}, 质量比 = {vol:.3f}")
                self.progress_signal.emit(self.iter_count, total_c, vol)

                # 此处发出渲染信号唤醒 update_plot
                self.plot_signal.emit(x.copy(), total_dc.copy())

                self.optimizer.history['compliance'].append(total_c)
                self.optimizer.history['vol'].append(vol)

                return total_c / self.objective_c0

            opt.set_min_objective(mma_objective)

            def mma_vol_constraint(x, grad):
                if grad.size > 0: grad[:] = np.ones_like(x) / len(x)
                return np.mean(x) - self.optimizer.vol_frac

            opt.add_inequality_constraint(mma_vol_constraint, 1e-6)

            if hasattr(self.optimizer, 'casting_constraint') and self.optimizer.casting_constraint is not None:
                def mma_cast_con(x, grad):
                    try:
                        g, dgdx = self.optimizer.casting_constraint.get_constraint(x)
                        if grad.size > 0:
                            grad[:] = dgdx.flatten()
                        return float(g[0])
                    except Exception:
                        if grad.size > 0: grad[:] = 0.0
                        return 0.0

                opt.add_inequality_constraint(mma_cast_con, 1e-4)

            opt.set_maxeval(self.max_iter)
            opt.set_xtol_rel(1e-3)

            opt.optimize(self.rho_array.copy())
            self.log_signal.emit("🎉 MMA 计算平稳收敛：已达到最优轻量化极限！")

        except Exception as e:
            if "人工" not in str(e):
                import traceback
                traceback.print_exc()
            self.log_signal.emit(f"提示: MMA 已停止推演 ({str(e)})")
        finally:
            self.finished_signal.emit()


class MainWindow(QMainWindow):
    def __init__(self, domain, facet_tags):
        super().__init__()
        self.domain = domain
        self.facet_tags = facet_tags
        self.running = False
        self.worker = None

        self.crh380a_db = [
            {"name": "LC-01 超常垂向 (默认)", "Z": 313600, "Y": 0, "X": 0, "Twist": 0, "note": "抗压安全基础验证"},
            {"name": "LC-02 超常横向", "Z": 0, "Y": 85750, "X": 0, "Twist": 0, "note": "模拟离心侧摆受力"},
            {"name": "LC-03 超常纵向", "Z": 0, "Y": 0, "X": 213591, "Twist": 0, "note": "制动冲击综合考察"},
            {"name": "LC-04 超常垂+横", "Z": 313600, "Y": 85750, "X": 0, "Twist": 0, "note": "过弯道综合极限"},
            {"name": "LC-05 超常垂+纵", "Z": 313600, "Y": 0, "X": 213591, "Twist": 0, "note": "减速大避障测试"},
            {"name": "LC-06 超常垂+横+纵", "Z": 313600, "Y": 85750, "X": 213591, "Twist": 0, "note": "三向力全开测试"},
            {"name": "LC-07 轨道扭曲", "Z": 0, "Y": 0, "X": 0, "Twist": 25, "note": "评估非对称结构刚度"},
            {"name": "LC-08 超常垂+扭曲", "Z": 313600, "Y": 0, "X": 0, "Twist": 25, "note": "重载结合轨道失准失真度"},
            {"name": "LC-09 超常垂+横+扭曲", "Z": 313600, "Y": 85750, "X": 0, "Twist": 25,
             "note": "侧风与变形叠加影响"},
            {"name": "LC-10 超常垂+纵+扭曲", "Z": 313600, "Y": 0, "X": 213591, "Twist": 25, "note": "减减速变轨大挑战"},
            {"name": "LC-11 超常全组合", "Z": 313600, "Y": 85750, "X": 213591, "Twist": 25, "note": "史诗级最严苛试验"},
            {"name": "LC-12 运营垂向", "Z": 188160, "Y": 0, "X": 0, "Twist": 0, "note": "日复一日正常载客基础应力"},
            {"name": "LC-13 运营垂+横", "Z": 188160, "Y": 21038, "X": 0, "Twist": 0, "note": "常规过曲线行驶状态"},
            {"name": "LC-14 运营垂+横+纵", "Z": 188160, "Y": 21038, "X": 213591, "Twist": 0, "note": "加粗度加速运行"},
            {"name": "LC-15 运营全组合", "Z": 188160, "Y": 21038, "X": 213591, "Twist": 25,
             "note": "高频日常疲劳热点区"},
        ]

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("MMA转向架拓扑优化系统 - CRH380A标准库工作站")
        self.resize(1600, 950)

        # 🎨 UI基础：强力放大输入控件尺寸
        self.setStyleSheet("""
            QGroupBox { font-size: 16px; font-weight: bold; border: 1px solid #CCC; border-radius: 6px; margin-top: 15px; background: white; }
            QLabel { font-size: 15px; }
            QDoubleSpinBox, QSpinBox, QComboBox { font-size: 15px; height: 35px; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_splitter = QSplitter(Qt.Horizontal)

        # ==========================================
        # 1. 左侧：霸气宽大的树状菜单与物理说明书
        # ==========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["变量 / 工况类别", "配置参数确认"])

        self.tree.setColumnWidth(0, 260)
        self.tree.setAlternatingRowColors(True)
        # 💥 关键改动1：将每一行的高度强制拉高到 45px，表头拉升至42px，使得整体舒展撑满
        self.tree.setStyleSheet("""
            QTreeWidget { font-size: 15px; background-color: #FAFAFA; border: 1px solid #CCC; }
            QHeaderView::section { background-color: #DDE4EE; font-weight: bold; padding: 5px 8px; height: 42px; font-size: 15px; }
            QTreeWidget::item { height: 45px; border-bottom: 1px solid #EEE; }
            QTreeWidget::item:hover { background-color: #E6F3FF; }
        """)

        self.inputs = {}
        # A) 几何
        geo_cat = QTreeWidgetItem(self.tree, ["设计几何体与空间", ""])
        for lbl, v, mx, step in [("Lx (长度):", 2.0, 100, 0.1), ("Ly (宽度):", 0.5, 100, 0.1),
                                 ("Lz (高度):", 0.5, 100, 0.1), ("网格单元 tc:", 0.2, 10, 0.05)]:
            it = QTreeWidgetItem(geo_cat);
            it.setText(0, lbl)
            s = QDoubleSpinBox();
            s.setRange(0.01, mx);
            s.setValue(v);
            s.setSingleStep(step)
            self.tree.setItemWidget(it, 1, s);
            self.inputs[lbl] = s
        for lbl, v in [("nx 单元数:", 40), ("ny 单元数:", 10), ("nz 单元数:", 10)]:
            it = QTreeWidgetItem(geo_cat);
            it.setText(0, lbl)
            s = QSpinBox();
            s.setRange(2, 500);
            s.setValue(v)
            self.tree.setItemWidget(it, 1, s);
            self.inputs[lbl] = s

        # B) 工况库
        load_cat = QTreeWidgetItem(self.tree, ["CRH380A 多轴工况载荷库", "(系统模板/可微调)"])
        it_combo = QTreeWidgetItem(load_cat);
        it_combo.setText(0, "预设工况选择:")
        self.combo_load = QComboBox();
        self.combo_load.addItems([i["name"] for i in self.crh380a_db])
        self.tree.setItemWidget(it_combo, 1, self.combo_load)

        self.spin_z, self.spin_y, self.spin_x, self.spin_twist = [QDoubleSpinBox() for _ in range(4)]
        for w in [self.spin_z, self.spin_y, self.spin_x]: w.setRange(-9e8, 9e8); w.setDecimals(1); w.setSingleStep(1000)
        self.spin_twist.setRange(-1000, 1000)
        for lbl, w in [("Z垂向载荷 (N):", self.spin_z), ("Y横向载荷 (N):", self.spin_y),
                       ("X纵向载荷 (N):", self.spin_x), ("轨道扭曲位移 (mm):", self.spin_twist)]:
            it = QTreeWidgetItem(load_cat);
            it.setText(0, lbl);
            self.tree.setItemWidget(it, 1, w)

        # C) 工艺约束
        cast_cat = QTreeWidgetItem(self.tree, ["真实工程制造工艺约束", ""])
        it1 = QTreeWidgetItem(cast_cat);
        self.cb_cast = QCheckBox("强制开启脱模制约");
        self.cb_cast.setChecked(True);
        self.tree.setItemWidget(it1, 0, self.cb_cast)
        self.combo_mat = QComboBox();
        self.combo_mat.addItems(["steel", "cast_iron", "aluminum"])
        it2 = QTreeWidgetItem(cast_cat);
        it2.setText(0, "构架主材料:");
        self.tree.setItemWidget(it2, 1, self.combo_mat)
        self.spin_thick = QDoubleSpinBox();
        self.spin_thick.setRange(0.5, 15);
        self.spin_thick.setValue(3.0)
        it3 = QTreeWidgetItem(cast_cat);
        it3.setText(0, "极限截面壁厚(mm):");
        self.tree.setItemWidget(it3, 1, self.spin_thick)
        self.combo_dir = QComboBox();
        self.combo_dir.addItems(["Z 向 (车底脱模)", "Y 向", "X 向"])
        it4 = QTreeWidgetItem(cast_cat);
        it4.setText(0, "模具安全拔模向:");
        self.tree.setItemWidget(it4, 1, self.combo_dir)

        # D) 算法
        mma_cat = QTreeWidgetItem(self.tree, ["MMA物理算法收敛引擎", ""])
        for lbl, val, wtype in [("体积分数容限率:", 0.35, QDoubleSpinBox), ("伪度惩罚阶次:", 3.0, QDoubleSpinBox),
                                ("网状结构过滤半径:", 0.15, QDoubleSpinBox), ("MMA推演技最大代:", 50, QSpinBox)]:
            it = QTreeWidgetItem(mma_cat);
            it.setText(0, lbl)
            w = wtype();
            w.setValue(val);
            self.inputs[lbl] = w
            if isinstance(w, QSpinBox):
                w.setRange(1, 1000)
            else:
                w.setRange(0.01, 10.0); w.setSingleStep(0.05)
            self.tree.setItemWidget(it, 1, w)

        self.tree.expandAll()

        # 💥 关键改动2：这里去除了限制伸缩比(stretch factors)
        # 这样 TreeWidget 会凭借上面设定的巨大行高自动舒适地向下延展
        left_layout.addWidget(self.tree)

        # 🌟 日志面板（工况说明区）：将其垫厚垫大，托住空白区域
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(130)  # 调大下限，有效垫起上方内容
        self.info_box.setMaximumHeight(160)
        self.info_box.setStyleSheet(
            "background-color:#F5F7FA; border:1px solid #CCC; border-radius:4px; font-size:15px; padding:10px;")

        left_layout.addWidget(self.info_box)

        main_splitter.addWidget(left_widget)

        # ==========================================
        # 2. 右侧工作区：三维图形+底置控制台 (无改动区)
        # ==========================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)

        # 双屏图形阵列
        self.model_splitter = QSplitter(Qt.Horizontal)
        self.plotter_left = QtInteractor(self)
        self.plotter_left.set_background("white", top="lightgray")
        self.model_splitter.addWidget(self.plotter_left.interactor)

        self.plotter_right = QtInteractor(self)
        self.plotter_right.set_background("white", top="lightgray")
        self.model_splitter.addWidget(self.plotter_right.interactor)
        right_layout.addWidget(self.model_splitter, 8)

        # 右侧底下：巨大粗壮的仪表盘控制台
        bottom_console = QGroupBox("运算调度与模型流日志区室")
        console_layout = QVBoxLayout(bottom_console)

        btn_layout = QHBoxLayout()
        # 🌟 强化了大按钮设计
        self.start_btn = QPushButton("▶️  以此工况推演：启动架构生成")
        self.start_btn.setMinimumHeight(48)
        self.start_btn.setStyleSheet("""
            QPushButton { background-color: #0066CC; color: white; border-radius: 6px; font-weight: bold; font-size: 16px; }
            QPushButton:hover { background-color: #0052A3; }
            QPushButton:disabled { background-color: #A0C4E5; }
        """)

        self.stop_btn = QPushButton("⏹  挂起计算进程")
        self.stop_btn.setMinimumHeight(48)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton { background-color: #D9383A; color: white; border-radius: 6px; font-weight: bold; font-size: 16px; }
            QPushButton:hover { background-color: #B32021; }
            QPushButton:disabled { background-color: #ECA3A4; }
        """)

        btn_layout.addWidget(self.start_btn, 3)
        btn_layout.addWidget(self.stop_btn, 1)

        self.status_bar = QProgressBar()
        self.status_bar.setValue(0)
        self.status_bar.setFixedHeight(8)

        # 黑客终端视觉风
        self.log_txt = QTextEdit()
        self.log_txt.setReadOnly(True)
        self.log_txt.setMaximumHeight(85)
        self.log_txt.setStyleSheet("background-color:#1E1E1E; color:#00FF00; font-family: Consolas; font-size: 13px;")

        console_layout.addLayout(btn_layout)
        console_layout.addWidget(self.status_bar)
        console_layout.addWidget(self.log_txt)
        right_layout.addWidget(bottom_console, 2)

        main_splitter.addWidget(right_widget)
        h_layout = QHBoxLayout(main_widget)
        h_layout.addWidget(main_splitter)

        # 调配布局宽窄度
        main_splitter.setSizes([450, 1150])
        self.model_splitter.setSizes([575, 575])

        # 信号绑定
        self.combo_load.currentIndexChanged.connect(self.on_load_preset)
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.start_btn.clicked.connect(lambda: self.start_optimization_with_params(self.get_params(), None, None))

        # 默认选中 LC-06 稍微居中的效果
        self.combo_load.setCurrentIndex(5)
        self.on_load_preset(5)

    def on_load_preset(self, index):
        db = self.crh380a_db[index]
        self.spin_z.setValue(db["Z"])
        self.spin_y.setValue(db["Y"])
        self.spin_x.setValue(db["X"])
        self.spin_twist.setValue(db["Twist"])
        self.info_box.setHtml(f"<b style='color:#0066CC'>[工况解读]</b> {db['name']}<br><hr>"
                              f"<span style='color:#555;'>物理背景：{db['note']}</span>")

    def log(self, text):
        self.log_txt.append(text)
        self.log_txt.verticalScrollBar().setValue(self.log_txt.verticalScrollBar().maximum())

    def update_plot(self, rho_arr, dc_arr):
        try:
            tdim = self.domain.topology.dim
            self.domain.topology.create_connectivity(tdim, 0)
            num_cells = self.domain.topology.index_map(tdim).size_local
            cell_Nodes_1d = self.domain.topology.connectivity(tdim, 0).array

            cell_type_str = str(self.domain.topology.cell_type)
            if "tetrahedron" in cell_type_str:
                npc, vtk_type = 4, pv.CellType.TETRA
            elif "hexahedron" in cell_type_str:
                npc, vtk_type = 8, pv.CellType.HEXAHEDRON
            else:
                return

            cell_nodes_2d = cell_Nodes_1d.reshape((num_cells, npc))
            padding = np.full((num_cells, 1), npc, dtype=np.int64)
            cells_vtk = np.hstack([padding, cell_nodes_2d]).flatten()
            celltypes = np.full(num_cells, vtk_type, dtype=np.uint8)

            mesh_vtk = pv.UnstructuredGrid(cells_vtk, celltypes, self.domain.geometry.x)

            # 数据涂装
            mesh_vtk.cell_data["density"] = rho_arr.astype(np.float64)
            stress_px = np.abs(dc_arr)
            p99 = np.percentile(stress_px, 99) if len(stress_px) > 0 else 1.0
            mesh_vtk.cell_data["Strain Energy"] = np.clip(stress_px, 0, p99)

            # 切掉边角料保留实体
            solid_mesh = mesh_vtk.threshold(value=0.4, scalars="density", invert=False)

            # 🖥️ 【左侧屏幕】实体模型追踪
            self.plotter_left.clear_actors()
            if solid_mesh.n_cells > 0:
                self.plotter_left.add_mesh(solid_mesh, color="#1D5C92", opacity=1.0, show_edges=True,
                                           edge_color="black")
                shell = mesh_vtk.threshold(value=0.0, scalars="density", invert=False)
                self.plotter_left.add_mesh(shell, color="#999999", style="wireframe", opacity=0.08)

            # 🖥️ 【右侧屏幕】应变能流图
            self.plotter_right.clear_actors()
            self.plotter_right.add_mesh(mesh_vtk, scalars="Strain Energy", cmap="turbo", show_edges=True,
                                        edge_color="darkgray")

            # 双屏同步镜头
            if not hasattr(self, "_cam_init"):
                self.plotter_left.view_isometric()
                self.plotter_right.camera = self.plotter_left.camera
                self.plotter_left.iren.add_observer("InteractionEvent", lambda o, e: self.plotter_right.render())
                self.plotter_right.iren.add_observer("InteractionEvent", lambda o, e: self.plotter_left.render())
                self._cam_init = True

            self.plotter_left.update()
            self.plotter_right.update()

        except Exception as e:
            self.log(f"渲染重构提示: {e}")

    def update_progress(self, it, c, vol):
        max_iter = self.inputs["MMA推演技最大代:"].value()
        self.status_bar.setValue(int(min(100, (it / max_iter) * 100)))

    def stop_optimization(self):
        if self.worker:
            self.worker.is_running = False
            self.log("接到干预信号，正在平滑挂起退出...")

    def optimization_done(self):
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.inputs["MMA推演技最大代:"].setEnabled(True)
        self.status_bar.setValue(100)

    def get_params(self):
        p = {'material': {'E0': 200e9, 'nu': 0.3}}
        p['optimization'] = {
            'vol_frac': self.inputs["体积分数容限率:"].value(), 'p': self.inputs["伪度惩罚阶次:"].value(),
            'rmin': self.inputs["网状结构过滤半径:"].value(), 'max_iter': self.inputs["MMA推演技最大代:"].value()
        }

        val_z, val_y, val_x = self.spin_z.value(), self.spin_y.value(), self.spin_x.value()
        active_loads = []
        if abs(val_z) > 1e-3: active_loads.append({"dir": (0, 0, -1), "mag": val_z, "weight": 1.0})
        if abs(val_y) > 1e-3: active_loads.append({"dir": (0, 1, 0), "mag": val_y, "weight": 1.0})
        if abs(val_x) > 1e-3: active_loads.append({"dir": (1, 0, 0), "mag": val_x, "weight": 1.0})
        if not active_loads: active_loads.append({"dir": (0, 0, -1), "mag": 1e-3, "weight": 1.0})

        # ✨ 关键：把 active_loads 也直接拍在根目录下，防止底层的任何意外
        p['active_loads'] = active_loads

        p['load'] = {
            'active_loads': active_loads,
            'enable_vert': abs(val_z) > 1e-3, 'mag_vert': val_z,
            'enable_lat': abs(val_y) > 1e-3, 'mag_lat': val_y,
            'enable_long': abs(val_x) > 1e-3, 'mag_long': val_x,
            'twist_mm': self.spin_twist.value()
        }

        dir_map = {"Z 向 (车底脱模)": [0, 0, 1], "Y 向": [0, 1, 0], "X 向": [1, 0, 0]}
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
        self.inputs["MMA推演技最大代:"].setEnabled(False)
        self.status_bar.setValue(0)
        self.log(
            f"📌 复合受力组合注入底层！ Z={params['load']['mag_vert']} | Y={params['load']['mag_lat']} | X={params['load']['mag_long']}")

        if optimizer is None:
            class DummyOpt:
                def __init__(self):
                    self.fea, self.params, self.vol_frac = None, params, params['optimization']['vol_frac']
                    self.history = {'compliance': [], 'vol': [], 'change': []}
                    self.casting_constraint = None

            optimizer, rho_arr_test = DummyOpt(), np.zeros(10)
            bounds = (np.ones(10), np.ones(10))
        else:
            # 🚀 最核心的一步！强行把 UI 刷新的参数字典扣入 `main.py` 喂进来的 optimizer 里
            optimizer.params.update(params)

            bounds = (np.zeros_like(rho_func.x.array), np.ones_like(rho_func.x.array))
            rho_arr_test = rho_func.x.array

        self.worker = OptimizationWorker(optimizer, bounds, params['optimization']['max_iter'], rho_arr_test)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log)
        self.worker.plot_signal.connect(self.update_plot)
        self.worker.finished_signal.connect(self.optimization_done)
        self.worker.start()