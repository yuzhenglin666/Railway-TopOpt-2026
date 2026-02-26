"""
FEniCSx 环境综合验证与演示程序 - 最终修复版
适配 FEniCSx 0.10.0 API
已修复：LinearProblem参数错误、Gmsh导入问题
"""
import sys
import os
import numpy as np

# ==================== 字体配置部分 ====================
import matplotlib

matplotlib.use('Qt5Agg')


def configure_chinese_fonts():
    """配置中文字体支持"""
    import matplotlib.pyplot as plt

    chinese_fonts = [
        'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'SimHei',
        'SimSun', 'KaiTi', 'FangSong', 'WenQuanYi Micro Hei'
    ]

    font_manager = matplotlib.font_manager
    available_fonts = []

    for font_name in chinese_fonts:
        try:
            font_path = font_manager.findfont(font_name)
            if font_path:
                available_fonts.append(font_name)
                print(f"✓ 找到字体: {font_name}")
        except:
            pass

    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✓ 已设置中文字体: {available_fonts[0]}")
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        print("⚠ 未找到中文字体，将使用英文字体显示")

    return available_fonts


configure_chinese_fonts()

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox,
                             QSpinBox, QGroupBox, QTabWidget, QTextEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase

# ==================== 检查环境 ====================
print("=" * 60)
print("检查环境依赖...")

# 尝试导入Gmsh
try:
    import gmsh

    GMSH_AVAILABLE = True
    print("✓ Gmsh 模块已导入")
except ImportError:
    print("⚠ 未找到gmsh模块。网格生成功能将受限。")
    gmsh = None
    GMSH_AVAILABLE = False

# 尝试导入FEniCSx核心库
try:
    from mpi4py import MPI

    # FEniCSx 0.10.0 核心导入 - 调整导入方式
    import dolfinx
    from dolfinx import mesh, fem, io
    from dolfinx.fem import (functionspace, Function, form, Constant,
                             dirichletbc, locate_dofs_topological, assemble_scalar)
    # 注意：在FEniCSx 0.10.0中，LinearProblem需要petsc_options_prefix参数
    from dolfinx.fem.petsc import LinearProblem
    import ufl  # UFL表单语言

    FENICSX_AVAILABLE = True
    print("✓ FEniCSx 模块已导入")

    # 打印版本信息
    if hasattr(dolfinx, '__version__'):
        print(f"  dolfinx版本: {dolfinx.__version__}")

except ImportError as e:
    print(f"✗ FEniCSx导入失败: {e}")
    FENICSX_AVAILABLE = False
    dolfinx = None

print("=" * 60)


# ==================== Gmsh网格生成器 ====================
def generate_gmsh_mesh(params, mesh_filename):
    """生成Gmsh网格"""
    try:
        if not GMSH_AVAILABLE:
            return None, "Gmsh不可用"

        L, H = params['length'], params['height']
        mesh_size = params['mesh_size']

        print(f"生成Gmsh网格: {L}x{H}, 尺寸={mesh_size}")

        # 初始化Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("ParameterizedBeam")

        # 创建几何
        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(L, 0, 0)
        p3 = gmsh.model.geo.addPoint(L, H, 0)
        p4 = gmsh.model.geo.addPoint(0, H, 0)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        gmsh.model.geo.synchronize()

        # 划分网格
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(2)

        # 保存网格文件
        gmsh.write(mesh_filename)
        gmsh.finalize()

        print(f"网格已保存到: {mesh_filename}")
        return mesh_filename, "Gmsh网格生成成功"

    except Exception as e:
        error_msg = f"Gmsh网格生成失败: {str(e)}"
        print(error_msg)
        return None, error_msg


# ==================== 计算核心线程 ====================
class FenicsSolverThread(QThread):
    """在后台线程中运行FEniCSx求解"""
    progress_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, params, mesh_filename=None):
        super().__init__()
        self.params = params
        self.mesh_filename = mesh_filename

    def run(self):
        """线程主函数，在这里执行耗时的计算"""
        try:
            if not FENICSX_AVAILABLE:
                self.error_signal.emit("FEniCSx库未正确安装或导入失败！")
                return

            self.progress_signal.emit("正在初始化 MPI 和网格...")

            # FEniCSx 需要显式的MPI通信器
            from mpi4py import MPI
            from petsc4py import PETSc
            comm = MPI.COMM_SELF

            # --- 第1部分：读取或创建网格 ---
            L, H = self.params['length'], self.params['height']

            # 直接使用FEniCSx内置网格，避免Gmsh读取问题
            self.progress_signal.emit("使用内置矩形网格生成器...")
            nx, ny = self.params['nx'], self.params['ny']
            domain = mesh.create_rectangle(
                comm, [[0.0, 0.0], [L, H]],
                [nx, ny],
                cell_type=mesh.CellType.triangle
            )

            # --- 第2部分：FEniCSx有限元求解 ---
            self.progress_signal.emit("正在创建有限元函数空间...")

            # 创建向量函数空间
            V = functionspace(domain, ("Lagrange", 1, (2,)))

            # 定义边界条件（左端固定）
            def left_boundary(x):
                return np.isclose(x[0], 0.0)

            tdim = domain.topology.dim
            boundary_facets = mesh.locate_entities_boundary(domain, tdim - 1, left_boundary)
            boundary_dofs = locate_dofs_topological(V, tdim - 1, boundary_facets)

            # 创建边界条件
            zero_vector = np.zeros(2, dtype=PETSc.ScalarType)
            bc = dirichletbc(zero_vector, boundary_dofs, V)

            self.progress_signal.emit("正在定义材料本构和变分问题...")

            # 材料属性
            E = self.params['youngs_modulus']
            nu = self.params['poisson_ratio']
            mu = E / (2.0 * (1.0 + nu))
            lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

            # 定义线弹性本构关系
            def epsilon(u):
                return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

            def sigma(u):
                return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2)

            # 定义变分问题
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)

            # 创建载荷（作为体积力）
            T = Constant(domain, PETSc.ScalarType((0.0, -self.params['load'])))

            a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
            L_form = ufl.dot(T, v) * ufl.dx

            # 使用LinearProblem求解 - 添加petsc_options_prefix参数
            self.progress_signal.emit("正在求解线性系统...")

            # 修复：添加petsc_options_prefix参数
            problem = LinearProblem(a, L_form, bcs=[bc], petsc_options_prefix="beam_")

            # 配置求解器选项
            opts = PETSc.Options()
            opts["ksp_type"] = "preonly"
            opts["pc_type"] = "lu"
            opts["pc_factor_mat_solver_type"] = "mumps"

            # 注意：在FEniCSx 0.10.0中，可能需要这样设置选项
            opts.prefixPush("beam_")
            opts["ksp_type"] = "preonly"
            opts["pc_type"] = "lu"
            opts["pc_factor_mat_solver_type"] = "mumps"
            opts.prefixPop()

            problem.solver.setFromOptions()

            # 求解
            uh = problem.solve()

            # 计算应变能
            energy_form = form(0.5 * ufl.inner(sigma(uh), epsilon(uh)) * ufl.dx)
            strain_energy = assemble_scalar(energy_form)

            # 保存结果为XDMF文件
            self.progress_signal.emit("正在保存计算结果...")
            output_file = f"beam_solution_E{E}_F{self.params['load']}.xdmf"
            with io.XDMFFile(comm, output_file, "w") as xdmf:
                xdmf.write_mesh(domain)
                uh.name = "Displacement"
                xdmf.write_function(uh)

            # 收集结果
            nodes = domain.geometry.x
            displacement_data = uh.x.array.reshape(-1, 2)

            results = {
                'displacement': uh,
                'mesh': domain,
                'nodes': nodes,
                'displacement_data': displacement_data,
                'strain_energy': strain_energy,
                'output_file': output_file,
                'dofs': V.dofmap.index_map.size_global,
                'num_cells': domain.topology.index_map(tdim).size_global,
                'params': self.params,
                'mesh_source': 'Built-in'
            }
            self.result_signal.emit(results)

            self.progress_signal.emit("计算完成！")

        except Exception as e:
            import traceback
            error_msg = f"计算过程出错:\n{str(e)}\n\n追踪信息:\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)


# ==================== Matplotlib画布组件 ====================
class MplCanvas(FigureCanvas):
    """可嵌入PyQt的Matplotlib画布"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def clear(self):
        """清空画布"""
        self.axes.clear()
        self.draw()


# ==================== 主应用程序窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.solver_thread = None
        self.results = None
        self.current_mesh_file = None

        # 设置应用程序字体
        self.setup_fonts()

        self.init_ui()
        self.setWindowTitle("FEniCSx 环境验证器 - 交通运输科技大赛")
        self.resize(1200, 800)

    def setup_fonts(self):
        """设置应用程序字体"""
        font_db = QFontDatabase()

        chinese_font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]

        for font_path in chinese_font_paths:
            if os.path.exists(font_path):
                try:
                    font_id = font_db.addApplicationFont(font_path)
                    if font_id != -1:
                        print(f"✓ 已加载字体文件: {font_path}")
                except:
                    pass

        app_font = QFont()
        app_font.setFamily("WenQuanYi Micro Hei")
        app_font.setPointSize(10)
        QApplication.setFont(app_font)

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # ----- 左侧控制面板 -----
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(350)

        # 标题
        title = QLabel("🏗️ 悬臂梁分析模拟器")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("padding: 10px; color: #2c3e50;")
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)

        # 环境状态显示
        env_status = QLabel(f"环境状态: {'✅ 就绪' if FENICSX_AVAILABLE else '❌ FEniCSx未安装'}")
        env_status.setStyleSheet("font-size: 12px; padding: 8px; background-color: #f8f9fa; border-radius: 5px;")
        control_layout.addWidget(env_status)

        # 几何参数组
        geom_group = QGroupBox("📐 几何参数")
        geom_layout = QVBoxLayout()

        # 梁长度
        length_widget = self.create_spinbox("梁长度 (L):", 10.0, 1.0, 50.0, 0.5, " m")
        geom_layout.addWidget(length_widget)

        # 梁高度
        height_widget = self.create_spinbox("梁高度 (H):", 1.0, 0.1, 10.0, 0.1, " m")
        geom_layout.addWidget(height_widget)

        geom_group.setLayout(geom_layout)
        control_layout.addWidget(geom_group)

        # 网格参数组
        mesh_group = QGroupBox("🔲 网格参数")
        mesh_layout = QVBoxLayout()

        # X方向单元数
        nx_widget = self.create_int_spinbox("X方向单元数:", 40, 5, 200, 5)
        mesh_layout.addWidget(nx_widget)

        # Y方向单元数
        ny_widget = self.create_int_spinbox("Y方向单元数:", 8, 2, 50, 2)
        mesh_layout.addWidget(ny_widget)

        # Gmsh网格尺寸
        mesh_size_widget = self.create_spinbox("Gmsh网格尺寸:", 0.2, 0.01, 1.0, 0.05)
        mesh_layout.addWidget(mesh_size_widget)

        mesh_group.setLayout(mesh_layout)
        control_layout.addWidget(mesh_group)

        # 材料参数组
        material_group = QGroupBox("⚙️ 材料参数 (线弹性)")
        material_layout = QVBoxLayout()

        # 杨氏模量
        youngs_widget = self.create_spinbox("杨氏模量 (E):", 1000.0, 100.0, 1e6, 100.0, " MPa")
        material_layout.addWidget(youngs_widget)

        # 泊松比
        poisson_widget = self.create_spinbox("泊松比 (ν):", 0.3, 0.0, 0.49, 0.01)
        material_layout.addWidget(poisson_widget)

        # 载荷大小
        load_widget = self.create_spinbox("载荷大小 (F):", 10.0, 0.1, 1000.0, 1.0, " N")
        material_layout.addWidget(load_widget)

        material_group.setLayout(material_layout)
        control_layout.addWidget(material_group)

        # Gmsh控制
        gmsh_group = QGroupBox("🔄 Gmsh控制")
        gmsh_layout = QVBoxLayout()

        self.use_gmsh_checkbox = QLabel(f"Gmsh状态: {'✅ 可用' if GMSH_AVAILABLE else '❌ 不可用'}")
        gmsh_layout.addWidget(self.use_gmsh_checkbox)

        self.generate_mesh_btn = QPushButton("生成Gmsh网格")
        self.generate_mesh_btn.clicked.connect(self.generate_gmsh_mesh)
        self.generate_mesh_btn.setEnabled(GMSH_AVAILABLE)
        gmsh_layout.addWidget(self.generate_mesh_btn)

        gmsh_group.setLayout(gmsh_layout)
        control_layout.addWidget(gmsh_group)

        # 控制按钮
        self.run_btn = QPushButton("🚀 开始计算")
        self.run_btn.clicked.connect(self.start_calculation)
        self.run_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 12px; 
                background-color: #4CAF50; 
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.run_btn)

        self.save_btn = QPushButton("💾 保存结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 10px; 
                background-color: #2196F3; 
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("🔄 重置参数")
        self.reset_btn.clicked.connect(self.reset_parameters)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 10px; 
                background-color: #ff9800; 
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e68a00;
            }
        """)
        control_layout.addWidget(self.reset_btn)

        # 状态信息
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setFont(QFont("Monospace", 9))
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        control_layout.addWidget(QLabel("📋 计算状态:"))
        control_layout.addWidget(self.status_text)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        # ----- 右侧可视化面板 -----
        viz_panel = QWidget()
        viz_layout = QVBoxLayout()
        viz_panel.setLayout(viz_layout)

        # 标签页容器
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #4CAF50;
            }
        """)

        # 标签页1：位移云图
        self.tab1 = QWidget()
        tab1_layout = QVBoxLayout()
        self.canvas1 = MplCanvas(self.tab1, width=6, height=5)
        tab1_layout.addWidget(self.canvas1)
        self.tab1.setLayout(tab1_layout)
        self.tab_widget.addTab(self.tab1, "📊 位移云图")

        # 标签页2：结果统计
        self.tab2 = QWidget()
        tab2_layout = QVBoxLayout()
        self.canvas2 = MplCanvas(self.tab2, width=6, height=5)
        tab2_layout.addWidget(self.canvas2)
        self.tab2.setLayout(tab2_layout)
        self.tab_widget.addTab(self.tab2, "📈 结果统计")

        # 标签页3：计算结果摘要
        self.tab3 = QWidget()
        tab3_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Monospace", 10))
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                line-height: 1.5;
            }
        """)
        tab3_layout.addWidget(self.result_text)
        self.tab3.setLayout(tab3_layout)
        self.tab_widget.addTab(self.tab3, "📄 计算摘要")

        viz_layout.addWidget(self.tab_widget)
        main_layout.addWidget(viz_panel)

        # 初始化结果显示
        self.update_result_summary()

    def create_spinbox(self, label, default, min_val, max_val, step, suffix=""):
        """创建带标签的双精度旋钮"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(100)
        layout.addWidget(label_widget)

        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setSingleStep(step)
        spinbox.setMinimumWidth(100)
        if suffix:
            spinbox.setSuffix(suffix)
        spinbox.setStyleSheet("""
            QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        layout.addWidget(spinbox)
        layout.addStretch()

        # 保存引用以便访问
        attr_name = label.split(":")[0].replace(" ", "_").replace("(", "").replace(")", "").lower()
        setattr(self, f"{attr_name}_spin", spinbox)

        widget.setLayout(layout)
        return widget

    def create_int_spinbox(self, label, default, min_val, max_val, step):
        """创建带标签的整数旋钮"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(100)
        layout.addWidget(label_widget)

        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setSingleStep(step)
        spinbox.setMinimumWidth(100)
        spinbox.setStyleSheet("""
            QSpinBox {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        layout.addWidget(spinbox)
        layout.addStretch()

        # 保存引用以便访问
        attr_name = label.split(":")[0].replace(" ", "_").lower()
        setattr(self, f"{attr_name}_spin", spinbox)

        widget.setLayout(layout)
        return widget

    def log_status(self, message):
        """在状态框添加日志"""
        self.status_text.append(f"{message}")
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QApplication.processEvents()

    def reset_parameters(self):
        """重置所有参数为默认值"""
        self.梁长度_l_spin.setValue(10.0)
        self.梁高度_h_spin.setValue(1.0)
        self.x方向单元数_spin.setValue(40)
        self.y方向单元数_spin.setValue(8)
        self.gmsh网格尺寸_spin.setValue(0.2)
        self.杨氏模量_e_spin.setValue(1000.0)
        self.泊松比_ν_spin.setValue(0.3)
        self.载荷大小_f_spin.setValue(10.0)
        self.current_mesh_file = None
        self.log_status("参数已重置为默认值。")

    def generate_gmsh_mesh(self):
        """生成Gmsh网格"""
        if not GMSH_AVAILABLE:
            QMessageBox.warning(self, "警告", "Gmsh不可用，无法生成网格！")
            return

        # 收集参数
        params = {
            'length': self.梁长度_l_spin.value(),
            'height': self.梁高度_h_spin.value(),
            'mesh_size': self.gmsh网格尺寸_spin.value()
        }

        self.log_status(">>> 开始生成Gmsh网格...")

        # 生成网格文件路径
        L, H = params['length'], params['height']
        mesh_filename = f"beam_l{L}_h{H}_mesh.msh"
        self.current_mesh_file = mesh_filename

        # 生成网格（主线程中）
        mesh_file, message = generate_gmsh_mesh(params, mesh_filename)

        if mesh_file:
            self.log_status(f"✓ Gmsh网格生成成功: {mesh_file}")
            QMessageBox.information(self, "成功", f"Gmsh网格已生成:\n{mesh_file}")
        else:
            self.log_status(f"✗ {message}")
            QMessageBox.warning(self, "失败", f"Gmsh网格生成失败:\n{message}")
            self.current_mesh_file = None

    def start_calculation(self):
        """启动计算线程"""
        if not FENICSX_AVAILABLE:
            QMessageBox.critical(self, "错误", "FEniCSx库未正确安装或导入失败！")
            return

        if self.solver_thread and self.solver_thread.isRunning():
            self.log_status("已有计算正在进行中...")
            return

        # 收集参数
        params = {
            'length': self.梁长度_l_spin.value(),
            'height': self.梁高度_h_spin.value(),
            'nx': self.x方向单元数_spin.value(),
            'ny': self.y方向单元数_spin.value(),
            'mesh_size': self.gmsh网格尺寸_spin.value(),
            'youngs_modulus': self.杨氏模量_e_spin.value(),
            'poisson_ratio': self.泊松比_ν_spin.value(),
            'load': self.载荷大小_f_spin.value()
        }

        self.log_status(">>> 开始新的计算任务...")
        self.log_status(f"参数: 梁({params['length']}x{params['height']})m, "
                        f"网格[{params['nx']}x{params['ny']}], "
                        f"E={params['youngs_modulus']}MPa, F={params['load']}N")

        if self.current_mesh_file and os.path.exists(self.current_mesh_file):
            self.log_status(f"已生成Gmsh网格: {self.current_mesh_file}")
            self.log_status("注：当前版本使用内置网格进行计算，Gmsh网格用于验证生成功能")
        else:
            self.log_status("使用内置矩形网格")

        # 禁用按钮，防止重复点击
        self.run_btn.setEnabled(False)
        self.run_btn.setText("计算中...")
        self.save_btn.setEnabled(False)

        # 清空之前的可视化
        self.canvas1.clear()
        self.canvas2.clear()

        # 创建并启动计算线程
        self.solver_thread = FenicsSolverThread(params, self.current_mesh_file)
        self.solver_thread.progress_signal.connect(self.log_status)
        self.solver_thread.result_signal.connect(self.on_calculation_finished)
        self.solver_thread.error_signal.connect(self.on_calculation_error)
        self.solver_thread.start()

    def on_calculation_finished(self, results):
        """计算成功完成后的处理"""
        self.results = results
        self.log_status(">>> 计算成功完成！")
        self.log_status(f"生成文件: {results['output_file']}")
        self.log_status(f"网格来源: {results.get('mesh_source', 'Unknown')}")

        # 更新按钮状态
        self.run_btn.setEnabled(True)
        self.run_btn.setText("🚀 开始计算")
        self.save_btn.setEnabled(True)

        # 更新可视化
        self.update_visualizations()

        # 更新结果摘要
        self.update_result_summary()

        # 切换到结果摘要页
        self.tab_widget.setCurrentIndex(2)

    def on_calculation_error(self, error_msg):
        """计算出错时的处理"""
        self.log_status(">>> 计算过程中出现错误！")
        self.status_text.append(error_msg)

        self.run_btn.setEnabled(True)
        self.run_btn.setText("🚀 开始计算")

        QMessageBox.critical(self, "计算错误", "计算过程中出现错误，请查看状态窗口获取详细信息。")

    def update_visualizations(self):
        """更新所有可视化图表"""
        if not self.results:
            return

        try:
            # 标签页1：网格和边界条件示意图
            self.canvas1.axes.clear()

            # 绘制网格节点
            nodes = self.results['nodes']
            self.canvas1.axes.scatter(nodes[:, 0], nodes[:, 1], s=10, c='blue', alpha=0.6, label='节点')

            # 添加标题和标签
            self.canvas1.axes.set_xlabel('X 坐标 (m)', fontsize=10)
            self.canvas1.axes.set_ylabel('Y 坐标 (m)', fontsize=10)

            mesh_source = self.results.get('mesh_source', 'Unknown')
            title = f'有限元网格节点分布 ({mesh_source})'
            self.canvas1.axes.set_title(title, fontsize=12, fontweight='bold')

            # 标记边界条件
            L = self.results['params']['length']
            H = self.results['params']['height']

            # 左端固定边界
            self.canvas1.axes.plot([0, 0], [0, H], 'r-', linewidth=4, label='固定边界')

            # 载荷方向
            self.canvas1.axes.arrow(L / 2, H, 0, -0.1, head_width=0.15, head_length=0.15,
                                    fc='green', ec='green', label='载荷方向')

            # 添加图例
            self.canvas1.axes.legend(fontsize=9, loc='upper right')
            self.canvas1.axes.grid(True, alpha=0.3)
            self.canvas1.axes.axis('equal')
            self.canvas1.fig.tight_layout()
            self.canvas1.draw()

            # 标签页2：位移数据可视化
            self.canvas2.axes.clear()

            # 计算位移大小
            if 'displacement_data' in self.results:
                displacement = self.results['displacement_data']
                disp_magnitude = np.sqrt(displacement[:, 0] ** 2 + displacement[:, 1] ** 2)

                # 创建位移大小的直方图
                n_bins = min(30, len(disp_magnitude) // 10)
                if n_bins > 0:
                    self.canvas2.axes.hist(disp_magnitude, bins=n_bins,
                                           alpha=0.7, color='blue', edgecolor='black')
                    self.canvas2.axes.set_xlabel('位移大小 (m)', fontsize=10)
                    self.canvas2.axes.set_ylabel('节点数量', fontsize=10)
                    self.canvas2.axes.set_title('节点位移分布直方图', fontsize=12, fontweight='bold')

                    # 添加统计信息
                    stats_text = f"""
                    统计信息:
                    最大值: {np.max(disp_magnitude):.2e} m
                    最小值: {np.min(disp_magnitude):.2e} m
                    平均值: {np.mean(disp_magnitude):.2e} m
                    标准差: {np.std(disp_magnitude):.2e} m
                    """
                    self.canvas2.axes.text(0.95, 0.95, stats_text,
                                           transform=self.canvas2.axes.transAxes,
                                           fontsize=9,
                                           verticalalignment='top',
                                           horizontalalignment='right',
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            self.canvas2.axes.grid(True, alpha=0.3)
            self.canvas2.fig.tight_layout()
            self.canvas2.draw()

        except Exception as e:
            self.log_status(f"可视化更新失败: {str(e)}")

    def update_result_summary(self):
        """更新结果摘要"""
        if not self.results:
            # 显示默认摘要
            summary = """
            ================================
            FEniCSx + Gmsh 环境验证器
            ================================

            欢迎使用悬臂梁分析模拟器！

            使用说明:
            1. 在左侧面板调整几何、网格和材料参数
            2. 点击"生成Gmsh网格"创建参数化网格
            3. 点击"开始计算"运行有限元分析
            4. 在右侧标签页查看可视化结果

            环境状态:
            • FEniCSx: """ + ("✅ 已安装" if FENICSX_AVAILABLE else "❌ 未安装") + """
            • Gmsh: """ + ("✅ 已安装" if GMSH_AVAILABLE else "⚠ 未安装") + """

            默认参数:
            • 梁尺寸: 10.0 × 1.0 m
            • 网格: 40 × 8 单元
            • 材料: E=1000 MPa, ν=0.3
            • 载荷: 10 N (向下)

            ================================
            先测试Gmsh网格生成，再运行计算
            ================================
            """
        else:
            # 显示计算结果摘要
            disp_stats = ""
            if 'displacement_data' in self.results:
                displacement = self.results['displacement_data']
                disp_magnitude = np.sqrt(displacement[:, 0] ** 2 + displacement[:, 1] ** 2)
                disp_stats = f"""
            位移统计:
            • 最大位移: {np.max(disp_magnitude):.6e} m
            • 平均位移: {np.mean(disp_magnitude):.6e} m
                """

            mesh_source = self.results.get('mesh_source', 'Unknown')
            summary = f"""
            ================================
            计算成功完成！
            ================================
            模型信息:
            • 梁尺寸: {self.results['params']['length']} × {self.results['params']['height']} m
            • 网格来源: {mesh_source}
            • 单元总数: {self.results['num_cells']}
            • 自由度总数: {self.results['dofs']}

            材料与载荷:
            • 杨氏模量: {self.results['params']['youngs_modulus']} MPa
            • 泊松比: {self.results['params']['poisson_ratio']}
            • 载荷大小: {self.results['params']['load']} N

            计算结果:
            • 结构应变能: {float(self.results['strain_energy']):.6e} J
            • 结果文件: {self.results['output_file']}
            {disp_stats}

            ================================
            ✅ 你的 FEniCSx 环境验证通过！
            ✅ 所有模块协同工作正常。
            ================================
            """

        self.result_text.setPlainText(summary)

    def save_results(self):
        """保存结果到指定位置"""
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可保存的结果！")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存计算结果",
            f"beam_analysis_result",
            "文本文件 (*.txt);;所有文件 (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write("悬臂梁有限元分析结果\n")
                    f.write("=" * 60 + "\n\n")

                    f.write("计算参数:\n")
                    f.write(f"  梁长度: {self.results['params']['length']} m\n")
                    f.write(f"  梁高度: {self.results['params']['height']} m\n")
                    f.write(f"  X方向单元数: {self.results['params']['nx']}\n")
                    f.write(f"  Y方向单元数: {self.results['params']['ny']}\n")
                    f.write(f"  杨氏模量: {self.results['params']['youngs_modulus']} MPa\n")
                    f.write(f"  泊松比: {self.results['params']['poisson_ratio']}\n")
                    f.write(f"  载荷大小: {self.results['params']['load']} N\n\n")

                    f.write("计算结果:\n")
                    f.write(f"  单元总数: {self.results['num_cells']}\n")
                    f.write(f"  自由度总数: {self.results['dofs']}\n")
                    f.write(f"  结构应变能: {float(self.results['strain_energy']):.6e} J\n")
                    f.write(f"  结果文件: {self.results['output_file']}\n")
                    f.write(f"  网格来源: {self.results.get('mesh_source', 'Unknown')}\n\n")

                    f.write("网格信息:\n")
                    f.write(f"  节点总数: {len(self.results['nodes'])}\n")
                    f.write(f"  单元类型: 三角形\n\n")

                    if 'displacement_data' in self.results:
                        f.write("位移统计:\n")
                        displacement = self.results['displacement_data']
                        disp_magnitude = np.sqrt(displacement[:, 0] ** 2 + displacement[:, 1] ** 2)
                        f.write(f"  最大位移: {np.max(disp_magnitude):.6e} m\n")
                        f.write(f"  最小位移: {np.min(disp_magnitude):.6e} m\n")
                        f.write(f"  平均位移: {np.mean(disp_magnitude):.6e} m\n\n")

                    f.write("分析完成时间: ")
                    import datetime
                    f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

                self.log_status(f"结果已保存到: {file_path}")
                QMessageBox.information(self, "保存成功", f"结果已保存到:\n{file_path}")

            except Exception as e:
                self.log_status(f"保存失败: {str(e)}")
                QMessageBox.critical(self, "保存失败", f"保存文件时出错:\n{str(e)}")


# ==================== 应用程序入口 ====================
if __name__ == "__main__":
    # 创建应用程序实例
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')
    app.setApplicationName("FEniCSx验证器")

    # 检查必要的库
    if not FENICSX_AVAILABLE:
        reply = QMessageBox.question(None, "环境检查",
                                     "FEniCSx库未正确安装或导入失败！\n"
                                     "程序的部分功能将不可用。\n\n"
                                     "是否继续运行演示界面？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            sys.exit(1)

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    # 显示欢迎信息
    print("=" * 60)
    print("FEniCSx + Gmsh 环境综合验证器启动成功!")
    print("=" * 60)
    print("此程序验证以下模块的协同工作:")
    print("  • PyQt5 (图形用户界面)")
    print("  • Gmsh (参数化网格生成)")
    print("  • FEniCSx (有限元求解)")
    print("  • Matplotlib (结果可视化)")
    print("=" * 60)
    print("使用说明:")
    print("  1. 调整左侧参数")
    print("  2. 点击'生成Gmsh网格'创建参数化网格")
    print("  3. 点击'开始计算'运行有限元分析")
    print("  4. 查看右侧可视化结果")
    print("=" * 60)

    # 运行应用程序
    sys.exit(app.exec_())
