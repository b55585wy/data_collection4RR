import asyncio
import os
import threading
import time

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, \
    QLineEdit, QComboBox, QListWidget, QMessageBox, QCheckBox, QInputDialog, QSplitter
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from collections import deque
from multiprocessing import Process, Value
from PyQt5.QtCore import QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

from data_processing import KalmanFilter, calculate_breathing_rate, design_fir_filter, detect_breathing_phase_by_derivative, fuse_data
from serial_reader import get_serial_ports, read_serial_data
import socket
import subprocess

class IMUPlotter(QMainWindow):
    def __init__(self, data_queue, max_len=1000):
        super().__init__()
        self.serial_process = None
        self.data_queue = data_queue
        self.serial_ports = get_serial_ports()
        self.max_len = max_len
        self.min_signal_len = 1000
        self.signal_buffer = deque(maxlen=50 * 100)
        self.signal_buffer_first_60s = deque(maxlen=60 * 100)
        self.count_since_last_calculation = 0  # 初始化计数器

        self.kalman_filter = KalmanFilter(process_variance=1e-5, measurement_variance=1e-2,
                                               estimated_measurement_variance=1.0)
        self.last_breathing_rate = 0

        self.stabilized = False
        self.average_calculated = False
        self.start_time = None

        self.x_vals = {key: deque([0] * max_len, maxlen=max_len) for key in
                       ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']}
        self.users = ["User1", "User2"]
        self.use_filter = False
        self.colors = {
            'x_acc': 'r', 'y_acc': 'g', 'z_acc': 'b',
            'x_gyro': 'm', 'y_gyro': 'c', 'z_gyro': 'y'
        }
        self.plots = {}

        # 设置窗口标题和图标
        self.setWindowTitle("IMU Data Plotter")
        self.setWindowIcon(QIcon('path_to_your_icon.png'))  # 替换为你的图标路径

        # 设置窗口大小
        self.setGeometry(100, 100, 1200, 800)

        # 初始化UI
        self.init_ui()

        # 定时器用于更新图形界面和呼吸信息
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)

        # 定时器2：刷新串口列表
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_serial_ports)
        self.refresh_timer.start(5000)

        self.tcp_server = None
        self.tcp_clients = []
        self.tcp_thread = None

    def start_tcp_server(self, host='0.0.0.0', port=12242):
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.bind((host, port))
        self.tcp_server.listen(5)
        print(f"TCP server listening on {host}:{port}")
        self.tcp_thread = threading.Thread(target=self.accept_connections, daemon=True)
        self.tcp_thread.start()
        
        # 启动 tcp_client.py
        data_files = os.listdir('./data')
        if data_files:
            first_file = os.path.join('./data', data_files[0])
            subprocess.Popen(['python', 'tcp_client.py', host, str(port), first_file])
        else:
            print("No data files found in ./data directory")

        # 启动更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(10)  # 每10毫秒更新一次

    def accept_connections(self):
        while True:
            client, addr = self.tcp_server.accept()
            self.tcp_clients.append(client)
            threading.Thread(target=self.handle_client, args=(client,), daemon=True).start()

    def handle_client(self, client):
        while True:
            try:
                data = client.recv(1024).decode().strip()
                if not data:
                    break
                # 解析接收到的数据
                values = list(map(float, data.split(',')))
                if len(values) == 6:
                    self.data_queue.put(values)
            except:
                break
        client.close()
        self.tcp_clients.remove(client)

    def send_tcp_data(self, data):
        for client in self.tcp_clients:
            try:
                client.send(data.encode())
            except:
                self.tcp_clients.remove(client)

    def stop_tcp_server(self):
        if self.tcp_server:
            self.tcp_server.close()
            for client in self.tcp_clients:
                client.close()
            self.tcp_clients.clear()
            self.tcp_thread.join()
            print("TCP server stopped")

    def add_user(self):
        """添加用户"""
        new_user, ok = QInputDialog.getText(self, 'Add User', 'Enter new user name:')
        if ok and new_user:
            self.users.append(new_user)
            self.user_input.addItem(new_user)
            print(f"Added user: {new_user}")

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # 设置全局字体
        app = QApplication.instance()
        app.setFont(QFont('Arial', 10))

        # 设置左侧面板的样式
        left_panel = QWidget()
        left_panel.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border-right: 1px solid #cccccc;
            }
            QLabel {
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                margin: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QComboBox, QLineEdit {
                padding: 5px;
                margin: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        left_panel.setLayout(left_layout)

        # 设置右侧面板的样式
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)
        right_panel.setLayout(right_layout)

        # 使用QSplitter来允许用户调整左右面板的大小
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([300, 900])  # 置初始大小

        # 设置主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.splitter)

        # 用户选择和管理
        self.user_label = QLabel("Select User:")
        self.user_input = QComboBox()
        self.user_input.addItems(self.users)
        left_layout.addWidget(self.user_label)
        left_layout.addWidget(self.user_input)

        # 用户操作布局（水平布局）
        user_buttons_layout = QHBoxLayout()
        self.add_user_button = QPushButton("Add User")
        self.add_user_button.clicked.connect(self.add_user)
        user_buttons_layout.addWidget(self.add_user_button)

        self.delete_user_button = QPushButton("Delete User")
        self.delete_user_button.clicked.connect(self.delete_user)
        user_buttons_layout.addWidget(self.delete_user_button)

        left_layout.addLayout(user_buttons_layout)

        # 芯片类型选择
        self.chip_label = QLabel("Select Chip:")
        self.chip_input = QComboBox()
        self.chip_input.addItems(["C3", "S3"])
        left_layout.addWidget(self.chip_label)
        left_layout.addWidget(self.chip_input)

        # 动作类型和呼吸类型选择
        self.motion_type_label = QLabel("Motion Type:")
        self.motion_type_input = QComboBox()
        self.motion_type_input.addItems(["Standing", "Walking", "Sitting", "Lying"])
        left_layout.addWidget(self.motion_type_label)
        left_layout.addWidget(self.motion_type_input)

        self.breathing_type_label = QLabel("Breathing Type:")
        self.breathing_type_input = QComboBox()
        self.breathing_type_input.addItems(["Deep Breathing", "Shallow Breathing"])
        left_layout.addWidget(self.breathing_type_label)
        left_layout.addWidget(self.breathing_type_input)

        # 串口号选择
        self.serial_port_label = QLabel("Serial Port:")
        self.serial_port_input = QComboBox()
        self.serial_port_input.addItems(get_serial_ports())
        left_layout.addWidget(self.serial_port_label)
        left_layout.addWidget(self.serial_port_input)

        # 采样率设置
        self.sample_rate_label = QLabel("Sampling Rate:")
        self.sample_rate_input = QLineEdit("100")
        left_layout.addWidget(self.sample_rate_label)
        left_layout.addWidget(self.sample_rate_input)

        # FIR滤波器阶数设置
        self.filter_order_label = QLabel("Filter Order:")
        self.filter_order_input = QLineEdit("5")
        left_layout.addWidget(self.filter_order_label)
        left_layout.addWidget(self.filter_order_input)

        filter_button_layout = QHBoxLayout()
        

        
        # 绘图控件
        self.graphWidget = PlotWidget()
        self.graphWidget.enableAutoRange(axis='y')
        self.graphWidget.addLegend()
        right_layout.addWidget(self.graphWidget)

        # 初始化绘图对象
        for key, color in self.colors.items():
            self.plots[key] = self.graphWidget.plot(np.arange(self.max_len), np.zeros(self.max_len),
                                                    pen=pg.mkPen(color, width=2), name=key)

        # IMU复选框，控制是否绘制
        self.checkboxes = {}
        imu_keys = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']

        for i in range(0, len(imu_keys), 3):
            row_layout = QHBoxLayout()
            for key in imu_keys[i:i + 3]:
                checkbox = QCheckBox(f"Plot {key.replace('_', ' ').title()}")
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(self.update_plot_visibility)
                row_layout.addWidget(checkbox)
                self.checkboxes[key] = checkbox
            right_layout.addLayout(row_layout)

        # 显示文件列表
        self.file_list = QListWidget()
        left_layout.addWidget(self.file_list)
        self.load_files()  # 确保界面启动时显示已有的文件

        # 删除文件按钮
        self.delete_button = QPushButton("Delete Selected File")
        self.delete_button.clicked.connect(self.delete_file)
        left_layout.addWidget(self.delete_button)


        # 应用FIR滤波器设置按钮
        self.apply_filter_button = QPushButton("Apply FIR Filter")
        self.apply_filter_button.clicked.connect(self.apply_filter_order)
        filter_button_layout.addWidget(self.apply_filter_button)
        left_layout.addLayout(filter_button_layout)

        # 添加取消滤波器按钮的功能
        self.remove_filter_button = QPushButton("Remove Filter")
        self.remove_filter_button.clicked.connect(self.remove_filter)
        filter_button_layout.addWidget(self.remove_filter_button)
        left_layout.addLayout(filter_button_layout)
        controls_button_layout = QHBoxLayout()
        # 采集数据按钮
        self.start_collect_button = QPushButton("Start Collecting Data")
        self.start_collect_button.clicked.connect(self.start_collecting_data)
        controls_button_layout.addWidget(self.start_collect_button)

        self.stop_collect_button = QPushButton("Stop Collecting Data")
        self.stop_collect_button.clicked.connect(self.stop_collecting_data)
        controls_button_layout.addWidget(self.stop_collect_button)

        left_layout.addLayout(controls_button_layout)
        
        # 添加 TCP 连接控制按钮
        tcp_button_layout = QHBoxLayout()

        self.start_tcp_button = QPushButton("Start TCP Server")
        self.start_tcp_button.clicked.connect(lambda: self.start_tcp_server())
        tcp_button_layout.addWidget(self.start_tcp_button)

        self.stop_tcp_button = QPushButton("Stop TCP Server")
        self.stop_tcp_button.clicked.connect(self.stop_tcp_server)
        tcp_button_layout.addWidget(self.stop_tcp_button)

        left_layout.addLayout(tcp_button_layout)  # 将 TCP 控制按钮加入布局
        
        # 添加呼吸速率显示
        self.realtime_breathing_label = QLabel("Real-time Breathing Rate: 0 bpm")
        self.average_breathing_label = QLabel("first 60s Average Breathing Rate: 0 bpm")

        # 美化呼吸速率标签
        self.realtime_breathing_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #333333;
            background-color: #e6f3ff;
            padding: 5px;
            border-radius: 3px;
        """)
        self.average_breathing_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #333333;
            background-color: #e6ffe6;
            padding: 5px;
            border-radius: 3px;
        """)

        Breathing_label_layout = QHBoxLayout()
        # 将实时和平均呼吸速率标签加入右侧布局
        Breathing_label_layout.addWidget(self.realtime_breathing_label)
        Breathing_label_layout.addWidget(self.average_breathing_label)
        right_layout.addLayout(Breathing_label_layout)

        # 设置窗口中心控件
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def delete_user(self):
        """删除用户"""
        user_to_remove = self.user_input.currentText()
        if user_to_remove:
            reply = QMessageBox.question(self, 'Confirm', f"Are you sure you want to delete {user_to_remove}?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.users.remove(user_to_remove)
                self.user_input.removeItem(self.user_input.currentIndex())
                print(f"Deleted user: {user_to_remove}")

    def refresh_serial_ports(self):
        current_ports = get_serial_ports()
        if current_ports != self.serial_ports:
            self.serial_ports = current_ports
            self.serial_port_input.clear()
            self.serial_port_input.addItems(self.serial_ports)
            print("Serial ports updated:", self.serial_ports)

    def load_files(self):
        files = os.listdir('./data')
        self.file_list.clear()
        self.file_list.addItems(files)

    def delete_file(self):
        selected_file = self.file_list.currentItem().text()
        reply = QMessageBox.question(self, 'Confirm', f"Are you sure you want to delete {selected_file}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            os.remove(f'./data/{selected_file}')
            self.load_files()

    def apply_filter_order(self):
        try:
            order = int(self.filter_order_input.text())
            sample_rate = int(self.sample_rate_input.text())
            self.filter_taps = design_fir_filter(0.5, sample_rate, order)
            self.use_filter = True
            print(f"Filter order set to {order} with sample rate {sample_rate} Hz")

            if self.serial_process and self.serial_process.is_alive():
                self.running_flag.value = 0
                self.serial_process.join()

                self.running_flag.value = 1
                self.serial_process = Process(target=read_serial_data, args=(
                    self.serial_port, 115200, self.data_queue, self.filter_taps, self.running_flag, self.file_path))
                self.serial_process.start()
                print("Applied new filter and restarted data collection.")

        except ValueError:
            print("Invalid filter order or sample rate")

    def remove_filter(self):
        self.use_filter = False
        print("Filter removed. Showing raw data.")

        if self.serial_process and self.serial_process.is_alive():
            self.running_flag.value = 0
            self.serial_process.join()

            self.running_flag.value = 1
            self.serial_process = Process(target=read_serial_data, args=(
                self.serial_port, 115200, self.data_queue, None, self.running_flag, self.file_path))
            self.serial_process.start()
            print("Restarted data collection without filter.")

    def start_collecting_data(self):
        self.stabilized = False
        self.average_calculated = False
        self.start_time = time.time()

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        motion_type = self.motion_type_input.currentText()
        breathing_type = self.breathing_type_input.currentText()
        user = self.user_input.currentText()
        chip = self.chip_input.currentText()
        sample_rate = self.sample_rate_input.text()
        filter_order = self.filter_order_input.text()

        if self.use_filter:
            file_name = f"{user}_{chip}_{motion_type}_{breathing_type}_{sample_rate}Hz_FilterOrder{filter_order}_{timestamp}.csv"
        else:
            file_name = f"{user}_{chip}_{motion_type}_{breathing_type}_{sample_rate}Hz_raw_data_{timestamp}.csv"

        self.file_path = f"./data/{file_name}"

        self.serial_port = self.serial_port_input.currentText()
        self.running_flag.value = 1
        #初始化串口进程，如波特率，数据队列，滤波器，运行标志，文件路径，是否使用滤波器
        self.serial_process = Process(target=read_serial_data, args=(
            self.serial_port, 115200, self.data_queue, self.filter_taps, self.running_flag, self.file_path,
            self.use_filter))
        self.serial_process.start()
        print(f"Started collecting data: {file_name}")

        self.start_collect_button.setEnabled(False)
        self.stop_collect_button.setEnabled(True)

    def stop_collecting_data(self):
        end_time = time.time()
        sampling_duration = round(end_time - self.start_time, 2)

        reply = QMessageBox.question(self, 'Save Data',
                                     f"Sampling duration: {sampling_duration}s. Do you want to save this data?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.running_flag.value = 0
            self.serial_process.join()
            print(f"Data collection stopped and data saved. Sampling time: {sampling_duration}s")
        else:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            print("Data collection stopped and data discarded.")

        self.stabilized = False
        self.average_calculated = False
        self.load_files()

        self.start_collect_button.setEnabled(True)
        self.stop_collect_button.setEnabled(False)

    def update_plot_visibility(self):
        for key in self.checkboxes:
            if self.checkboxes[key].isChecked():
                self.plots[key].setVisible            
            else:
                self.plots[key].setVisible(False)
                self.x_vals[key].clear()
                self.plots[key].setData([], [])

    def update_plot(self):
        while not self.data_queue.empty():
            data = self.data_queue.get()
            self.process_and_plot_data(data)

    def process_and_plot_data(self, data):
        current_time = time.time()

        if not self.stabilized and self.start_time:
            if current_time - self.start_time >= 5:
                self.stabilized = True
            return

        x_acc, y_acc, z_acc = data[:3]

        fused_signal = fuse_data(x_acc, y_acc, z_acc)

        self.signal_buffer.append(fused_signal)

        if len(self.signal_buffer_first_60s) < 60 * 100:
            self.signal_buffer_first_60s.append(fused_signal)

        if len(self.signal_buffer_first_60s) == 60 * 100 and not self.average_calculated:
            self.average_calculated = True
            average_breathing_rate = calculate_breathing_rate(list(self.signal_buffer_first_60s),
                                                                   sample_rate=100)
            self.average_breathing_label.setText(f"60s Average Breathing Rate: {average_breathing_rate:.2f} bpm")

        if len(self.signal_buffer) >= self.min_signal_len:
            self.count_since_last_calculation += 1  # 每次满足条件后自增计数器

            if self.count_since_last_calculation >= 100:  # 满足100个点之后才计算
                current_breathing_rate = int(calculate_breathing_rate(list(self.signal_buffer), sample_rate=100, last_breathing_rate=self.last_breathing_rate))
                
                # 检测呼吸阶段
                current_phase, _ = detect_breathing_phase_by_derivative(list(self.signal_buffer))
                
                # 通过TCP发送呼吸阶段
                breathing_signal = "1" if current_phase == "Expiration" else "0"
                self.send_tcp_data(breathing_signal)
                
                if current_breathing_rate < 4:
                    self.realtime_breathing_label.setText(f"Invalid data: {current_breathing_rate} bpm")
                elif current_breathing_rate > 20:
                    self.realtime_breathing_label.setText("Invalid data")
                else:
                    if current_breathing_rate > 12:
                        self.realtime_breathing_label.setText(f"Breathing too fast! Rate: {current_breathing_rate} bpm")
                    else:
                        self.realtime_breathing_label.setText(f"Real-time Breathing Rate: {current_breathing_rate} bpm")
                self.count_since_last_calculation = 0  # 重置计数器

        self.update_graph(data)

    def update_graph(self, data):
        keys = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']
        for i, key in enumerate(keys):
            if self.checkboxes[key].isChecked():
                self.x_vals[key].append(data[i])
                x_range = np.arange(len(self.x_vals[key]))
                self.plots[key].setData(x_range, list(self.x_vals[key]))
    
