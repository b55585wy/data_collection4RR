import sys
import os
import time

import neurokit2
import serial
import serial.tools.list_ports
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, \
    QLineEdit, QComboBox, QListWidget, QMessageBox, QCheckBox, QInputDialog
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from collections import deque
from multiprocessing import Process, Queue, Value
from scipy import signal
import logging
from PyQt5.QtCore import QTimer
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO)


# Helper function to convert two's complement and scale
def twos_complement_and_scale(raw_data, range_val):
    return (float(raw_data) / 32767.0) * range_val


# FIR 低通滤波器设计
def design_fir_filter(cutoff_freq, sample_rate, num_taps):
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    taps = signal.firwin(num_taps, normalized_cutoff, window='hamming')
    print(f"Designed FIR filter with {num_taps} taps:")
    print(taps)  # 打印滤波器系数
    return taps


# 应用FIR滤波器
def apply_fir_filter(data, taps):
    filtered_data = signal.lfilter(taps, 1.0, data)
    return filtered_data  # 返回最新的滤波结果


# 保存数据到文件
def save_data_to_file(file_path, data):
    with open(file_path, 'a') as f:
        for d in data:
            f.write(','.join(map(str, d)) + '\n')


# nk.rsp_phase()
def detect_breathing_phase_by_derivative(signal_data):
    """
    Detect the respiratory phase (inhalation or exhalation) based on the signal derivative (slope).
    Parameters:
    signal : list or array
        The respiratory signal for which to detect phases.
    Returns:
    tuple
        A tuple with the current phase ('Inhalation' or 'Exhalation') and the completion percentage.
    """
    # 使用最近的100个数据点进行趋势分析
    if len(signal_data) < 100:
        smoothed_signal = np.array(signal_data)
    else:
        smoothed_signal = np.array(signal_data[-100:])

    # 计算信号的导数（变化率）
    derivative = np.diff(smoothed_signal)
    print(derivative)
    # 判断当前阶段，根据导数的符号确定呼吸阶段
    if derivative[-1] > 0:
        current_phase = "Inhalation"
    else:
        current_phase = "Exhalation"

    # 计算呼吸阶段完成度，可以根据信号的相对位置进行估计
    # 这里简单地根据导数的趋势变化来模拟呼吸的完成度
    phase_completion = np.abs(derivative[-1]) / np.max(np.abs(derivative)) if np.max(np.abs(derivative)) > 0 else 0

    return current_phase, phase_completion


# 获取串口设备列表
def get_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def cal_breathing_phases(signal_data):
    peaks, valleys = find_breathing_points(signal_data)
    print(f'peaks,valleys:{peaks}, {valleys}')


def find_breathing_points(signal_data):
    # 找到吸气点（峰值）
    peaks, _ = signal.find_peaks(signal_data)

    # 找到吐气点（谷值）
    valleys, _ = signal.find_peaks(-signal_data)
    return peaks, valleys


# 子进程中负责读取串口数据和滤波处理
def read_serial_data(serial_port, baud_rate, data_queue, filter_taps, running_flag, file_path, use_filter):
    try:
        ser = serial.Serial(serial_port, baud_rate)
        print(f"Connected to {serial_port} at {baud_rate} baud.")
    except serial.SerialException as e:
        print(f"Failed to connect to {serial_port}: {e}")
        return

    acc_range = 1  # Accelerometer range in g
    gyro_range = 1  # Gyroscope range in degrees/sec
    data_list = []  # 保存收集的数据
    start_time = time.time()  # 记录开始时间

    # Initialize buffers for accumulating data for filtering and phase detection
    acc_buffer = {'x_acc': [], 'y_acc': [], 'z_acc': [], 'x_gyro': [], 'y_gyro': [], 'z_gyro': []}
    while running_flag.value:
        if ser.in_waiting > 0:
            # try:
            # 读取串口数据
            line = ser.readline().decode('utf-8').strip()
            values = line.split(',')

            # 检查数据长度是否足够
            if len(values) != 6:
                raise ValueError(f"Incorrect data length: {line}")

            # 将字符串转换为整型，确保每个值都能转换成功
            try:
                raw_x_acc = int(values[0])
                raw_y_acc = int(values[1])
                raw_z_acc = int(values[2])
                raw_x_gyro = int(values[3])
                raw_y_gyro = int(values[4])
                raw_z_gyro = int(values[5])
            except ValueError as ve:
                raise ValueError(f"ValueError in converting values to int: {line}")

            # 转换和缩放加速度和角速度数据
            x_acc = twos_complement_and_scale(raw_x_acc, acc_range)
            y_acc = twos_complement_and_scale(raw_y_acc, acc_range)
            z_acc = twos_complement_and_scale(raw_z_acc, acc_range)
            x_gyro = twos_complement_and_scale(raw_x_gyro, gyro_range)
            y_gyro = twos_complement_and_scale(raw_y_gyro, gyro_range)
            z_gyro = twos_complement_and_scale(raw_z_gyro, gyro_range)
            origin_data = [x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro]

            # Accumulate data in buffers for filtering
            if use_filter:
                acc_buffer['x_acc'].append(x_acc)
                acc_buffer['y_acc'].append(y_acc)
                acc_buffer['z_acc'].append(z_acc)
                acc_buffer['x_gyro'].append(x_gyro)
                acc_buffer['y_gyro'].append(y_gyro)
                acc_buffer['z_gyro'].append(z_gyro)
                if len(acc_buffer['x_acc']) >= 150:
                    x_acc_filtered = apply_fir_filter(acc_buffer['x_acc'], filter_taps)
                    y_acc_filtered = apply_fir_filter(acc_buffer['y_acc'], filter_taps)
                    z_acc_filtered = apply_fir_filter(acc_buffer['z_acc'], filter_taps)
                    x_gyro_filtered = apply_fir_filter(acc_buffer['x_gyro'], filter_taps)
                    y_gyro_filtered = apply_fir_filter(acc_buffer['y_gyro'], filter_taps)
                    z_gyro_filtered = apply_fir_filter(acc_buffer['z_gyro'], filter_taps)

                    # Append the filtered data
                    filtered_data = [x_acc_filtered, y_acc_filtered, z_acc_filtered,
                                     x_gyro_filtered, y_gyro_filtered, z_gyro_filtered]

                    # cal_breathing_phases(filtered_data[0][-100:])
                    # 实时呼吸阶段检测

                    current_phase, phase_completion = detect_breathing_phase_by_derivative(list(filtered_data[0]))

                    print(f"Current phase: {current_phase}, Phase completion: {phase_completion:.2f}")
                    acc_buffer['x_acc'] = []
                    acc_buffer['y_acc'] = []
                    acc_buffer['z_acc'] = []
                    acc_buffer['x_gyro'] = []
                    acc_buffer['y_gyro'] = []
                    acc_buffer['z_gyro'] = []

            data_list.append(origin_data)

            # 每次收集1000个数据保存一次
            if len(data_list) >= 1000:
                save_data_to_file(file_path, data_list)
                data_list.clear()

            # 将原始数据发送到主进程进行绘图
            data_queue.put(origin_data)

    end_time = time.time()
    sampling_duration = end_time - start_time
    print(f"Sampling duration: {sampling_duration:.2f} seconds")

    if data_list:
        save_data_to_file(file_path, data_list)
        print(f"Saved remaining data to {file_path}")
    return sampling_duration


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

        self.kalman_filter = self.KalmanFilter(process_variance=1e-5, measurement_variance=1e-2,
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
        # 添加取消滤波器按钮的功能
        self.remove_filter_button = QPushButton("Remove Filter")
        self.remove_filter_button.clicked.connect(self.remove_filter)
        filter_button_layout.addWidget(self.remove_filter_button)

        # 应用FIR滤波器设置按钮
        self.apply_filter_button = QPushButton("Apply FIR Filter")
        self.apply_filter_button.clicked.connect(self.apply_filter_order)
        filter_button_layout.addWidget(self.apply_filter_button)

        left_layout.addLayout(filter_button_layout)
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

        controls_button_layout = QHBoxLayout()
        # 采集数据按钮
        self.start_collect_button = QPushButton("Start Collecting Data")
        self.start_collect_button.clicked.connect(self.start_collecting_data)
        controls_button_layout.addWidget(self.start_collect_button)

        self.stop_collect_button = QPushButton("Stop Collecting Data")
        self.stop_collect_button.clicked.connect(self.stop_collecting_data)
        controls_button_layout.addWidget(self.stop_collect_button)

        left_layout.addLayout(controls_button_layout)

        # 添加呼吸速率显示
        self.realtime_breathing_label = QLabel("Real-time Breathing Rate: 0 bpm")
        self.average_breathing_label = QLabel("first 60s Average Breathing Rate: 0 bpm")

        Breathing_label_layout = QHBoxLayout()
        # 将实时和平均呼吸速率标签加入右侧布局
        Breathing_label_layout.addWidget(self.realtime_breathing_label)
        Breathing_label_layout.addWidget(self.average_breathing_label)
        right_layout.addLayout(Breathing_label_layout)

        # 将左栏和右栏分别加入主布局
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 9)

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

        self.serial_process = Process(target=read_serial_data, args=(
            self.serial_port, 115200, self.data_queue, self.filter_taps, self.running_flag, self.file_path,
            self.use_filter))
        self.serial_process.start()
        print(f"Started collecting data: {file_name}")

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

    def update_plot_visibility(self):
        for key in self.checkboxes:
            if self.checkboxes[key].isChecked():
                self.plots[key].setVisible(True)
            else:
                self.plots[key].setVisible(False)
                self.x_vals[key].clear()
                self.plots[key].setData([], [])

    def calculate_breathing_rate(self, signal_data, sample_rate):
        fft_result = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), d=1 / sample_rate)

        positive_freqs = freqs[freqs > 0]
        positive_fft = np.abs(fft_result[freqs > 0])

        valid_idx = np.where((positive_freqs >= 0.05) & (positive_freqs <= 0.5))
        valid_freqs = positive_freqs[valid_idx]
        valid_fft = positive_fft[valid_idx]

        if len(valid_fft) == 0:
            print("No valid frequency components found in the 0.05Hz-0.5Hz range. Using last breathing rate.")
            return self.last_breathing_rate

        dominant_freq = valid_freqs[np.argmax(valid_fft)]
        breathing_rate = dominant_freq * 60

        self.last_breathing_rate = breathing_rate
        return breathing_rate

    class KalmanFilter:
        def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
            self.process_variance = process_variance
            self.measurement_variance = measurement_variance
            self.estimated_measurement_variance = estimated_measurement_variance
            self.posteri_estimate = 0.0
            self.posteri_error_estimate = 1.0
            self.last_breathing_rate = 0

        def update(self, measurement):
            priori_estimate = self.posteri_estimate
            priori_error_estimate = self.posteri_error_estimate + self.process_variance

            blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
            self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
            self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

            return self.posteri_estimate

    def fuse_data(self, x, y, z):
        return np.sqrt(x ** 2)

    def update_plot(self):
        current_time = time.time()

        if not self.stabilized and self.start_time:
            if current_time - self.start_time >= 5:
                self.stabilized = True
            return

        while not self.data_queue.empty():
            data = self.data_queue.get()
            x_acc, y_acc, z_acc = data[:3]

            fused_signal = self.fuse_data(x_acc, y_acc, z_acc)

            self.signal_buffer.append(fused_signal)



            if len(self.signal_buffer_first_60s) < 60 * 100:
                self.signal_buffer_first_60s.append(fused_signal)

            if len(self.signal_buffer_first_60s) == 60 * 100 and not self.average_calculated:
                self.average_calculated = True
                average_breathing_rate = self.calculate_breathing_rate(list(self.signal_buffer_first_60s),
                                                                       sample_rate=100)
                self.average_breathing_label.setText(f"60s Average Breathing Rate: {average_breathing_rate:.2f} bpm")

            if len(self.signal_buffer) >= self.min_signal_len:
                self.count_since_last_calculation += 1  # 每次满足条件后自增计数器

                if self.count_since_last_calculation >= 100:  # 满足100个点之后才计算
                    current_breathing_rate = self.calculate_breathing_rate(list(self.signal_buffer), sample_rate=100)
                    self.realtime_breathing_label.setText(f"Real-time Breathing Rate: {current_breathing_rate:.2f} bpm")
                    self.count_since_last_calculation = 0  # 重置计数器

            self.update_graph(data)

    def update_graph(self, data):
        keys = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']
        for i, key in enumerate(keys):
            if self.checkboxes[key].isChecked():
                self.x_vals[key].append(data[i])
                x_range = np.arange(len(self.x_vals[key]))
                self.plots[key].setData(x_range, list(self.x_vals[key]))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    if not os.path.exists('./data'):
        os.makedirs('./data')

    data_queue = Queue()
    running_flag = Value('i', 1)

    window = IMUPlotter(data_queue)
    window.filter_taps = design_fir_filter(0.5, 100, 21)
    window.running_flag = running_flag
    window.show()

    sys.exit(app.exec_())
