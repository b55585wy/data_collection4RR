import sys
import os
import time
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
# 设置日志记录
logging.basicConfig(level=logging.INFO)


# Helper function to convert two's complement and scale
def twos_complement_and_scale(raw_data, range_val):
    # if raw_data & 0x8000:
    #     raw_data = -((~raw_data + 1) & 0xFFFF)
    return (float(raw_data) / 32767.0) * range_val


# FIR 低通滤波器设计
def design_fir_filter(cutoff_freq, sample_rate, num_taps):
    nyquist = 0.5 * sample_rate  # 奈奎斯特频率
    normalized_cutoff = cutoff_freq / nyquist
    taps = signal.firwin(num_taps, normalized_cutoff, window='hamming')
    return taps


# 应用FIR滤波器
def apply_fir_filter(data, taps):
    # 使用 lfilter 滤波时，只对最近一次数据（单个值）进行滤波
    filtered_data = signal.lfilter(taps, 1.0, data)
    print(f'filtered data: {filtered_data}')
    return filtered_data[-1]  # 返回最新的滤波结果


# 保存数据到文件
def save_data_to_file(file_path, data):
    with open(file_path, 'a') as f:
        for d in data:
            f.write(','.join(map(str, d)) + '\n')


# 获取串口设备列表
def get_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


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

                if use_filter:  # 使用传入的 use_filter 变量
                    # 如果有滤波器，应用滤波
                    x_acc_filtered = apply_fir_filter([x_acc], filter_taps)
                    y_acc_filtered = apply_fir_filter([y_acc], filter_taps)
                    z_acc_filtered = apply_fir_filter([z_acc], filter_taps)
                    x_gyro_filtered = apply_fir_filter([x_gyro], filter_taps)
                    y_gyro_filtered = apply_fir_filter([y_gyro], filter_taps)
                    z_gyro_filtered = apply_fir_filter([z_gyro], filter_taps)
                    filtered_data = [x_acc_filtered, y_acc_filtered, z_acc_filtered, x_gyro_filtered, y_gyro_filtered,
                                     z_gyro_filtered]
                else:
                    # 显示原始数据
                    filtered_data = [x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro]

                data_list.append(filtered_data)

                # 每次收集1000个数据保存一次
                if len(data_list) >= 1000:
                    save_data_to_file(file_path, data_list)
                    data_list.clear()

                # 将滤波后的数据（或原始数据）发送到主进程进行绘图
                data_queue.put(filtered_data)

            # except ValueError as e:
            #     print(f"Invalid data received: {line}, Error: {e}")


    end_time = time.time()  # 记录结束时间
    sampling_duration = end_time - start_time  # 计算采样时长
    print(f"Sampling duration: {sampling_duration:.2f} seconds")

    # 停止时将剩余数据保存
    if data_list:
        save_data_to_file(file_path, data_list)
        print(f"Saved remaining data to {file_path}")
    return sampling_duration



class IMUPlotter(QMainWindow):
    def __init__(self, data_queue, max_len=1000):
        super().__init__()
        self.serial_process = None  # 初始化为None
        self.data_queue = data_queue
        self.serial_ports = get_serial_ports()  # 初始化串口列表属性
        self.max_len = max_len
        self.min_signal_len = 1000  # 设置最小信号长度，用于呼吸率计算
        self.signal_buffer = deque(maxlen=60 * 100)  # 60秒的信号缓冲区，假设采样率100Hz

        # 实例化卡尔曼滤波器
        self.kalman_filter = self.KalmanFilter(process_variance=1e-5, measurement_variance=1e-2,
                                               estimated_measurement_variance=1.0)

        self.x_vals = {key: deque([0] * max_len, maxlen=max_len) for key in
                       ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']}
        self.users = ["User1", "User2"]  # 初始用户列表
        self.use_filter = False
        self.colors = {
            'x_acc': 'r', 'y_acc': 'g', 'z_acc': 'b',
            'x_gyro': 'm', 'y_gyro': 'c', 'z_gyro': 'y'
        }
        self.plots = {}

        # 设置界面布局
        self.init_ui()

        # 定时器用于更新图形界面和呼吸信息
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 每50毫秒更新一次绘图

        # 定时器2：刷新串口列表（每5秒一次）
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_serial_ports)
        self.refresh_timer.start(5000)  # 每5秒刷新一次串口列表

        # self.fft_figure_timer = QTimer()
        # self.fft_figure.timeout.connect(self.fft_plot)
    def init_ui(self):
        main_layout = QHBoxLayout()  # 创建主布局，左右两栏
        left_layout = QVBoxLayout()  # 左边栏布局
        right_layout = QVBoxLayout()  # 右边栏布局

        # 用户选择和管理
        self.user_label = QLabel("Select User:")
        self.user_input = QComboBox()
        self.user_input.addItems(self.users)
        left_layout.addWidget(self.user_label)
        left_layout.addWidget(self.user_input)

        self.add_user_button = QPushButton("Add User")
        self.add_user_button.clicked.connect(self.add_user)
        left_layout.addWidget(self.add_user_button)

        self.delete_user_button = QPushButton("Delete User")
        self.delete_user_button.clicked.connect(self.delete_user)
        left_layout.addWidget(self.delete_user_button)

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
        self.serial_port_input.addItems(get_serial_ports())  # 获取当前串口列表
        left_layout.addWidget(self.serial_port_label)
        left_layout.addWidget(self.serial_port_input)

        # 采样率设置
        self.sample_rate_label = QLabel("Sampling Rate:")
        self.sample_rate_input = QLineEdit("100")  # 默认采样率100 Hz
        left_layout.addWidget(self.sample_rate_label)
        left_layout.addWidget(self.sample_rate_input)

        # FIR滤波器阶数设置
        self.filter_order_label = QLabel("Filter Order:")
        self.filter_order_input = QLineEdit("5")
        left_layout.addWidget(self.filter_order_label)
        left_layout.addWidget(self.filter_order_input)

        # 添加取消滤波器按钮的功能
        self.remove_filter_button = QPushButton("Remove Filter")
        self.remove_filter_button.clicked.connect(self.remove_filter)
        left_layout.addWidget(self.remove_filter_button)

        # 应用FIR滤波器设置按钮
        self.apply_filter_button = QPushButton("Apply FIR Filter")
        self.apply_filter_button.clicked.connect(self.apply_filter_order)
        left_layout.addWidget(self.apply_filter_button)

        # 绘图控件
        self.graphWidget = PlotWidget()
        self.graphWidget.enableAutoRange(axis='y')
        self.graphWidget.addLegend()  # 直接在图形中显示标签
        right_layout.addWidget(self.graphWidget)


        # 初始化绘图对象
        for key, color in self.colors.items():
            self.plots[key] = self.graphWidget.plot(np.arange(self.max_len), np.zeros(self.max_len),
                                                    pen=pg.mkPen(color, width=2), name=key)

        # IMU复选框，控制是否绘制，三个复选框放在一行
        self.checkboxes = {}
        imu_keys = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']

        for i in range(0, len(imu_keys), 3):
            row_layout = QHBoxLayout()  # 每行三个复选框
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

        # 删除文件按钮
        self.delete_button = QPushButton("Delete Selected File")
        self.delete_button.clicked.connect(self.delete_file)
        left_layout.addWidget(self.delete_button)

        # 采集数据按钮
        self.start_collect_button = QPushButton("Start Collecting Data")
        self.start_collect_button.clicked.connect(self.start_collecting_data)
        left_layout.addWidget(self.start_collect_button)

        self.stop_collect_button = QPushButton("Stop Collecting Data")
        self.stop_collect_button.clicked.connect(self.stop_collecting_data)
        left_layout.addWidget(self.stop_collect_button)

        # 添加呼吸速率显示
        self.realtime_breathing_label = QLabel("Real-time Breathing Rate: 0 bpm")
        self.average_breathing_label = QLabel("60s Average Breathing Rate: 0 bpm")

        # 将实时和平均呼吸速率标签加入右侧布局
        right_layout.addWidget(self.realtime_breathing_label)
        right_layout.addWidget(self.average_breathing_label)

        # 将左栏和右栏分别加入主布局
        main_layout.addLayout(left_layout)  # 左边控件
        main_layout.addLayout(right_layout)  # 右边控件

        # 设置窗口中心控件
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def refresh_serial_ports(self):
        """刷新串口列表"""
        current_ports = get_serial_ports()
        if current_ports != self.serial_ports:  # 检查是否有新的串口设备
            self.serial_ports = current_ports
            self.serial_port_input.clear()
            self.serial_port_input.addItems(self.serial_ports)
            print("Serial ports updated:", self.serial_ports)
    def load_files(self):
        """加载目录中的数据文件"""
        files = os.listdir('./data')
        self.file_list.clear()
        self.file_list.addItems(files)

    def delete_file(self):
        """删除选中的文件"""
        selected_file = self.file_list.currentItem().text()
        reply = QMessageBox.question(self, 'Confirm', f"Are you sure you want to delete {selected_file}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            os.remove(f'./data/{selected_file}')
            self.load_files()

    def add_user(self):
        """添加用户"""
        new_user, ok = QInputDialog.getText(self, 'Add User', 'Enter new user name:')
        if ok and new_user:
            self.users.append(new_user)
            self.user_input.addItem(new_user)
            print(f"Added user: {new_user}")

    def delete_user(self):
        """删除用户"""
        user_to_remove = self.user_input.currentText()
        reply = QMessageBox.question(self, 'Confirm', f"Are you sure you want to delete {user_to_remove}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.users.remove(user_to_remove)
            self.user_input.removeItem(self.user_input.currentIndex())
            print(f"Deleted user: {user_to_remove}")

    def apply_filter_order(self):
        """应用滤波器"""
        try:
            # 获取用户输入的滤波器阶数和采样率
            order = int(self.filter_order_input.text())
            sample_rate = int(self.sample_rate_input.text())

            # 设计FIR滤波器
            self.filter_taps = design_fir_filter(0.5, sample_rate, order)
            self.use_filter = True  # 设置为 True，表示正在使用滤波器
            print(f"Filter order set to {order} with sample rate {sample_rate} Hz")

            # 如果已经在采集数据，重启进程以应用新滤波器
            if self.serial_process and self.serial_process.is_alive():
                self.running_flag.value = 0  # 停止当前进程
                self.serial_process.join()

                # 重新启动进程，并应用新的滤波器
                self.running_flag.value = 1
                self.serial_process = Process(target=read_serial_data, args=(
                    self.serial_port, 115200, self.data_queue, self.filter_taps, self.running_flag, self.file_path))
                self.serial_process.start()
                print("Applied new filter and restarted data collection.")

        except ValueError:
            print("Invalid filter order or sample rate")

    # 取消滤波器的逻辑
    def remove_filter(self):
        """取消滤波器，显示原始数据"""
        self.use_filter = False  # 取消滤波器
        print("Filter removed. Showing raw data.")

        # 如果采集正在进行，重启子进程以显示原始数据
        if self.serial_process and self.serial_process.is_alive():
            self.running_flag.value = 0  # 停止当前进程
            self.serial_process.join()

            # 重启子进程，不应用滤波器
            self.running_flag.value = 1
            self.serial_process = Process(target=read_serial_data, args=(
                self.serial_port, 115200, self.data_queue, None, self.running_flag, self.file_path))  # 不传递滤波器系数
            self.serial_process.start()
            print("Restarted data collection without filter.")

    def start_collecting_data(self):
        """开始采集数据"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        motion_type = self.motion_type_input.currentText()
        breathing_type = self.breathing_type_input.currentText()
        user = self.user_input.currentText()
        chip = self.chip_input.currentText()
        sample_rate = self.sample_rate_input.text()
        filter_order = self.filter_order_input.text()  # 获取滤波器阶数

        # 根据是否应用了滤波器来命名文件
        if self.use_filter:
            file_name = f"{user}_{chip}_{motion_type}_{breathing_type}_{sample_rate}Hz_FilterOrder{filter_order}_{timestamp}.csv"
        else:
            file_name = f"{user}_{chip}_{motion_type}_{breathing_type}_{sample_rate}Hz_raw_data_{timestamp}.csv"

        self.file_path = f"./data/{file_name}"

        self.serial_port = self.serial_port_input.currentText()
        self.running_flag.value = 1
        self.start_time = time.time()  # 记录采样开始时间
        # 传递 self.use_filter 作为参数
        self.serial_process = Process(target=read_serial_data, args=(
            self.serial_port, 115200, self.data_queue, self.filter_taps, self.running_flag, self.file_path,
            self.use_filter))
        self.serial_process.start()
        print(f"Started collecting data: {file_name}")

    def stop_collecting_data(self):
        """停止采集数据，并提示是否保存"""
        end_time = time.time()
        sampling_duration = round(end_time - self.start_time, 2)  # 计算采样时长，保留两位小数

        reply = QMessageBox.question(self, 'Save Data',
                                     f"Sampling duration: {sampling_duration}s. Do you want to save this data?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.running_flag.value = 0
            self.serial_process.join()  # 等待进程结束
            print(f"Data collection stopped and data saved. Sampling time: {sampling_duration}s")
        else:
            # 如果选择不保存，可以删除文件
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            print("Data collection stopped and data discarded.")

        self.load_files()

    def update_plot_visibility(self):
        """更新绘图的可见性"""
        for key in self.checkboxes:
            if self.checkboxes[key].isChecked():
                self.plots[key].setVisible(True)
            else:
                # 隐藏并清空数据
                self.plots[key].setVisible(False)
                self.x_vals[key].clear()  # 清除对应数据
                # 设置为空数据以防止错误
                self.plots[key].setData([], [])

    # 计算呼吸速率
    def calculate_breathing_rate(self, signal_data, sample_rate):
        # 对数据进行快速傅里叶变换（FFT）
        fft_result = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), d=1 / sample_rate)

        # 只考虑正频率分量
        positive_freqs = freqs[freqs > 0]
        positive_fft = np.abs(fft_result[freqs > 0])

        # 筛选0.05Hz-0.5Hz之间的频率范围
        valid_idx = np.where((positive_freqs >= 0.05) & (positive_freqs <= 0.05))
        valid_freqs = positive_freqs[valid_idx]
        valid_fft = positive_fft[valid_idx]

        # 找到峰值频率对应的呼吸频率
        dominant_freq = valid_freqs[np.argmax(valid_fft)]
        breathing_rate = dominant_freq * 60  # 转换为每分钟呼吸率
        return breathing_rate

    # 卡尔曼滤波器类
    class KalmanFilter:
        def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
            self.process_variance = process_variance
            self.measurement_variance = measurement_variance
            self.estimated_measurement_variance = estimated_measurement_variance
            self.posteri_estimate = 0.0
            self.posteri_error_estimate = 1.0

        def update(self, measurement):
            priori_estimate = self.posteri_estimate
            priori_error_estimate = self.posteri_error_estimate + self.process_variance

            blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
            self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
            self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

            return self.posteri_estimate

    def fuse_data(self, x, y, z):        # 使用平方和方法进行简单融合
        return np.sqrt(x ** 2)

    def update_plot(self):
        while not self.data_queue.empty():
            # 从队列中获取数据
            data = self.data_queue.get()
            x_acc, y_acc, z_acc = data[:3]  # 只取加速度数据

            # 融合数据
            fused_signal = self.fuse_data(x_acc, y_acc, z_acc)

            # 使用卡尔曼滤波器平滑数据
            # filtered_signal = self.kalman_filter.update(fused_signal)
            fused_signal = self.kalman_filter.update(fused_signal)

            # 将信号添加到缓存中，确保缓存只保存过去60秒的数据
            # self.signal_buffer.append(filtered_signal)
            self.signal_buffer.append(fused_signal)

            # 计算实时呼吸率，每100个点更新一次
            if len(self.signal_buffer) >= self.min_signal_len:  # 确保足够的数据长度
                if len(self.signal_buffer) % 100 == 0:  # 每进入100个数据点时更新
                    current_breathing_rate = self.calculate_breathing_rate(list(self.signal_buffer), sample_rate=100)
                    self.realtime_breathing_label.setText(f"Real-time Breathing Rate: {current_breathing_rate:.2f} bpm")

                    # 呼吸率提醒
                    if current_breathing_rate < 5:
                        print(f"Warning: Breathing too slow! Current rate: {current_breathing_rate:.2f} BPM")
                    elif current_breathing_rate > 10:
                        print(f"Warning: Breathing too fast! Current rate: {current_breathing_rate:.2f} BPM")

            # 计算过去60秒的平均呼吸率
            if len(self.signal_buffer) >= 60 * 100:  # 假设采样率为100Hz
                average_breathing_rate = self.calculate_breathing_rate(list(self.signal_buffer), sample_rate=100)
                self.average_breathing_label.setText(f"60s Average Breathing Rate: {average_breathing_rate:.2f} bpm")

            # 更新绘图（保持已有的绘图逻辑）
            keys = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']
            for i, key in enumerate(keys):
                if self.checkboxes[key].isChecked():  # 根据复选框状态更新绘图
                    self.x_vals[key].append(data[i])
                    x_range = np.arange(len(self.x_vals[key]))
                    self.plots[key].setData(x_range, list(self.x_vals[key]))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 检查并创建数据存储文件夹
    if not os.path.exists('./data'):
        os.makedirs('./data')

    # 进程间通信队列和控制标志
    data_queue = Queue()
    running_flag = Value('i', 1)

    # 创建并启动主窗口
    window = IMUPlotter(data_queue)
    window.filter_taps = design_fir_filter(0.5, 100, 21)  # 初始滤波器设置
    window.running_flag = running_flag
    window.show()

    sys.exit(app.exec_())
