import sys
import serial
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QLineEdit
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from collections import deque
from PyQt5.QtCore import QTimer
from multiprocessing import Process, Queue, Value
from scipy import signal
import neurokit2 as nk
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
# Helper function to convert two's complement and scale
def twos_complement_and_scale(raw_data, range_val):
    if raw_data & 0x8000:
        raw_data = -((~raw_data + 1) & 0xFFFF)
    return (float(raw_data) / 32768.0) * range_val

# FIR 低通滤波器设计
def design_fir_filter(cutoff_freq, sample_rate, num_taps):
    nyquist = 0.5 * sample_rate  # 奈奎斯特频率
    normalized_cutoff = cutoff_freq / nyquist
    taps = signal.firwin(num_taps, normalized_cutoff, window='hamming')
    return taps

# 应用FIR滤波器
def apply_fir_filter(data, taps):
    return signal.lfilter(taps, 1.0, data)

# 子进程中负责读取串口数据和滤波处理
# 子进程中负责读取串口数据和滤波处理
def read_serial_data(serial_port, baud_rate, data_queue, filter_taps, running_flag):
    try:
        ser = serial.Serial(serial_port, baud_rate)
        print(f"Connected to {serial_port} at {baud_rate} baud.")
    except serial.SerialException as e:
        print(f"Failed to connect to {serial_port}: {e}")
        return

    acc_range = 2  # Accelerometer range in g
    x_acc_data = deque([0] * 100, maxlen=100)  # Store last 100 X-axis accelerometer data
    z_acc_data = deque([0] * 100, maxlen=100)  # Store last 100 Z-axis accelerometer data

    last_x_filtered = 0  # 用于存储最后一次成功的x轴滤波数据
    last_z_filtered = 0  # 用于存储最后一次成功的z轴滤波数据

    while running_flag.value:  # running_flag 控制进程运行与停止
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                print(f"Raw data from serial: {line}")
                values = line.split(',')
                if len(values) >= 3:
                    # 读取IMU数据并进行二进制补码转换
                    raw_x_acc = int(values[0])
                    raw_z_acc = int(values[2])

                    # 转换和缩放加速度数据
                    x_acc = twos_complement_and_scale(raw_x_acc, acc_range)
                    z_acc = twos_complement_and_scale(raw_z_acc, acc_range)

                    print(f"Converted x_acc: {x_acc}, z_acc: {z_acc}")

                    # 储存数据用于滤波
                    x_acc_data.append(x_acc)
                    z_acc_data.append(z_acc)

                    # 应用FIR低通滤波器
                    x_filtered = apply_fir_filter(list(x_acc_data), filter_taps)
                    z_filtered = apply_fir_filter(list(z_acc_data), filter_taps)

                    last_x_filtered = x_filtered[-1]
                    last_z_filtered = z_filtered[-1]

                    print(f"Filtered x: {x_filtered[-1]}, Filtered z: {z_filtered[-1]}")
                    try:
                        # 使用NeuroKit2进行呼吸信号的处理
                        rsp_signals_x = nk.rsp_process(x_filtered, sampling_rate=100)
                        rsp_signals_z = nk.rsp_process(z_filtered, sampling_rate=100)

                        # 获取呼吸阶段（吸气/呼气）和幅度信息
                        rsp_phase_x = rsp_signals_x[0]['RSP_Phase'][-1]  # 当前呼吸阶段
                        amplitude_x = rsp_signals_x[0]['RSP_Amplitude'][-1]  # 当前吸气幅度

                        rsp_phase_z = rsp_signals_z[0]['RSP_Phase'][-1]  # 当前呼吸阶段
                        amplitude_z = rsp_signals_z[0]['RSP_Amplitude'][-1]  # 当前吸气幅度

                    except Exception as e:
                            # logging.warning(f"NeuroKit2 failed to process signal: {e}")
                            # 如果NeuroKit2处理失败，仍然使用上一个有效的数据
                            rsp_phase_x = "unknown"
                            amplitude_x = 0
                            rsp_phase_z = "unknown"
                            amplitude_z = 0
                            pass

                    # 无论是否发生错误，都将数据发送到主进程进行绘图
                    data_queue.put((last_x_filtered, last_z_filtered, rsp_phase_x, amplitude_x, rsp_phase_z, amplitude_z))

            except ValueError:
                print(f"Invalid data received: {line}")

class IMUPlotter(QMainWindow):
    def __init__(self, data_queue, max_len=100):
        super().__init__()
        self.data_queue = data_queue
        self.max_len = max_len
        self.x_vals = deque([0] * max_len, maxlen=max_len)
        self.z_vals = deque([0] * max_len, maxlen=max_len)

        # 设置界面布局
        self.init_ui()

        # 定时器用于更新图形界面和呼吸信息
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 每50毫秒更新一次绘图

    def init_ui(self):
        layout = QVBoxLayout()

        # 绘图
        self.graphWidget = PlotWidget()
        self.graphWidget.enableAutoRange(axis='y')
        self.graphWidget.addLegend()
        self.line_x = self.graphWidget.plot(np.arange(self.max_len), np.zeros(self.max_len), pen=pg.mkPen('r', width=2), name="X-Axis")
        self.line_z = self.graphWidget.plot(np.arange(self.max_len), np.zeros(self.max_len), pen=pg.mkPen('b', width=2), name="Z-Axis")
        layout.addWidget(self.graphWidget)

        # 控制界面部分
        self.filter_order_label = QLabel("Filter Order:")
        self.filter_order_input = QLineEdit("5")
        self.filter_order_button = QPushButton("Apply Filter Order")
        self.filter_order_button.clicked.connect(self.apply_filter_order)
        layout.addWidget(self.filter_order_label)
        layout.addWidget(self.filter_order_input)
        layout.addWidget(self.filter_order_button)

        # 呼吸信息显示
        self.breathing_info_label = QLabel("Breathing Info: ")
        self.breathing_info = QLabel("Phase: --, Amplitude: --")
        layout.addWidget(self.breathing_info_label)
        layout.addWidget(self.breathing_info)

        # 串口启停控制按钮
        self.start_button = QPushButton("Start Serial")
        self.stop_button = QPushButton("Stop Serial")
        self.start_button.clicked.connect(self.start_serial)
        self.stop_button.clicked.connect(self.stop_serial)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        # 设置窗口中心控件
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def apply_filter_order(self):
        try:
            order = int(self.filter_order_input.text())
            self.filter_taps = design_fir_filter(0.5, 100, order)
            print(f"Filter order set to {order}")
        except ValueError:
            print("Invalid filter order")

    def start_serial(self):
        self.running_flag.value = 1
        self.serial_process = Process(target=read_serial_data, args=(self.serial_port, 115200, self.data_queue, self.filter_taps, self.running_flag))
        self.serial_process.start()
        print("Serial started")

    def stop_serial(self):
        self.running_flag.value = 0
        self.serial_process.join()  # 等待进程结束
        print("Serial stopped")

    def update_plot(self):
        while not self.data_queue.empty():
            # 从队列中获取数据
            x, z, phase_x, amplitude_x, phase_z, amplitude_z = self.data_queue.get()

            # 更新绘图数据
            self.x_vals.append(x)
            self.z_vals.append(z)
            self.line_x.setData(np.arange(self.max_len), list(self.x_vals))
            self.line_z.setData(np.arange(self.max_len), list(self.z_vals))

            # 更新呼吸阶段和幅度信息显示
            self.breathing_info.setText(
                f"X: Phase = {phase_x}, Amplitude = {amplitude_x:.2f}, Z: Phase = {phase_z}, Amplitude = {amplitude_z:.2f}")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 进程间通信队列和控制标志
    data_queue = Queue()
    running_flag = Value('i', 1)

    # 创建并启动主窗口
    window = IMUPlotter(data_queue)
    window.serial_port = 'COM4'  # 设置串口号
    window.filter_taps = design_fir_filter(0.5, 100, 5)  # 初始滤波器设置
    window.running_flag = running_flag
    window.show()

    sys.exit(app.exec_())
