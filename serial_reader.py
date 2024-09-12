import socket
import time

import serial
import serial.tools.list_ports
from collections import deque

from data_processing import apply_fir_filter, detect_breathing_phase_by_derivative, twos_complement_and_scale

def save_data_to_file(file_path, data):
    with open(file_path, 'a') as f:
        for d in data:
            f.write(','.join(map(str, d)) + '\n')




# 子进程中负责读取串口数据和滤波处理
def read_serial_data(serial_port, baud_rate, data_queue, filter_taps, running_flag, file_path, use_filter):
    try:
        ser = serial.Serial(serial_port, baud_rate)
        print(f"Connected to {serial_port} at {baud_rate} baud.")
        # 创建 socket 连接
        # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 使用 UDP
        # sock.connect(('192.168.1.100', 7897))  # 连接到下位机的 IP 和端口

    except serial.SerialException as e:
        print(f"Failed to connect to {serial_port}: {e}")
        return

    acc_range = 1  # Accelerometer range in g
    gyro_range = 1  # Gyroscope range in degrees/sec
    data_list = []  # 保存收集的数据
    start_time = time.time()  # 记录开始时间
    filtered_data_list = []
    # Initialize buffers for accumulating data for filtering and phase detection
    acc_buffer = {key: deque(maxlen=500) for key in {'x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro'}}
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
            flag = 1
            # Accumulate data in buffers for filtering
            if use_filter:
                acc_buffer['x_acc'].append(x_acc)
                acc_buffer['y_acc'].append(y_acc)
                acc_buffer['z_acc'].append(z_acc)
                acc_buffer['x_gyro'].append(x_gyro)
                acc_buffer['y_gyro'].append(y_gyro)
                acc_buffer['z_gyro'].append(z_gyro)
                # 如果缓冲区有maxlen代表已经稳定，可以进行滤波。
                if len(acc_buffer['x_acc']) == acc_buffer['x_acc'].maxlen and flag == 0:
                    # acc_buffer滤波历史数据已经稳定，可以进行滤波，
                    x_acc_filtered = apply_fir_filter(acc_buffer['x_acc'], filter_taps)
                    y_acc_filtered = apply_fir_filter(acc_buffer['y_acc'], filter_taps)
                    z_acc_filtered = apply_fir_filter(acc_buffer['z_acc'], filter_taps)
                    x_gyro_filtered = apply_fir_filter(acc_buffer['x_gyro'], filter_taps)
                    y_gyro_filtered = apply_fir_filter(acc_buffer['y_gyro'], filter_taps)
                    z_gyro_filtered = apply_fir_filter(acc_buffer['z_gyro'], filter_taps)

                    # Append the filtered data
                    filtered_data_list[0].append(x_acc_filtered[-1])
                    filtered_data_list[1].append(y_acc_filtered[-1])
                    filtered_data_list[2].append(z_acc_filtered[-1])
                    filtered_data_list[3].append(x_gyro_filtered[-1])
                    filtered_data_list[4].append(y_gyro_filtered[-1])
                    filtered_data_list[5].append(z_gyro_filtered[-1])
                    print(filtered_data_list[0][-100:])
                    # cal_breathing_phases(filtered_data[0][-100:])
                    # 实时呼吸阶段检测
                    
                    current_phase, phase_completion = detect_breathing_phase_by_derivative(list(filtered_data_list[0]))
                    # sock.sendall(f"{current_phase}\n".encode('utf-8'))  # 发送 current_phase
                    print(f"Current phase: {current_phase}, Phase completion: {phase_completion:.2f}")
                elif  len(acc_buffer['x_acc']) == acc_buffer['x_acc'].maxlen and flag == 1:
                    flag = 0
                    x_acc_filtered = apply_fir_filter(acc_buffer['x_acc'], filter_taps)
                    y_acc_filtered = apply_fir_filter(acc_buffer['y_acc'], filter_taps)
                    z_acc_filtered = apply_fir_filter(acc_buffer['z_acc'], filter_taps)
                    x_gyro_filtered = apply_fir_filter(acc_buffer['x_gyro'], filter_taps)
                    y_gyro_filtered = apply_fir_filter(acc_buffer['y_gyro'], filter_taps)
                    z_gyro_filtered = apply_fir_filter(acc_buffer['z_gyro'], filter_taps)
                    # Append the filtered data
                    filtered_data_list = [x_acc_filtered, y_acc_filtered, z_acc_filtered,
                                     x_gyro_filtered, y_gyro_filtered, z_gyro_filtered]
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