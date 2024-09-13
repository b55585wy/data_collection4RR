import socket
import time
from collections import deque
from data_processing import apply_fir_filter, detect_breathing_phase_by_derivative, twos_complement_and_scale

def save_data_to_file(file_path, data):
    with open(file_path, 'a') as f:
        for d in data:
            f.write(','.join(map(str, d)) + '\n')

def read_tcp_data(host, port, data_queue, filter_taps, running_flag, file_path, use_filter):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print(f"Connected to {host}:{port} over TCP.")

            acc_range = 1  # Accelerometer range in g
            gyro_range = 1  # Gyroscope range in degrees/sec
            data_list = []  # 保存收集的数据
            start_time = time.time()  # 记录开始时间
            filtered_data_list = []
            
            # Initialize buffers for accumulating data for filtering and phase detection
            acc_buffer = {key: deque(maxlen=500) for key in {'x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro'}}

            while running_flag.value:
                try:
                    # 从 TCP 连接接收数据
                    line = s.recv(1024).decode('utf-8').strip()
                    values = line.split(',')

                    # 检查数据长度是否足够
                    if len(values) != 6:
                        raise ValueError(f"Incorrect data length: {line}")

                    # 将字符串转换为整型
                    raw_x_acc, raw_y_acc, raw_z_acc, raw_x_gyro, raw_y_gyro, raw_z_gyro = map(int, values)

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
                        for key, value in zip(acc_buffer.keys(), origin_data):
                            acc_buffer[key].append(value)

                        if len(acc_buffer['x_acc']) == acc_buffer['x_acc'].maxlen:
                            # Apply FIR filtering once buffer is full
                            filtered_data = [apply_fir_filter(acc_buffer[key], filter_taps) for key in acc_buffer.keys()]
                            filtered_data_list = filtered_data

                            current_phase, phase_completion = detect_breathing_phase_by_derivative(filtered_data[0])
                            print(f"Current phase: {current_phase}, Phase completion: {phase_completion:.2f}")

                            # 发送呼吸阶段信息到下位机
                            breathing_signal = "1" if current_phase == "Expiration" else "0"
                            s.send(breathing_signal.encode())

                    data_list.append(origin_data)

                    # 每次收集1000个数据保存一次
                    if len(data_list) >= 1000:
                        save_data_to_file(file_path, data_list)
                        data_list.clear()

                    # 将原始数据发送到主进程进行绘图
                    data_queue.put(origin_data)

                except socket.error as e:
                    print(f"Connection error: {e}")
                    break

            end_time = time.time()
            sampling_duration = end_time - start_time
            print(f"Sampling duration: {sampling_duration:.2f} seconds")

            if data_list:
                save_data_to_file(file_path, data_list)
                print(f"Saved remaining data to {file_path}")

    except Exception as e:
        print(f"Failed to connect to TCP server: {e}")
    
    return sampling_duration
