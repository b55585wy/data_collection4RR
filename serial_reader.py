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
def read_serial_data(data_queue, running_flag, port, baud_rate, sampling_duration):
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)  # 添加1秒超时
        print(f"Connected to {port} at {baud_rate} baud.")

        while running_flag.value:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line:  # 确保读取到了数据
                        # 处理数据...
                        pass  # 添加这行作为占位符
                    else:
                        print("Received empty line, skipping...")
                else:
                    time.sleep(0.1)  # 如果没有数据，短暂休眠以避免CPU过度使用
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                time.sleep(1)  # 出错时等待1秒再重试
                continue

    except serial.SerialException as e:
        print(f"Failed to connect to {port}: {e}")
    finally:
        if 'ser' in locals():
            ser.close()
    return sampling_duration


# 获取串口设备列表
def get_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]