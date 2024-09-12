from scipy import signal
import numpy as np
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
    # 计算呼吸阶段完成度，可以根据信号的相对位置进行估计
    # 这里简单地根据导数的趋势变化来模拟呼吸的完成度
    phase_completion = np.abs(derivative[-1]) / np.max(np.abs(derivative)) if np.max(np.abs(derivative)) > 0 else 0

    derivative_bool = np.where(derivative > 0, 1, -1)
    # print(derivative)
    total = sum(derivative_bool)
    print(f'total:{total}')
    if abs(total) <= 7:
        current_phase = "None"
    else:
        # 判断当前阶段，根据符号和确定呼吸阶段
        if total > 0:
            current_phase = "Inhalation"
        else:
            current_phase = "Exhalation"
        print(f"当前呼吸阶段: {current_phase}")



    return current_phase, phase_completion

def cal_breathing_phases(signal_data):
    peaks, valleys = find_breathing_points(signal_data)
    print(f'peaks,valleys:{peaks}, {valleys}')


def find_breathing_points(signal_data):
    # 找到吸气点（峰值）
    peaks, _ = signal.find_peaks(signal_data)

    # 找到吐气点（谷值）
    valleys, _ = signal.find_peaks(-signal_data)
    return peaks, valleys
