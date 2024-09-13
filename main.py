import sys
from PyQt5.QtWidgets import QApplication
from multiprocessing import Queue, Value
from data_processing import design_fir_filter
from serial_reader import read_serial_data
from imuPloter import IMUPlotter


if __name__ == '__main__':
    app = QApplication(sys.argv)

    data_queue = Queue()
    running_flag = Value('i', 1)

    window = IMUPlotter(data_queue)
    window.filter_taps = design_fir_filter(0.5, 100, 21)
    window.running_flag = running_flag
    window.show()
  
    
    sys.exit(app.exec_())