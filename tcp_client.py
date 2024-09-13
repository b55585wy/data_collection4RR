import socket
import sys
import time

host = '0.0.0.0'#指向服务器的ip地址，因为是做测试，所以指向本机
port = 12242
# file_path = 'data/User1_C3_Standing_Deep Breathing_100Hz_FilterOrder5_20240912-205617.csv'
file_path = ''
def send_data_from_file(host, port, file_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connected to {host}:{port}")

        with open(file_path, 'r') as file:
            for line in file:
                # 发送数据
                s.sendall(line.strip().encode())
                # 等待一小段时间，模拟实时数据发送
                time.sleep(0.01)

        print("Finished sending data")

if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python tcp_client.py <host> <port> <file_path>")
    #     sys.exit(1)

    # host = sys.argv[1]
    # port = int(sys.argv[2])
    # file_path = sys.argv[3]

    send_data_from_file(host, port, file_path)
