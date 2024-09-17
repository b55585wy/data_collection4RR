import socket
import sys
import time

host = 'localhost'#指向服务器的ip地址，因为是做测试，所以指向本机
port = 12242
file_path = 'data/User1_C3_Standing_Deep Breathing_100Hz_FilterOrder5_20240912-205617.csv'
# file_path = ''
def send_data_from_file(host, port, file_path, max_retries=5):
    for attempt in range(max_retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                print(f"Connected to {host}:{port}")

                with open(file_path, 'r') as file:
                    for line in file:
                        try:
                            s.sendall(line.strip().encode())
                            time.sleep(0.05)  # 增加延迟时间
                        except BrokenPipeError:
                            print("Broken pipe error. Connection lost.")
                            break

                print("Finished sending data")
            break  # 如果成功连接并发送数据，跳出循环
        except ConnectionRefusedError:
            print(f"Connection refused. Retrying in 2 seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
    else:
        print("Failed to connect after multiple attempts")

if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python tcp_client.py <host> <port> <file_path>")
    #     sys.exit(1)

    # host = sys.argv[1]
    # port = int(sys.argv[2])
    # file_path = sys.argv[3]

    send_data_from_file(host, port, file_path)
