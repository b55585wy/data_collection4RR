import socket
from data_processing import detect_breathing_phase_by_derivative

# 指定服务器的IP地址和端口号
SERVER_HOST = '192.168.1.100'  # 监听所有可用的网络接口
SERVER_PORT = 7897  # 指定要监听的端口号

# 创建socket对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定服务器地址和端口
server_socket.bind((SERVER_HOST, SERVER_PORT))

# 开始监听连接请求
server_socket.listen(1)
print(f"Server is listening on {SERVER_HOST}:{SERVER_PORT}")

while True:
    # 等待客户端连接
    client_socket, address = server_socket.accept()
    print(f"Connected to client: {address}")

    # 接收客户端发送的数据
    data = client_socket.recv(1024).decode('utf-8')
    print(f"Received data from client: {data}")

    # 判断呼吸阶段并发送相应的指令给客户端
    if data == "request_phase":
        # 获取当前的呼吸阶段
        current_phase, phase_completion = detect_breathing_phase_by_derivative(list(window.signal_buffer))

        # 向客户端发送呼吸阶段信息
        response = f"{current_phase},{phase_completion:.2f}"
        client_socket.send(response.encode('utf-8'))
        print(f"Sent phase to client: {response}")
    else:
        # 处理其他请求或错误情况
        response = "Unknown request"
        client_socket.send(response.encode('utf-8'))

    # 关闭客户端连接
    client_socket.close()