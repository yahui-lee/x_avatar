import socket
import time
import torch

def generate_tensor_string():
    # 生成随机 tensor 数据
    t1 = torch.zeros(1, 3).to('cuda:0')
    t2 = torch.zeros(53, 3).to('cuda:0')  # 50 行的 rotvec 模拟
    t3 = torch.randn(10).to('cuda:0')

    tensor_strs = [
        f"tensor({repr(t1.tolist())}, device='cuda:0')",
        f"tensor({repr(t2.tolist())}, device='cuda:0')",
        f"tensor({repr(t3.tolist())}, device='cuda:0')"
    ]
    return "[" + ", ".join(tensor_strs) + "]"

# 设置服务器 IP 和端口
server_ip = '100.106.236.82'
#server_ip = '192.168.50.73'
server_port = 8082

# 初始化 UDP socket
s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#s2.bind((server_ip, server_port))  # 你是 sender 的话，其实可以不需要 bind
while True:
    try:
        data_str = generate_tensor_string()
        data = data_str.encode('utf-8')  # 编码为字节串
        t1 = time.time()
        s2.sendto(data, (server_ip, server_port))
        time.sleep(0.02)  # 发送间隔
        print("发送耗时：", time.time() - t1)
        print("发送内容：", data_str[:200] + "...")  # 只打印前 200 个字符
    except Exception as e:
        print("UDP 发送异常:", e)
