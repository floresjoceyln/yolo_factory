import psutil
from flask import Flask, request, jsonify

app = Flask(__name__)

mode = 1
PORT = 9007


def convert_bytes(bytes_value):
    """
    将字节数转换为合适的单位（GB, MB, etc.）
    :param bytes_value: 字节数
    :return: 字符串格式的大小
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}PB"  # 如果是PB级别的数据


#查询服务器状态接口
@app.route('/api/ServerInfo', methods=['GET'])
def get_server_info():
    """
    获取服务器的CPU、内存和存储信息
    """
    global mode
    # 获取CPU使用率和核心数
    cpu_percent = psutil.cpu_percent(interval=1)  # 获取1秒钟的CPU使用率
    cpu_cores = psutil.cpu_count(logical=False)  # 获取物理核心数

    # 获取内存使用情况
    memory = psutil.virtual_memory()
    memory_used = convert_bytes(memory.used)
    memory_total = convert_bytes(memory.total)

    # 获取磁盘使用情况
    disk = psutil.disk_usage('/')
    storage_used = convert_bytes(disk.used)
    storage_total = convert_bytes(disk.total)

    # 返回服务器状态
    server_info = {
        "mode": mode,
        "cpu": f"{cpu_cores}核",
        "cpuUse": cpu_percent,  # CPU使用率
        "disk": storage_total,  # 磁盘总容量
        "diskUse": disk.percent,  # 磁盘使用率
        "memory": memory_total,  # 内存总容量
        "memoryUse": memory.percent  # 内存使用率
    }

    return jsonify({
        "CODE": 200,
        "SUCCESS": True,
        "DATA": server_info,
        "MESSAGE": "操作成功"
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=PORT)
