import os
import sys
from pathlib import Path
import psutil
from flask import Flask, request, jsonify
from threading import Lock
from multiprocessing import Process

from utils.datasets import IMG_FORMATS, VID_FORMATS
import detect as v5_detect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

app = Flask(__name__)

# 定义常量
PORT = 9007                                                     # 端口号
train_save_data = ROOT / "train_save_data"                      # 训练数据集保存路径
test_save_data  = ROOT / "test_save_data"                       # 训练数据集保存路径
DETECT_PROCESS_ID = -1   



@app.route('/api/Detect', methods=['POST'])
def detect():

    global DETECT_PROCESS_ID

    params = request.form if request.form else request.json

    # 判断参数是否为空
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    test_required_keys = ['train_project_name', 'test_project_name']
    test_missing_keys = [key for key in test_required_keys if key not in params]

    # 判断是否有缺失参数
    if test_missing_keys:
        misskey = ', '.join(test_missing_keys)
        return jsonify({"CODE": 400, "ERROR": f"缺失参数: {misskey}"}), 400

    detect_train_project_name = params.get('train_project_name')         # 训练项目名称
    detect_test_project_name = params.get('test_project_name')           # 测试项目名称

    # 判断训练项目是否存在,以及是否有权重文件
    if not os.path.exists(train_save_data / detect_train_project_name):
        return jsonify({"CODE": 400, "ERROR": "训练项目不存在"}), 400 
    elif not os.path.exists(train_save_data / detect_train_project_name / "weights" / "best.pt"):
        return jsonify({"CODE": 400, "ERROR": "训练项目没有权重文件,已被删除或还没有开始训练"}), 400

    # 判断测试项目是否存在
    if not os.path.exists(test_save_data / detect_test_project_name):
        return jsonify({"CODE": 400, "ERROR": "测试项目不存在"}), 400

    #获取detect的opt
    opt = v5_detect.parse_opt()
    opt.source = str(test_save_data / detect_test_project_name / "images")
    opt.weights = str(train_save_data / detect_train_project_name / "weights" / "best.pt")
    opt.project = test_save_data / detect_test_project_name
    opt.name = 'result'
    opt.agnostic_nms = True
    opt.exist_ok = True
    # print("######################################",'\n',opt)
    # return jsonify({"CODE": "200", "MESSAGE": "检测成功"}), 200
    #开启一个子进程进行检测
    try:
        process = Process(target=v5_detect_task,args=(opt, ))
        process.start()
        DETECT_PROCESS_ID = process.pid
        process.join()
        save_dir = test_save_data / detect_test_project_name / "result"
        return jsonify({"CODE": 200, "MESSAGE": "检测成功","save_dir":str(save_dir)}), 200
    except Exception as e:
        return jsonify({"CODE": 400, "ERROR": f"检测失败: {e}"}), 400


@app.route('/api/StopDetect', methods=['POST'])
def stopdetect():

    global DETECT_PROCESS_ID

    params = request.form if request.form else request.json

    # 判断参数是否为空
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    # 判断参数是否完整
    stop_test_required_keys = [ 'detectaction']
    stop_test_missing_keys = [key for key in stop_test_required_keys if key not in params]
    if stop_test_missing_keys:
        return jsonify({"CODE": 400, "ERROR": f"缺少参数: {stop_test_missing_keys}"}), 400
    
    trainactin = params.get('detectaction')     #获取是否停止检测
    trainactin = True if trainactin == 'True' else False

    if trainactin:
        if psutil.pid_exists(DETECT_PROCESS_ID):
            p = psutil.Process(DETECT_PROCESS_ID)
            p.terminate()
            return jsonify({"CODE": 200, "MESSAGE": "停止检测成功"}), 200
        else:
            return jsonify({"CODE": 400, "ERROR": "没有检测任务"}), 400


@app.route('/api/DetectInfo', methods=['post'])
def detectstatus():

    params = request.form if request.form else request.json

    # 判断参数是否为空
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400
        
    # 判断参数是否完整
    test_info_required_keys = [ 'test_project_name']
    test_info_missing_keys = [key for key in test_info_required_keys if key not in params]
    if test_info_missing_keys:
        return jsonify({"CODE": 400, "ERROR": f"缺少参数: {test_info_missing_keys}"}), 400

    state_test_project_name = params.get('test_project_name')    # 检测项目名称

    test_history = [f for f in os.listdir(test_save_data) if os.path.isdir(os.path.join(test_save_data, f))]
    if state_test_project_name not in test_history:
        return jsonify({"CODE": 400, "ERROR": "请前往<我的训练集>新建项目"}), 400
    else:
        #读取results文件夹下检出的图片数量或视频数量,在去读取images下所有的图片或视频数量,计算出检测进度,返回给用户
        result_path = test_save_data / state_test_project_name / 'result'
        images_path = test_save_data / state_test_project_name / 'images'
        if not os.path.exists(result_path):
            return jsonify({"CODE": 400, "ERROR": "检测结果文件夹不存在"}), 400
        if not os.path.exists(images_path):
            return jsonify({"CODE": 400, "ERROR": "检测图片文件夹不存在"}), 400
        result_num = len(os.listdir(result_path))
        images_num = len(os.listdir(images_path))
        progress = result_num / images_num
        return jsonify({"CODE": 200, "MESSAGE": "检测进度", "progress": progress}), 200


def v5_detect_task(opt):
    v5_detect.main(opt)
    print("detect finished")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=PORT)
