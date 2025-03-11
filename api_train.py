import os
import re
import json 
import yaml
import sys
import psutil
from pathlib import Path
from flask import Flask, request, jsonify
from threading import Lock
from multiprocessing import Process


from utils.datasets import IMG_FORMATS, VID_FORMATS
import train as v5_train


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

app = Flask(__name__)

# 定义常量
PORT = 9007                                                     # 端口号
TRAIN_PROCESS_ID = -1                                           # 训练进程的pid
train_save_data = ROOT / "train_save_data"                      # 训练数据集保存路径
epochs = 300                                                    # 训练轮数
base_project_name = ''                                          # 项目名称  
net_cfg = {
    "yolov5s": {
        "weights": ROOT / "weights/yolov5s.pt",
        "cfg": ROOT / "models/yolov5s.yaml",
        "hyp": ROOT / "data/hyps/hyp.scratch-low.yaml",
    },
    "yolov5m": {
            "weights": ROOT / "weights/yolov5m.pt",
            "cfg": ROOT / "models/yolov5m.yaml",
            "hyp": ROOT / "data/hyps/hyp.scratch-med.yaml",
        },
    "yolov5l": {
            "weights": ROOT / "weights/yolov5l.pt",
            "cfg": ROOT / "models/yolov5l.yaml",
            "hyp": ROOT / "data/hyps/hyp.scratch-high.yaml",
        },
    "yolov5x": {
            "weights": ROOT / "weights/yolov5x.pt",
            "cfg": ROOT / "models/yolov5x.yaml",
            "hyp": ROOT / "data/hyps/hyp.scratch-med.yaml",
        },
}

@app.route('/api/Train', methods=['POST'])
def train():

    global TRAIN_PROCESS_ID,epochs,base_project_name,net_cfg

    if psutil.pid_exists(TRAIN_PROCESS_ID):
        return jsonify({"CODE": 400, "ERROR": "训练任务已存在"}), 400

    params = request.form if request.form else request.json

    # 判断参数是否为空
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    # 判断参数是否完整
    train_required_keys = [ 'train_project_name']
    train_missing_keys = [key for key in train_required_keys if key not in params]
    if train_missing_keys:
        return jsonify({"CODE": 400, "ERROR": f"缺少参数: {train_missing_keys}"}), 400

    # 获取参数
    base_project_name = params.get('train_project_name')    # 项目名称
    modeltype  = params.get('model_type', 'yolov5m')         # yolov5s, yolov5m, yolov5l, yolov5x
    epochs = int(params.get('epochs', 300))                 # 训练轮数
    batch_size = int(params.get('batch_size', -1))          # 批次大小
    imgsz = int(params.get('imgsz', 640))                   # 图片大小
    resume = (params.get('resume', False))                  # 是否继续训练

    #处理base_project_name,如果存在历史信息中,则直接使用,如果不存在,则返回用户信息<请前往我的训练集新建项目>
    train_history = [f for f in os.listdir(train_save_data) if os.path.isdir(os.path.join(train_save_data, f))]
    if base_project_name not in train_history:
        return jsonify({"CODE": 400, "ERROR": "请前往<我的训练集>新建项目"}), 400
    else:
        data = train_save_data / base_project_name / f'{base_project_name}.yaml'

    #处理modeltype,在net_cfg中选择对应的模型的cfg,weights,hyp
    if modeltype not in net_cfg:
        return jsonify({"CODE": 400, "ERROR": "modeltype错误"}), 400
    else:
        cfg = net_cfg[modeltype]['cfg']
        weights = net_cfg[modeltype]['weights']
        hyp = net_cfg[modeltype]['hyp']

    #处理epochs
    if epochs <= 1:
        return jsonify({"CODE": 400, "ERROR": "epochs错误,训练轮数必须大于1"}), 400

    #处理batch_size,batch_size必须是2的倍数,且batch_size 可以为-1,表示自动选择
    if batch_size != -1 and (batch_size < 2 or batch_size % 2 != 0):
        return jsonify({"CODE": 400, "ERROR": "batch_size错误,必须是2的倍数"}), 400

    #处理imgsz,如果imgsz小于32,则返回错误,且imgsz必须是32的倍数
    if imgsz < 32 or imgsz % 32 != 0:
        return jsonify({"CODE": 400, "ERROR": "imgsz错误,必须是32的倍数"}), 400

    #处理resume,如果开启resume,则去train_save_data下查找是否存在对应的模型,如果存在,则继续训练,否则返回错误
    resume = True if resume == 'True' else False
    if resume:
        last_model = train_save_data / base_project_name / 'weights' / 'last.pt'
        resume = Path(last_model)
        if not os.path.exists(last_model):
            return jsonify({"CODE": 400, "ERROR": "resume错误,未找到对应的模型,或是第一次训练请关闭resume参数"}), 400

    #updata opt
    opt = v5_train.parse_opt()
    opt.cfg = str(cfg)
    opt.weights = str(weights)
    opt.hyp = str(hyp)
    opt.data = str(data)
    opt.batch_size = batch_size
    opt.img_size = imgsz
    opt.epochs = epochs
    opt.resume = resume
    opt.image_weights = True
    opt.project = train_save_data
    opt.name = base_project_name
    opt.exist_ok = True

    # print("YOLOv5(/v5train)[PARAM]", opt)
    #开启一个子线程进行训练，记录pid,后面可以通过pid来判断是否有训练任务
    try:
        print("start train")
        process = Process(target=v5_train_task, args=(opt,))
        process.start()
        TRAIN_PROCESS_ID = process.pid

        #开始训练更新对应项目的state.json状态码
        with open(train_save_data / base_project_name / 'state.json', 'w') as f:
            # f.write({"MESSAGE": "training","STATE": 1}) # 1表示训练中,2表示训练暂停,3表示训练停止,4表示训练完成
            json.dump({"MESSAGE": "training","STATE": 1}, f)

        # process.join()   #等待子线程结束

        # #完成训练更新对应项目的state.json状态码
        # with open(train_save_data / base_project_name / 'state.json', 'w') as f:
        #     f.write({"MESSAGE": "finsh","STATE": 4}) # 1表示训练中,2表示训练暂停,3表示训练停止,4表示训练完成

        # weights_path = train_save_data / base_project_name / 'weights' / 'last.pt'

        # return jsonify({"CODE": "200", "INFO": "训练成功","weights_path":str(weights_path)}), 200

        return jsonify({"CODE": 200, "INFO": "训练成功"}), 200
    except Exception as e:
        return jsonify({"CODE": 400, "INFO": "训练失败","ERROR": f"训练失败: {e}"}), 400


@app.route('/api/StopTrain', methods=['POST'])
def stoptrain():

    global TRAIN_PROCESS_ID,base_project_name

    params = request.form if request.form else request.json

    # 判断参数是否为空
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    # 判断参数是否完整
    stop_train_required_keys = [ 'train_action','trains_tate']
    stop_train_missing_keys = [key for key in stop_train_required_keys if key not in params]
    if stop_train_missing_keys:
        return jsonify({"CODE": 400, "ERROR": f"缺少参数: {stop_train_missing_keys}"}), 400
    
    trainactin = params.get('train_action')     # 是否停止训练
    trainState = params.get('train_state')      # 训练状态
    trainactin = True if trainactin == 'True' else False
    if trainactin:
        if psutil.pid_exists(TRAIN_PROCESS_ID):
            p = psutil.Process(TRAIN_PROCESS_ID)
            p.terminate()
            #停止训练或者暂停训练更新对应项目的state.json状态码
            with open(train_save_data / base_project_name / 'state.json', 'w') as f:
                json.dump({"MESSAGE": "stop","STATE": trainState}, f)

            return jsonify({"CODE": 200, "MESSAGE": "停止训练成功"}), 200
        else:
            return jsonify({"CODE": 400, "ERROR": "训练任务不存在"}), 400


@app.route('/api/TrainInfo', methods=['post'])
def trainstatus():
    """
    TODO: 读取训练目录下的state.json文件,返回训练状态
    TODO: 读取训练目录下的results_log.txt文件,返回训练日志,返回map值,返回训练时间
    epoch 53/299  p:0.855317  r:0.816225  mAP:0.858824 time 0.043 hours.下的时间*epoch的轮数,返回一个大概训练的时间
    """
    global TRAIN_PROCESS_ID

    params = request.form if request.form else request.json

    # 判断参数是否为空
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400
        
    # 判断参数是否完整
    train_info_required_keys = [ 'train_project_name']
    train_info_missing_keys = [key for key in train_info_required_keys if key not in params]
    if train_info_missing_keys:
        return jsonify({"CODE": 400, "ERROR": f"缺少参数: {train_info_missing_keys}"}), 400

    state_train_project_name = params.get('train_project_name')    # 项目名称

    #处理base_project_name,如果存在历史信息中,则直接使用,如果不存在,则返回用户信息<请前往我的训练集新建项目>
    train_history = [f for f in os.listdir(train_save_data) if os.path.isdir(os.path.join(train_save_data, f))]
    if state_train_project_name not in train_history:
        return jsonify({"CODE": 400, "ERROR": "请前往<我的训练集>新建项目"}), 400
    else:
        state_data_path = train_save_data / state_train_project_name / 'state.json'
        train_log_path = train_save_data / state_train_project_name / 'train_log.txt'
    
    if not os.path.exists(state_data_path):
        return jsonify({"CODE": 400, "ERROR": "state.json文件不存在,检查有无权限创建文件"}), 400
    
    # if not os.path.exists(train_log_path):
    #     return jsonify({"CODE": 400, "ERROR": "train_log.txt文件不存在,检查有无权限创建文件"}), 400

    #读取state.json文件,返回训练状态
    try:
        with open(state_data_path, 'r') as f:
            statejson = json.load(f)
            state = statejson['STATE']
    except Exception as e:
        return jsonify({"CODE": 400, "ERROR": f"读取state.json文件失败: {e}"}), 400

    if not os.path.exists(train_log_path):
        train_data_size = train_save_data / state_train_project_name / 'images'
        Approximate_time =  round(os.path.getsize(train_data_size) / 1024 / 1024 / 10, 2)*epochs
        return jsonify({"CODE": 200, "MESSAGE": "获取训练信息成功","state": state, "Approximate_time": str(Approximate_time)+'hours'}), 200
    else:
        #读取results_log.txt文件,计算大概训练时间,返回map值
        try:
            with open(train_log_path, 'r') as f:
                #必须要训练一轮之后才能读取到时间,否则给用户返回一个根据文件大小计算的时间
                lines = f.readlines()
                if len(lines) == 0  :
                    train_data_size = train_save_data / state_train_project_name / 'images'
                    Approximate_time =  round(os.path.getsize(train_data_size) / 1024 / 1024 / 10, 2)*epochs
                    return jsonify({"CODE": 200, "MESSAGE": "获取训练信息成功","STATE": state, "Approximate_time": str(Approximate_time)+'hours'}), 200
                else:
                    pattern = re.compile(r'time(.*?)hour.')
                    for line in lines:
                        match = pattern.search(line)
                        if match:
                            time = float(match.group(1))
                            return jsonify({"CODE": 200, "MESSAGE": "获取训练信息成功","state": state, "Approximate_time": str(round(time*epochs,2))+'hours'}), 200
                        else:
                            return jsonify({"CODE": 400, "ERROR": "re失败"}), 400
        except Exception as e:
            return jsonify({"CODE": 400, "ERROR": f"读取results_log.txt文件失败: {e}"}), 400
    
 

def v5_train_task(opt):
    v5_train.main(opt)
    print("train finished")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=PORT)
