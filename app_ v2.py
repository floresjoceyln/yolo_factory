import os
import re
import sys
import json
import yaml
import psutil
import shutil
import signal
import pynvml
import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, request, jsonify
from threading import Lock, Thread
from multiprocessing import Process
from queue import Queue
import time

# YOLOv5 环境依赖
from utils.datasets import IMG_FORMATS, VID_FORMATS
import train as v5_train
import detect as v5_detect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ------------------ Flask 应用 & 全局常量 ------------------ #
app = Flask(__name__)

PORT = 9007
train_ratio = 0.8               # 训练集划分比例
dataset_lock = Lock()           # 数据集互斥操作锁
mode = 1                        # 服务器模式

# 训练、测试数据根目录
train_save_data = ROOT / "train_save_data"
test_save_data  = ROOT / "test_save_data"
os.makedirs(train_save_data, exist_ok=True)
os.makedirs(test_save_data, exist_ok=True)

# ========== 队列相关 ==========
PERSIST_FILE = "train_tasks.json"    # 记录所有训练任务(含pending/running/finished/stopped等)
TRAIN_QUEUE = Queue()                # 内存队列，管理等待中的任务

# 全局记录“当前正在训练”的项目和PID
CURRENT_TRAINING_PROJECT = None
TRAIN_PROCESS_ID = -1

# 训练状态码（写入 state.json，用于前端查询）：
#   1 => "training"
#   2 => "paused"
#   3 => "stopped"
#   4 => "finished"
#   5 => "queued"

epochs = 300  # 供示例中估算用

# 预定义的模型配置
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


# ---------------------------------------------------------
# 工具函数：持久化队列的加载/保存
# ---------------------------------------------------------
def load_persistent_queue(file_path: str):
    """从本地json文件加载所有训练任务（无论是否pending/running/finished/...）"""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
            if isinstance(tasks, list):
                return tasks
            return []
    except:
        return []

def save_persistent_queue(file_path: str, queue_list: list):
    """把训练任务列表写回json文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(queue_list, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------
# 工具函数：YOLOv5 训练/检测子进程
# ---------------------------------------------------------
def v5_detect_task(opt):
    # v5_detect.main(opt)
    pass
    print("detect finished")

def v5_train_task(opt):
    # v5_train.main(opt)
    pass
    print("train finished")


# ---------------------------------------------------------
# 工具函数：训练队列管理线程（核心处）
# ---------------------------------------------------------
def training_queue_worker():
    """
    后台线程：循环从 TRAIN_QUEUE.get() 拿到一个状态="pending"的任务 -> 更新为running -> 启动子进程训练。
    训练完成后改为 finished/stopped/error,再取下一个,直到队列为空。

    当 process.join() 返回时，如果 exitcode != 0,则检查 train_tasks.json 是否在 /api/StopTrain 中
    已把 status 改为 "paused"；若是 => 不覆盖 paused;否则 => 写成 "stopped"。
    """
    global CURRENT_TRAINING_PROJECT, TRAIN_PROCESS_ID

    while True:
        task = TRAIN_QUEUE.get()  # 阻塞直到有新任务
        if not task:
            TRAIN_QUEUE.task_done()
            continue

        project_name = task["train_project_name"]
        # 1) 将 train_tasks.json 里对应条目的 status 改为 "running"
        all_tasks = load_persistent_queue(PERSIST_FILE)
        for t in all_tasks:
            if t.get("train_project_name") == project_name:
                t["status"] = "running"
        save_persistent_queue(PERSIST_FILE, all_tasks)

        # 2) 将项目 state.json 改成 1 => training
        state_file = train_save_data / project_name / 'state.json'
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump({"MESSAGE": "training", "STATE": 1}, f)

        # 记录当前正在训练
        CURRENT_TRAINING_PROJECT = project_name

        # 3) 构造 YOLOv5 训练 opt
        try:
            modeltype = task["model_type"]
            data_path = train_save_data / project_name / f'{project_name}.yaml'
            cfg_path  = net_cfg[modeltype]['cfg']
            wgt_path  = net_cfg[modeltype]['weights']
            hyp_path  = net_cfg[modeltype]['hyp']

            ep   = int(task["epochs"])
            bs   = int(task["batch_size"])
            imsz = int(task["imgsz"])
            rsm  = task.get("resume", False)

            opt = v5_train.parse_opt()
            opt.cfg         = str(cfg_path)
            opt.weights     = str(wgt_path)
            opt.hyp         = str(hyp_path)
            opt.data        = str(data_path)
            opt.batch_size  = bs
            opt.img_size    = imsz
            opt.epochs      = ep
            opt.resume      = rsm
            opt.image_weights = True
            opt.project     = train_save_data
            opt.name        = project_name
            opt.exist_ok    = True

        except Exception as e:
            # 构造opt失败 => 更新 state=3 (stopped/error)
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump({"MESSAGE": "error", "STATE": 3, "ERROR": str(e)}, f)
            # train_tasks.json 状态改为 "error"
            tasks_now = load_persistent_queue(PERSIST_FILE)
            for x in tasks_now:
                if x["train_project_name"] == project_name:
                    x["status"] = "error"
            save_persistent_queue(PERSIST_FILE, tasks_now)

            TRAIN_QUEUE.task_done()
            CURRENT_TRAINING_PROJECT = None
            TRAIN_PROCESS_ID = -1
            continue

        # 4) 启动子进程进行训练
        try:
            process = Process(target=v5_train_task, args=(opt,))
            process.start()
            TRAIN_PROCESS_ID = process.pid
            process.join()
            
            if process.exitcode == 0:
                # 训练正常结束 => finished
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump({"MESSAGE": "finish", "STATE": 4}, f)
                tasks_now = load_persistent_queue(PERSIST_FILE)
                for x in tasks_now:
                    if x["train_project_name"] == project_name:
                        x["status"] = "finished"
                save_persistent_queue(PERSIST_FILE, tasks_now)
            else:
                # 进程异常退出 => 可能是 stop / pause / error
                tasks_now = load_persistent_queue(PERSIST_FILE)
                # 查一下此时 tasks.json 里的最新状态
                this_status = None
                for x in tasks_now:
                    if x["train_project_name"] == project_name:
                        this_status = x["status"]

                if this_status == "paused":
                    # 用户在StopTrain接口里已经设置了 "paused"
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump({"MESSAGE": "paused", "STATE": 2}, f)
                    # 不覆盖 tasks.json，因为 StopTrain 里已经改过
                elif this_status == "error":
                    # 如果StopTrain或其他地方把它改成 "error"
                    # 则不覆盖
                    pass
                else:
                    # 默认 => stopped
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump({"MESSAGE": "stopped", "STATE": 3}, f)
                    for x in tasks_now:
                        if x["train_project_name"] == project_name and x["status"] == "running":
                            x["status"] = "stopped"
                    save_persistent_queue(PERSIST_FILE, tasks_now)

        except Exception as e:
            # 训练异常 => state=3, status=error
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump({"MESSAGE": "error", "STATE": 3, "ERROR": str(e)}, f)

            tasks_now = load_persistent_queue(PERSIST_FILE)
            for x in tasks_now:
                if x["train_project_name"] == project_name:
                    x["status"] = "error"
            save_persistent_queue(PERSIST_FILE, tasks_now)

        finally:
            TRAIN_PROCESS_ID = -1
            CURRENT_TRAINING_PROJECT = None
            TRAIN_QUEUE.task_done()


def start_training_queue_thread():
    """
    启动后台线程，并把所有 "pending" 状态的任务放进内存队列，让它们有机会执行
    """
    all_tasks = load_persistent_queue(PERSIST_FILE)
    for t in all_tasks:
        if t.get("status") == "pending":
            TRAIN_QUEUE.put(t)

    worker_thread = Thread(target=training_queue_worker, daemon=True)
    worker_thread.start()


# ---------------------------------------------------------
# 工具函数：数据集处理 / 校验（与之前类似，不变）
# ---------------------------------------------------------
def get_image_and_label_paths(imagespath, labelspath):
    img_paths, label_paths = [], []
    try:
        for root_, dirs, files in os.walk(imagespath):
            for file in files:
                if file.split('.')[-1].lower() in IMG_FORMATS:
                    img_paths.append(os.path.join(root_, file))
    except FileNotFoundError:
        pass

    try:
        for root_, dirs, files in os.walk(labelspath):
            for file in files:
                if file.split('.')[-1] == 'txt':
                    label_paths.append(os.path.join(root_, file))
    except FileNotFoundError:
        pass
    return img_paths, label_paths

def get_classes_from_labels(labelspath):
    classes_files = []
    for root_, dirs, files in os.walk(labelspath):
        if 'classes.txt' in files:
            classes_files.append(os.path.join(root_, 'classes.txt'))
    if not classes_files:
        return []

    max_classes_file = None
    max_classes_count = 0
    for classes_file in classes_files:
        with open(classes_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) > max_classes_count:
                max_classes_count = len(lines)
                max_classes_file = classes_file
    if not max_classes_file or max_classes_count == 0:
        return []
    with open(max_classes_file, 'r', encoding='utf-8') as f:
        classes = [c.strip() for c in f.readlines() if c.strip()]
    return classes

def check_images_labels_matching(img_paths, label_paths):
    img_filenames = {os.path.splitext(os.path.basename(img))[0].lower() for img in img_paths}
    label_filenames = {os.path.splitext(os.path.basename(label))[0].lower() for label in label_paths if 'classes.txt' not in label}
    
    missing_labels = img_filenames - label_filenames
    missing_images = label_filenames - img_filenames

    missing_files = {
        "missing_labels": [
            Path(img) for img in img_paths
            if os.path.splitext(os.path.basename(img))[0].lower() in missing_labels
        ],
        "missing_images": [
            Path(label) for label in label_paths
            if os.path.splitext(os.path.basename(label))[0].lower() in missing_images
        ]
    }
    return missing_files

def check_class_id_validity(label_paths, classes):
    invalid_labels = []
    for label in label_paths:
        if label.endswith('classes.txt'):
            continue
        with open(label, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                try:
                    class_id = int(line.split()[0])
                    if class_id >= len(classes):
                        invalid_labels.append({
                            "invalid_label": str(label),
                            "invalid_class_id": class_id
                        })
                except ValueError:
                    invalid_labels.append({
                        "invalid_label": str(label),
                        "error": f"标签文件格式错误，无法解析类ID: {line}"
                    })
    if invalid_labels:
        return {
            "CODE": "400", 
            "ERROR": "标签文件中存在无效的类ID或格式错误",
            "invalid_labels": invalid_labels
        }
    return None

def check_image_integrity_and_labels(img_paths, label_paths):
    invalid_images = []
    invalid_labels = []
    for img_path, label_path in zip(img_paths, label_paths):
        # 1. 验证图片是否损坏
        try:
            with Image.open(img_path) as img:
                img.verify()
        except (IOError, SyntaxError):
            invalid_images.append(str(img_path))
            continue

        # 2. 验证标签是否超出图片范围
        try:
            with open(label_path, 'r', encoding='utf-8') as label_file:
                with Image.open(img_path) as img_:
                    img_width, img_height = img_.size
                for line in label_file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width    = float(parts[3]) * img_width
                    height   = float(parts[4]) * img_height

                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    
                    if not (0 <= x_min < img_width and 0 <= y_min < img_height and
                            0 < x_max <= img_width and 0 < y_max <= img_height):
                        invalid_labels.append({
                            "label": str(label_path),
                            "image": str(img_path),
                            "bbox": [x_center, y_center, width, height],
                            "error": "标签超出图片范围"
                        })
        except Exception as e:
            invalid_labels.append({
                "label": str(label_path),
                "image": str(img_path),
                "error": f"标签文件读取或格式错误: {str(e)}"
            })
    return invalid_images, invalid_labels

def convert_bytes(bytes_value):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}PB"

def get_gpu_info():
    gpu_info = {
        "total": 0,
        "used_percent": 0
    }
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info["total"] += memory_info.total
            if memory_info.total:
                used_percent_this = (memory_info.used / memory_info.total) * 100
                gpu_info["used_percent"] = max(gpu_info["used_percent"], used_percent_this)
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        pass
    gpu_info["total"] = convert_bytes(gpu_info["total"])
    return gpu_info


# ---------------------------------------------------------
# 下面是各路由接口
# ---------------------------------------------------------

# ============== 数据集接口(TrainDataset, TestDataset) ============= #
@app.route('/api/TrainDataset', methods=['POST'])
def train_dataset():
    """
    数据集创建接口(训练集)
    """
    params = request.form if request.form else request.json
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    traindataset_required_keys = ['train_images', 'train_project_name', 'train_labels']
    train_missing_keys = [key for key in traindataset_required_keys if key not in params]
    if train_missing_keys:
        misskey = ', '.join(train_missing_keys)
        return jsonify({"CODE": 400, "ERROR": f"缺失参数: {misskey}"}), 400

    trainimages  = params.get('train_images', '')
    labelspath   = params.get('train_labels', '')
    classes      = params.get('classes_name', '')
    project_name = params.get('train_project_name', '')

    # 1. 获取图片和标签路径
    img_paths, label_paths = get_image_and_label_paths(trainimages, labelspath)
    if not img_paths:
        return jsonify({"CODE": 400, "ERROR": f"图片目录 {trainimages} 中没有找到有效的图片文件"}), 400
    if not label_paths:
        return jsonify({"CODE": 400, "ERROR": f"标签目录 {labelspath} 中没有找到有效的标签文件"}), 400

    # 2. 检查图片和标签是否一一对应
    missing_files = check_images_labels_matching(img_paths, label_paths)
    if missing_files["missing_images"] or missing_files["missing_labels"]:
        return jsonify({
            "CODE": 400,
            "ERROR": "图片和标签不匹配",
            "missing_images": [str(img) for img in missing_files["missing_images"]],
            "missing_labels": [str(label) for label in missing_files["missing_labels"]]
        }), 400

    # 3. 若没传入 classes 则自动获取
    if classes:
        classes = classes.split(",") if "," in classes else [classes]
        if len(classes) == 0:
            return jsonify({"CODE": 400, "ERROR": "传入的类别名称不能为空"}), 400
    else:
        classes = get_classes_from_labels(labelspath)
        if len(classes) == 0:
            return jsonify({"CODE": 400, "ERROR": "labels 文件夹中没有有效的 classes.txt 文件"}), 400

    # 4. 检查标签 ID
    class_check_result = check_class_id_validity(label_paths, classes)
    if class_check_result:
        return jsonify(class_check_result), 400

    # 5. 验证图片是否损坏 & 标签越界
    invalid_images, invalid_labels = check_image_integrity_and_labels(img_paths, label_paths)
    if invalid_images:
        return jsonify({"CODE": 400, "ERROR": "发现损坏的图片", "invalid_images": invalid_images}), 400
    if invalid_labels:
        return jsonify({"CODE": 400, "ERROR": "发现标签超出图片范围", "invalid_labels": invalid_labels}), 400

    # 6. 判断数据集名称是否已存在
    trainrepeatedata = []
    train_history = [f for f in os.listdir(train_save_data) if os.path.isdir(os.path.join(train_save_data, f))]
    train_project_exists = (project_name in train_history)

    if train_project_exists:
        # 数据集已存在 => 合并
        project_path = train_save_data / project_name
        images_path  = project_path / "images"
        labels_path  = project_path / "labels"
        train_txt_path = project_path / 'train.txt'
        val_txt_path   = project_path / 'val.txt'

        existing_img_paths = [str(x) for x in images_path.iterdir() if x.is_file()]
        existing_label_paths = [str(x) for x in labels_path.iterdir() if x.is_file()]

        for img_path, lbl_path in zip(img_paths, label_paths):
            img_name   = os.path.basename(img_path)
            label_name = os.path.basename(lbl_path)
            if img_name in [os.path.basename(ei) for ei in existing_img_paths]:
                trainrepeatedata.append(img_name)
                continue

            target_img_path   = images_path  / img_name
            target_label_path = labels_path / label_name

            os.makedirs(target_img_path.parent, exist_ok=True)
            shutil.copy(img_path, target_img_path)
            os.makedirs(target_label_path.parent, exist_ok=True)
            shutil.copy(lbl_path, target_label_path)

            existing_img_paths.append(str(target_img_path))
            existing_label_paths.append(str(target_label_path))

        np.random.shuffle(existing_img_paths)
        train_num = int(len(existing_img_paths) * train_ratio)
        train_imgs = existing_img_paths[:train_num]
        val_imgs   = existing_img_paths[train_num:]

        with open(train_txt_path, 'w', encoding='utf-8') as f:
            for img in train_imgs:
                f.write(str(img) + '\n')
        with open(val_txt_path, 'w', encoding='utf-8') as f:
            for img in val_imgs:
                f.write(str(img) + '\n')

    else:
        # 新建
        with dataset_lock:
            project_path = train_save_data / project_name
            images_path  = project_path / "images"
            labels_path  = project_path / "labels"
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(labels_path, exist_ok=True)

            for img_path in img_paths:
                target_path = images_path / os.path.basename(img_path)
                shutil.copy(img_path, target_path)

            for lbl_path in label_paths:
                target_path = labels_path / os.path.basename(lbl_path)
                shutil.copy(lbl_path, target_path)

            # 划分训练/验证
            img_paths = [str(images_path / os.path.basename(x)) for x in img_paths]
            np.random.shuffle(img_paths)
            train_num = int(len(img_paths) * train_ratio)
            train_imgs = img_paths[:train_num]
            val_imgs   = img_paths[train_num:]

            train_txt_path = project_path / 'train.txt'
            val_txt_path   = project_path / 'val.txt'
            with open(train_txt_path, 'w', encoding='utf-8') as f:
                for img in train_imgs:
                    f.write(str(img) + '\n')
            with open(val_txt_path, 'w', encoding='utf-8') as f:
                for img in val_imgs:
                    f.write(str(img) + '\n')

            yaml_path = project_path / f"{project_name}.yaml"
            try:
                with open(yaml_path, "w", encoding='utf-8') as y:
                    yaml.dump({
                        "train": str(train_txt_path),
                        "val": str(val_txt_path),
                        "nc": len(classes),
                        "names": classes
                    }, stream=y, allow_unicode=True)
            except OSError as e:
                return jsonify({"CODE": 400, "ERROR": f"写入 YAML 文件失败: {str(e)}"}), 400

    if trainrepeatedata:
        return jsonify({
            "CODE": 200,
            "MESSAGE": "创建数据集成功, 但以下图片已存在历史数据集中，已合并",
            "source": trainimages,
            "project_name": project_name,
            "Repeatedata": trainrepeatedata
        }), 200
    else:
        return jsonify({
            "CODE": 200,
            "MESSAGE": "创建数据集成功",
            "source": trainimages,
            "project_name": project_name,
        }), 200


@app.route('/api/TestDataset', methods=['POST'])
def test_dataset():
    """
    数据集创建接口(测试集)
    """
    params = request.form if request.form else request.json
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    testdataset_required_keys = ['test_images', 'test_project_name']
    test_missing_keys = [key for key in testdataset_required_keys if key not in params]
    if test_missing_keys:
        misskey = ', '.join(test_missing_keys)
        return jsonify({"CODE": 400, "ERROR": f"缺失参数: {misskey}"}), 400

    testimages        = params.get('test_images', '')
    test_project_name = params.get('test_project_name', '')

    if not os.path.exists(testimages):
        return jsonify({"CODE": 400, "ERROR": f"图片目录 {testimages} 不存在"}), 400

    testrepeatedata = []
    invalid_images  = []
    test_history    = [f for f in os.listdir(test_save_data) if os.path.isdir(os.path.join(test_save_data, f))]
    test_project_exists = (test_project_name in test_history)

    project_path = test_save_data / test_project_name
    images_path  = project_path / "images"

    if test_project_exists:
        existing_img_paths = [str(x.name) for x in images_path.iterdir() if x.is_file()]
        for img_path in os.listdir(testimages):
            img_full_path = os.path.join(testimages, img_path)
            if not os.path.isfile(img_full_path):
                continue
            try:
                with Image.open(img_full_path) as im:
                    im.verify()
            except (IOError, SyntaxError):
                invalid_images.append(img_path)
                continue

            img_name = os.path.basename(img_full_path)
            if img_name in existing_img_paths:
                testrepeatedata.append(img_name)
                continue

            target_img_path = images_path / img_name
            os.makedirs(target_img_path.parent, exist_ok=True)
            shutil.copy(img_full_path, target_img_path)

    else:
        with dataset_lock:
            os.makedirs(images_path, exist_ok=True)
            for img_path in os.listdir(testimages):
                img_full_path = os.path.join(testimages, img_path)
                if not os.path.isfile(img_full_path):
                    continue
                try:
                    with Image.open(img_full_path) as im:
                        im.verify()
                except (IOError, SyntaxError):
                    invalid_images.append(img_path)
                    continue
                target_path = images_path / os.path.basename(img_full_path)
                os.makedirs(target_path.parent, exist_ok=True)
                shutil.copy(img_full_path, target_path)

    total_images_count = len([x for x in images_path.iterdir() if x.is_file()])
    if testrepeatedata:
        return jsonify({
            "CODE": 200,
            "MESSAGE": "创建测试数据集成功, 但以下图片已存在于历史数据集中, 已合并",
            "source": testimages,
            "project_name": test_project_name,
            "Repeatedata": testrepeatedata,
            "InvalidImages": invalid_images,
            "TotalCount": total_images_count
        }), 200
    else:
        return jsonify({
            "CODE": 200,
            "MESSAGE": "创建测试数据集成功",
            "source": testimages,
            "project_name": test_project_name,
            "InvalidImages": invalid_images,
            "TotalCount": total_images_count
        }), 200


# ============== 训练接口（排队） ============== #
@app.route('/api/Train', methods=['POST'])
def train():
    """
    发起训练：在 train_tasks.json 中新增 / 更新一条 status="pending" 的任务，并放入内存队列。
    - 若发现同名项目处于 [running, paused, pending]，则拒绝(重复提交)；
    - 若发现同名项目处于 [finished, stopped, error]，更新该条并把 status 改为pending(相当于resume/重训)；
    - 若未找到同名项目，就创建新条目。
    """
    params = request.form if request.form else request.json
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    required = ['train_project_name']
    missing = [k for k in required if k not in params]
    if missing:
        return jsonify({"CODE": 400, "ERROR": f"缺少参数: {missing}"}), 400

    project_name = params["train_project_name"]
    modeltype  = params.get("model_type", "yolov5m")
    ep         = int(params.get("epochs", 300))
    bs         = int(params.get("batch_size", -1))
    imsz       = int(params.get("imgsz", 640))
    resume     = params.get("resume", False)

    # 校验训练项目文件夹是否存在
    train_history = [f for f in os.listdir(train_save_data) if os.path.isdir(os.path.join(train_save_data, f))]
    if project_name not in train_history:
        return jsonify({"CODE": 400, "ERROR": "请前往<我的训练集>新建项目"}), 400

    if modeltype not in net_cfg:
        return jsonify({"CODE": 400, "ERROR": "model_type错误"}), 400
    if ep <= 1:
        return jsonify({"CODE": 400, "ERROR": "epochs错误, 训练轮数必须大于1"}), 400
    if bs != -1 and (bs < 2 or bs % 2 != 0):
        return jsonify({"CODE": 400, "ERROR": "batch_size错误, 必须是2的倍数或者-1"}), 400
    if imsz < 32 or imsz % 32 != 0:
        return jsonify({"CODE": 400, "ERROR": "imgsz错误, 必须是32的倍数"}), 400

    resume_flag = True if str(resume).lower() == 'true' else False
    if resume_flag:
        last_model = train_save_data / project_name / 'weights' / 'last.pt'
        if not os.path.exists(last_model):
            return jsonify({"CODE": 400, "ERROR": "resume失败, 未找到对应的模型last.pt"}), 400

    # [关键] 加载 train_tasks.json
    tasks_list = load_persistent_queue(PERSIST_FILE)
    existing_task = None
    for t in tasks_list:
        if t["train_project_name"] == project_name:
            existing_task = t
            break

    if existing_task:
        # 如果已经有这个项目
        curr_status = existing_task["status"]
        if curr_status in ["running", "paused", "pending"]:
            # 说明它还没算真正结束 => 不允许重复提交
            return jsonify({
                "CODE": 400,
                "ERROR": f"当前项目 {project_name} 已在任务列表中，状态={curr_status}，不可重复提交"
            }), 400
        else:
            # 如果是 ["finished", "stopped", "error"] => 允许“更新”这条任务
            existing_task["model_type"] = modeltype
            existing_task["epochs"]     = ep
            existing_task["batch_size"] = bs
            existing_task["imgsz"]      = imsz
            existing_task["resume"]     = resume_flag
            existing_task["status"]     = "pending"
            # 这样就不再新增，而是复用旧条目
            updated_task = existing_task
    else:
        # 不存在同名 => 新建
        new_task = {
            "train_project_name": project_name,
            "model_type": modeltype,
            "epochs": ep,
            "batch_size": bs,
            "imgsz": imsz,
            "resume": resume_flag,
            "status": "pending"
        }
        tasks_list.append(new_task)
        updated_task = new_task

    # 更新 state.json => 5(queued)
    state_file = train_save_data / project_name / 'state.json'
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump({"MESSAGE": "queued", "STATE": 5}, f)

    # 写回
    save_persistent_queue(PERSIST_FILE, tasks_list)

    # 放进内存队列
    TRAIN_QUEUE.put(updated_task)

    return jsonify({"CODE": 200, "INFO": "训练请求已加入队列", "STATE": 5}), 200


@app.route('/api/StopTrain', methods=['POST'])
def stop_train():
    """
    停止/暂停训练接口，通过 train_state 区分：
      - train_state="stop"  => 停止 => 进程kill/队列移除 => 状态改"stopped"
      - train_state="pause" => 暂停 => 也是进程kill => 状态改"paused"
        下次可 resume 时再调用 /api/Train + resume=True
    """
    global CURRENT_TRAINING_PROJECT, TRAIN_PROCESS_ID

    params = request.form if request.form else request.json
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    train_project_name = params.get('train_project_name')
    train_state        = params.get('train_state')  # "stop" or "pause" or "2"/"3"

    # 兼容一些数值
    if train_state == 2 or train_state == "2":
        train_state = "pause"
    if train_state == 3 or train_state == "3":
        train_state = "stop"

    if not train_project_name:
        return jsonify({"CODE": 400, "ERROR": "缺少参数: train_project_name"}), 400
    if not train_state:
        return jsonify({"CODE": 400, "ERROR": "缺少参数: train_state( stop | pause )"}), 400

    project_path = train_save_data / train_project_name
    if not project_path.exists():
        return jsonify({"CODE": 400, "ERROR": "训练项目不存在"}), 400

    # 从 train_tasks.json 查找对应任务
    all_tasks = load_persistent_queue(PERSIST_FILE)
    task_found = None
    for t in all_tasks:
        if t.get("train_project_name") == train_project_name:
            task_found = t
            break

    if not task_found:
        return jsonify({"CODE": 400, "ERROR": f"在 train_tasks.json 中未找到项目 {train_project_name}"}), 400

    current_status = task_found.get("status", "")
    state_file = project_path / "state.json"

    # =========== 如果想 "暂停" ========== #
    if train_state == "pause":
        # 实际上跟“stop”一样，但我们把状态写成 paused
        # 只有正在训练的项目才能暂停
        if current_status == "running":
            if CURRENT_TRAINING_PROJECT == train_project_name and TRAIN_PROCESS_ID != -1:
                if psutil.pid_exists(TRAIN_PROCESS_ID):
                    try:
                        p = psutil.Process(TRAIN_PROCESS_ID)
                        # kill => 强制停止
                        p.terminate()
                    except Exception as e:
                        return jsonify({"CODE": 500, "ERROR": f"暂停进程失败: {str(e)}"}), 500
                else:
                    return jsonify({"CODE": 400, "ERROR": "进程PID不存在或已结束，无法暂停"}), 400

                # 更新 state.json => 2 => paused
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump({"MESSAGE": "paused", "STATE": 2}, f)

                # 更新 train_tasks.json => "paused"
                for x in all_tasks:
                    if x["train_project_name"] == train_project_name:
                        x["status"] = "paused"
                save_persistent_queue(PERSIST_FILE, all_tasks)

                TRAIN_PROCESS_ID = -1
                CURRENT_TRAINING_PROJECT = None

                return jsonify({"CODE": 200, "MESSAGE": f"项目 {train_project_name} 已暂停"}), 200
            else:
                return jsonify({"CODE": 400, "ERROR": "该项目状态为running，但不在当前TRAIN_PROCESS_ID中，无法暂停"}), 400
        else:
            return jsonify({"CODE": 400, "ERROR": f"项目 {train_project_name} 不是running状态，无法暂停"}), 400

    # =========== 如果是 "stop" ========== #
    if train_state == "stop":
        # 如果是pending => 说明还没开始训练，从内存队列剔除，改stopped
        if current_status == "pending":
            tmp_list = []
            while not TRAIN_QUEUE.empty():
                item = TRAIN_QUEUE.get_nowait()
                if item.get("train_project_name") != train_project_name:
                    tmp_list.append(item)
                TRAIN_QUEUE.task_done()
            for item in tmp_list:
                TRAIN_QUEUE.put(item)

            for x in all_tasks:
                if x["train_project_name"] == train_project_name:
                    x["status"] = "stopped"
            save_persistent_queue(PERSIST_FILE, all_tasks)

            # state.json => STATE=3
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump({"MESSAGE": "stopped", "STATE": 3}, f)

            return jsonify({"CODE": 200, "MESSAGE": f"项目 {train_project_name} 已停止 (未开始训练)"}), 200

        # 如果是running => kill进程
        if current_status == "running":
            if CURRENT_TRAINING_PROJECT == train_project_name and TRAIN_PROCESS_ID != -1:
                if psutil.pid_exists(TRAIN_PROCESS_ID):
                    try:
                        p = psutil.Process(TRAIN_PROCESS_ID)
                        p.terminate()
                    except Exception as e:
                        return jsonify({"CODE": 500, "ERROR": f"终止训练进程失败: {str(e)}"}), 500
                else:
                    return jsonify({"CODE": 400, "ERROR": "进程PID不存在或已结束，无法停止"}), 400

                # 更新 state.json => stopped
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump({"MESSAGE": "stopped", "STATE": 3}, f)

                for x in all_tasks:
                    if x["train_project_name"] == train_project_name:
                        x["status"] = "stopped"
                save_persistent_queue(PERSIST_FILE, all_tasks)

                TRAIN_PROCESS_ID = -1
                CURRENT_TRAINING_PROJECT = None

                return jsonify({"CODE": 200, "MESSAGE": f"项目 {train_project_name} 已停止 (正在训练中)"}), 200
            else:
                return jsonify({"CODE": 400, "ERROR": "该项目状态为running，但不在当前进程中或PID=-1，无法停止"}), 400

        # 如果是paused => 再stop，此时进程其实已经被kill过，但可以再改成stopped
        if current_status == "paused":
            if CURRENT_TRAINING_PROJECT == train_project_name and TRAIN_PROCESS_ID != -1:
                if psutil.pid_exists(TRAIN_PROCESS_ID):
                    try:
                        p = psutil.Process(TRAIN_PROCESS_ID)
                        p.terminate()
                    except Exception as e:
                        return jsonify({"CODE": 500, "ERROR": f"终止暂停中的进程失败: {str(e)}"}), 500

                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump({"MESSAGE": "stopped", "STATE": 3}, f)

                for x in all_tasks:
                    if x["train_project_name"] == train_project_name:
                        x["status"] = "stopped"
                save_persistent_queue(PERSIST_FILE, all_tasks)

                TRAIN_PROCESS_ID = -1
                CURRENT_TRAINING_PROJECT = None
                return jsonify({"CODE": 200, "MESSAGE": f"项目 {train_project_name} 已停止 (原本是paused)"}), 200
            else:
                # 如果进程PID不存在，说明已被杀过
                for x in all_tasks:
                    if x["train_project_name"] == train_project_name:
                        x["status"] = "stopped"
                save_persistent_queue(PERSIST_FILE, all_tasks)
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump({"MESSAGE": "stopped", "STATE": 3}, f)
                return jsonify({"CODE": 200, "MESSAGE": f"项目 {train_project_name} 已停止 (原本是paused, PID已不存在)"}), 200

        # 如果是finished/stopped/error
        return jsonify({"CODE": 400, "ERROR": f"项目 {train_project_name} 当前状态={current_status}，无需再次stop"}), 400

    # 若传进来的 train_state 不是 pause / stop
    return jsonify({"CODE": 400, "ERROR": f"不支持的 train_state: {train_state}，可选值: stop/pause"}), 400


@app.route('/api/TrainInfo', methods=['POST'])
def trainstatus():
    """
    获取训练状态接口
    """
    global epochs
    params = request.form if request.form else request.json
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    train_info_required_keys = ['train_project_name']
    train_info_missing_keys = [key for key in train_info_required_keys if key not in params]
    if train_info_missing_keys:
        return jsonify({"CODE": 400, "ERROR": f"缺少参数: {train_info_missing_keys}"}), 400

    state_train_project_name = params.get('train_project_name')
    train_history = [f for f in os.listdir(train_save_data) if os.path.isdir(os.path.join(train_save_data, f))]
    if state_train_project_name not in train_history:
        return jsonify({"CODE": 400, "ERROR": "请前往<我的训练集>新建项目"}), 400

    state_data_path = train_save_data / state_train_project_name / 'state.json'
    train_log_path  = train_save_data / state_train_project_name / 'train_log.txt'

    if not os.path.exists(state_data_path):
        return jsonify({"CODE": 400, "ERROR": "state.json 文件不存在"}), 400

    try:
        with open(state_data_path, 'r', encoding='utf-8') as f:
            statejson = json.load(f)
            state = statejson.get('STATE', -1)
    except Exception as e:
        return jsonify({"CODE": 400, "ERROR": f"读取 state.json 文件失败: {e}"}), 400

    # 如果是 5 => 排队中
    if state == 5:
        return jsonify({
            "CODE": 200,
            "MESSAGE": "获取训练信息成功",
            "STATE": 5,
            "info": "该项目正在排队中"
        }), 200

    # 如果没 train_log.txt，就只能做个估算
    if not os.path.exists(train_log_path):
        train_data_size = train_save_data / state_train_project_name / 'images'
        Approximate_time = 0.0
        if os.path.exists(train_data_size):
            Approximate_time = round(os.path.getsize(train_data_size) / 1024 / 1024 / 10, 2) * epochs
        return jsonify({
            "CODE": 200,
            "MESSAGE": "获取训练信息成功",
            "STATE": state,
            "Approximate_time": f"{Approximate_time}hours",
            "Training_progress":0
        }), 200
    else:
        try:
            with open(train_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    train_data_size = train_save_data / state_train_project_name / 'images'
                    Approximate_time = 0.0
                    if os.path.exists(train_data_size):
                        Approximate_time = round(os.path.getsize(train_data_size) / 1024 / 1024 / 10, 2) * epochs
                    return jsonify({
                        "CODE": 200,
                        "MESSAGE": "获取训练信息成功",
                        "STATE": state,
                        "Approximate_time": f"{Approximate_time}hours",
                        "Training_progress":0,
                        "MAP": 0
                    }), 200
                else:
                    pattern = re.compile(r'epoch\s*(\d+)/(\d+)\s*.*?mAP:(\d+\.\d+)\s*.*?time\s*(.*?)\s*hour')
                    for line_ in lines[::-1]:  # 倒序遍历，确保匹配到最新的time和epoch
                        match = pattern.search(line_)
                        if match:
                            epoch = int(match.group(1))
                            total_epochs = int(match.group(2))
                            mAP_value = float(match.group(3))  # 匹配到的 mAP 值
                            time_val = float(match.group(4))
                            training_progress = round(epoch / total_epochs, 2)
                            return jsonify({
                                "CODE": 200,
                                "MESSAGE": "获取训练信息成功",
                                "STATE": state,
                                "Approximate_time": f"{round(time_val * (total_epochs - epoch), 2)}hours",
                                "progress": training_progress,
                                "mAP": mAP_value  # 添加 mAP 值到返回结果中
                            }), 200
                        else:
                            return jsonify({"CODE": 200, "MESSAGE": "获取训练信息成功，但日志中未匹配到time hour"}), 200
                    return jsonify({"CODE": 200, "MESSAGE": "获取训练信息成功，但日志中未匹配到time hour"}), 200
        except Exception as e:
            return jsonify({"CODE": 400, "ERROR": f"读取 train_log.txt 文件失败: {e}"}), 400


# ============== 检测接口 ============== #
@app.route('/api/Detect', methods=['POST'])
def detect():
    """
    发起检测的接口
    """
    global DETECT_PROCESS_ID

    params = request.form if request.form else request.json
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    test_required_keys = ['train_project_name', 'test_project_name']
    test_missing_keys = [key for key in test_required_keys if key not in params]
    if test_missing_keys:
        misskey = ', '.join(test_missing_keys)
        return jsonify({"CODE": 400, "ERROR": f"缺失参数: {misskey}"}), 400

    detect_train_project_name = params.get('train_project_name')
    detect_test_project_name  = params.get('test_project_name')

    # 校验
    if not os.path.exists(train_save_data / detect_train_project_name):
        return jsonify({"CODE": 400, "ERROR": "训练项目不存在"}), 400
    elif not os.path.exists(train_save_data / detect_train_project_name / "weights" / "best.pt"):
        return jsonify({"CODE": 400, "ERROR": "训练项目没有权重文件，可能还未训练或已被删除"}), 400

    if not os.path.exists(test_save_data / detect_test_project_name):
        return jsonify({"CODE": 400, "ERROR": "测试项目不存在"}), 400

    opt = v5_detect.parse_opt()
    opt.source  = str(test_save_data / detect_test_project_name / "images")
    opt.weights = str(train_save_data / detect_train_project_name / "weights" / "best.pt")
    opt.project = test_save_data / detect_test_project_name
    opt.name    = 'result'
    opt.agnostic_nms = True
    opt.exist_ok     = True

    try:
        process = Process(target=v5_detect_task, args=(opt,))
        process.start()
        DETECT_PROCESS_ID = process.pid
        # process.join()

        save_dir = test_save_data / detect_test_project_name / "result"
        return jsonify({
            "CODE": 200,
            "MESSAGE": "检测成功",
            "save_dir": str(save_dir)
        }), 200
    except Exception as e:
        return jsonify({"CODE": 400, "ERROR": f"检测失败: {e}"}), 400


@app.route('/api/StopDetect', methods=['POST'])
def stopdetect():
    """
    停止检测的接口
    """
    global DETECT_PROCESS_ID

    params = request.form if request.form else request.json
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    stop_test_required_keys = ['detect_action']
    stop_test_missing_keys = [key for key in stop_test_required_keys if key not in params]
    if stop_test_missing_keys:
        return jsonify({"CODE": 400, "ERROR": f"缺少参数: {stop_test_missing_keys}"}), 400

    detectaction = params.get('detect_action')
    detectaction = True if detectaction == 'True' else False

    if detectaction:
        if psutil.pid_exists(DETECT_PROCESS_ID):
            p = psutil.Process(DETECT_PROCESS_ID)
            p.terminate()
            return jsonify({"CODE": 200, "MESSAGE": "停止检测成功"}), 200
        else:
            return jsonify({"CODE": 400, "ERROR": "没有正在执行的检测任务"}), 400
    else:
        return jsonify({"CODE": 400, "ERROR": "detectaction 未设置为 True"}), 400


@app.route('/api/DetectInfo', methods=['POST'])
def detectstatus():
    """
    查询检测进度的接口
    """
    params = request.form if request.form else request.json
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    test_info_required_keys = ['test_project_name']
    test_info_missing_keys = [key for key in test_info_required_keys if key not in params]
    if test_info_missing_keys:
        return jsonify({"CODE": 400, "ERROR": f"缺少参数: {test_info_missing_keys}"}), 400

    state_test_project_name = params.get('test_project_name')
    test_history = [f for f in os.listdir(test_save_data) if os.path.isdir(os.path.join(test_save_data, f))]
    if state_test_project_name not in test_history:
        return jsonify({"CODE": 400, "ERROR": "测试项目不存在"}), 400

    result_path = test_save_data / state_test_project_name / 'result'
    images_path = test_save_data / state_test_project_name / 'images'
    if not os.path.exists(result_path):
        return jsonify({"CODE": 400, "ERROR": "检测结果文件夹不存在"}), 400
    if not os.path.exists(images_path):
        return jsonify({"CODE": 400, "ERROR": "检测图片文件夹不存在"}), 400

    result_num = len(os.listdir(result_path))
    images_num = len(os.listdir(images_path))
    progress = 0.0 if images_num == 0 else result_num / images_num
    return jsonify({
        "CODE": 200,
        "MESSAGE": "检测进度",
        "progress": progress
    }), 200


# ============== 查询服务器状态接口 ============== #
@app.route('/api/ServerInfo', methods=['GET'])
def get_server_info():
    """
    获取服务器的CPU、内存、存储和GPU信息
    """
    global mode
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_cores   = psutil.cpu_count(logical=False)

    memory = psutil.virtual_memory()
    memory_used  = convert_bytes(memory.used)
    memory_total = convert_bytes(memory.total)

    disk = psutil.disk_usage('/')
    storage_used  = convert_bytes(disk.used)
    storage_total = convert_bytes(disk.total)

    gpu_info = get_gpu_info()

    server_info = {
        "mode": mode,
        "cpu": f"{cpu_cores}核",
        "cpuUse": cpu_percent,
        "disk": storage_total,
        "diskUse": disk.percent,
        "memory": memory_total,
        "memoryUse": memory.percent,
        "gpu": gpu_info
    }

    return jsonify({
        "CODE": 200,
        "SUCCESS": True,
        "DATA": server_info,
        "MESSAGE": "操作成功"
    }), 200


# ---------------------------------------------------------
# 启动入口
# ---------------------------------------------------------
if __name__ == '__main__':
    # 1) 启动前，把 train_tasks.json 中所有 "pending" 任务加载进内存队列
    start_training_queue_thread()
    # 2) 启动Flask
    app.run(host='0.0.0.0', debug=False, port=PORT)
