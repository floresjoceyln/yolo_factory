import os
import yaml
import sys
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, request, jsonify
from threading import Lock
from utils.datasets import IMG_FORMATS, VID_FORMATS

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

app = Flask(__name__)

# 定义常量
PORT = 9007                                                     # 端口号
train_ratio = 0.8                                               # 训练集比例
dataset_lock = Lock()                                           # 定义锁

data_train_save_data = ROOT / "train_save_data"                      # 训练数据集保存路径
os.makedirs(data_train_save_data, exist_ok=True)                     # 创建数据集保存路径
os.makedirs(ROOT / data_train_save_data / "testdir", exist_ok=True)  # 创建一个文件夹，不至于为空

data_test_save_data = ROOT / "test_save_data"                        # 训练数据集保存路径
os.makedirs(data_test_save_data, exist_ok=True)                      # 创建数据集保存路径
os.makedirs(ROOT / data_test_save_data / "testdir", exist_ok=True)   # 创建一个文件夹，不至于为空

# 扩展os.walk来获取所有的图片和标签文件
def get_image_and_label_paths(imagespath, labelspath):
    img_paths = []
    label_paths = []
        
    # 获取图片路径
    try:
        for root, dirs, files in os.walk(imagespath):
            for file in files:
                if file.split('.')[-1].lower() in IMG_FORMATS:
                    img_paths.append(os.path.join(root, file))
    except FileNotFoundError:
        return [], []  # 如果目录不存在，返回空列表
    
    # 获取标签路径
    try:
        for root, dirs, files in os.walk(labelspath):
            for file in files:
                if file.split('.')[-1] == 'txt':  # 只处理txt标签文件
                    label_paths.append(os.path.join(root, file))
    except FileNotFoundError:
        return [], []  # 如果目录不存在，返回空列表
                
    return img_paths, label_paths

# 选择类别最多的classes.txt文件
def get_classes_from_labels(labelspath):
    classes_files = []
    
    # 查找所有的classes.txt文件
    for root, dirs, files in os.walk(labelspath):
        if 'classes.txt' in files:
            classes_files.append(os.path.join(root, 'classes.txt'))
    
    if not classes_files:
        return []  # 如果没有找到classes.txt文件
    
    # 选择包含最多类别的classes.txt文件
    max_classes_file = None
    max_classes_count = 0
    for classes_file in classes_files:
        with open(classes_file, 'r') as f:
            lines = f.readlines()
            # 检查文件是否为空或者不符合规范（即至少包含一个非空行）
            if len([line for line in lines if line.strip()]) > max_classes_count:
                max_classes_count = len([line for line in lines if line.strip()])
                max_classes_file = classes_file
    
    if not max_classes_file or max_classes_count == 0:
        return []  # 如果没有找到有效的classes.txt文件或者文件为空/不符合规范
    
    # 读取类别
    with open(max_classes_file, 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes if c.strip()]  # 过滤掉空行
    
    if len(classes) == 0:
        return []  # 如果读取后仍然没有类别，返回空列表
    
    return classes

# 检查图片和标签是否一一对应
def check_images_labels_matching(img_paths, label_paths):
    # 提取图片文件名（去掉路径和扩展名）
    img_filenames = {os.path.splitext(os.path.basename(img))[0].lower() for img in img_paths}
    
    # 过滤掉classes.txt，避免它影响匹配
    label_filenames = {os.path.splitext(os.path.basename(label))[0].lower() for label in label_paths if 'classes.txt' not in label}
    
    # 查找不匹配的图片
    missing_labels = img_filenames - label_filenames
    missing_images = label_filenames - img_filenames
    
    # 返回不匹配的图片和标签
    missing_files = {
        "missing_images": [Path(img) for img in img_paths if os.path.splitext(os.path.basename(img))[0].lower() in missing_images],
        "missing_labels": [Path(label) for label in label_paths if os.path.splitext(os.path.basename(label))[0].lower() in missing_labels]
    }
    
    return missing_files

# 检查标签文件中的类ID是否超出classes.txt中的索引范围
def check_class_id_validity(label_paths, classes):
    invalid_labels = []  # 用于记录不合法的标签文件和问题
    for label in label_paths:
        if label.endswith('classes.txt'):
            continue  # 跳过classes.txt文件
        
        with open(label, 'r') as f:
            for line in f.readlines():
                try:
                    class_id = int(line.split()[0])  # 获取类ID
                    if class_id >= len(classes):
                        invalid_labels.append({
                            "invalid_label": str(label),  # 将Path对象转为字符串
                            "invalid_class_id": class_id
                        })
                except ValueError:
                    # 如果line.split()[0] 不是整数，跳过该行
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

# 验证图片是否损坏，并且验证标签是否超出图片范围
def check_image_integrity_and_labels(img_paths, label_paths):
    invalid_images = []
    invalid_labels = []
    
    for img_path, label_path in zip(img_paths, label_paths):
        # 1. 验证图片是否损坏
        try:
            with Image.open(img_path) as img:
                img.verify()  # 验证图片文件是否有效
        except (IOError, SyntaxError) as e:
            invalid_images.append(str(img_path))  # 添加损坏的图片路径
            continue  # 如果图片损坏，则跳过进一步的标签验证

        # 2. 验证标签是否超出图片范围
        try:
            with open(label_path, 'r') as label_file:
                img = Image.open(img_path)
                img_width, img_height = img.size  # 获取图片尺寸
                
                for line in label_file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # 跳过格式不正确的行
                    
                    # 获取标签的坐标信息
                    class_id, x_center, y_center, width, height = parts
                    
                    # 转换为像素坐标（假设标签是归一化的）
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width    = float(parts[3]) * img_width
                    height   = float(parts[4]) * img_height

                    # 计算矩形框的左上角和右下角坐标
                    x_min    = int(x_center - width / 2)
                    y_min    = int(y_center - height / 2)
                    x_max    = int(x_center + width / 2)
                    y_max    = int(y_center + height / 2)
                    
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


@app.route('/api/TrainDataset', methods=['POST'])
def train_dataset():
    params = request.form if request.form else request.json

    # 判断参数是否为空
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    traindataset_required_keys = ['train_images', 'train_project_name', 'train_labels']
    train_missing_keys = [key for key in traindataset_required_keys if key not in params]

    # 判断是否有缺失参数
    if train_missing_keys:
        misskey = ', '.join(train_missing_keys)
        return jsonify({"CODE": 400, "ERROR": f"缺失参数: {misskey}"}), 400

    trainimages    = params.get('train_images', '')    # 数据集路径  

    labelspath     = params.get('train_labels', '')     # 标签路径

    classes        = params.get('classes_name', '')   # 类别名称                             

    project_name   = params.get('train_project_name', '')   # 数据集名称

     # 获取图片和标签路径
    img_paths, label_paths = get_image_and_label_paths(trainimages, labelspath)

    # 检查图片路径是否为空
    if not img_paths:
        return jsonify({"CODE": 400, "ERROR": f"图片目录 {trainimages} 中没有找到有效的图片文件"}), 400

    # 检查标签路径是否为空
    if not label_paths:
        return jsonify({"CODE": 400, "ERROR": f"标签目录 {labelspath} 中没有找到有效的标签文件"}), 400

    # 检查图片和标签是否一一对应
    missing_files = check_images_labels_matching(img_paths, label_paths)
    if missing_files["missing_images"] or missing_files["missing_labels"]:
        return jsonify({
            "CODE": 400,
            "ERROR": "图片和标签不匹配",
            "missing_images": [str(img) for img in missing_files["missing_images"]],  # 转换为字符串
            "missing_labels": [str(label) for label in missing_files["missing_labels"]]  # 转换为字符串
        }), 400

    # 判断是否传入classes,如果存在则读取,否则根据labels文件夹下的文件生成classes
    if classes:
        classes = classes.split(",") if "," in classes else [classes]
        if len(classes) == 0:
            return jsonify({"CODE": 400, "ERROR": "传入的类别名称不能为空"}), 400
    else:
        # 使用获取类别的函数从labels文件夹中选择一个合适的classes.txt
        classes = get_classes_from_labels(labelspath)
        if len(classes) == 0:
            return jsonify({"CODE": 400, "ERROR": "labels文件夹中没有有效的classes.txt文件"}), 400
        
    # 检查标签是否超出classes.txt里面的类的索引
    class_check_result = check_class_id_validity(label_paths, classes)
    if class_check_result:
        return jsonify(class_check_result), 400

    # 验证图片是否损坏并检查标签是否超出图片范围
    invalid_images, invalid_labels = check_image_integrity_and_labels(img_paths, label_paths)

    if invalid_images:
        return jsonify({"CODE": 400, "ERROR": "发现损坏的图片", "invalid_images": invalid_images}), 400

    if invalid_labels:
        return jsonify({"CODE": 400, "ERROR": "发现标签超出图片范围", "invalid_labels": invalid_labels}), 400

    # 判断数据集名称是否存在历史记录,确保数据集名称唯一
    trainrepeatedata = []
    train_history = [f for f in os.listdir(data_train_save_data) if os.path.isdir(os.path.join(data_train_save_data, f))]
    train_project_exists = project_name in train_history

    # 如果数据集名称已经存在，执行合并操作
    if train_project_exists:
        # 获取历史数据集路径
        project_path = ROOT / "train_save_data" / project_name
        images_path = project_path / "images"
        labels_path = project_path / "labels"
        train_txt_path = project_path / 'train.txt'
        val_txt_path = project_path / 'val.txt'

        # 获取现有的图片和标签路径
        existing_img_paths = [str(images_path / img.name) for img in images_path.iterdir() if img.is_file()]
        existing_label_paths = [str(labels_path / label.name) for label in labels_path.iterdir() if label.is_file()]

        # 合并新数据到历史数据集中
        for img_path, label_path in zip(img_paths, label_paths):
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)

            # 如果图片已经存在于历史数据集中，跳过
            if img_name in [os.path.basename(existing_img) for existing_img in existing_img_paths]:
                trainrepeatedata.append(img_name)  # 保存重复的图片名称
                continue  # 跳过重复的图片

            # 复制图片和标签到历史数据集目录
            target_img_path = images_path / img_name
            target_label_path = labels_path / label_name

            os.makedirs(target_img_path.parent, exist_ok=True)
            shutil.copy(img_path, target_img_path)

            os.makedirs(target_label_path.parent, exist_ok=True)
            shutil.copy(label_path, target_label_path)

            # 更新路径
            existing_img_paths.append(str(target_img_path))
            existing_label_paths.append(str(target_label_path))

        # 更新 train.txt 和 val.txt
        np.random.shuffle(existing_img_paths)
        train_num = int(len(existing_img_paths) * train_ratio)
        train_imgs, val_imgs = existing_img_paths[:train_num], existing_img_paths[train_num:]

        with open(train_txt_path, 'w') as f:
            for img in train_imgs:
                f.write(str(img) + '\n')

        with open(val_txt_path, 'w') as f:
            for img in val_imgs:
                f.write(str(img) + '\n')

    else:
        # 如果数据集名称不存在，则创建新的数据集
        with dataset_lock:
            project_path = ROOT / "train_save_data" / project_name
            images_path = project_path / "images"
            labels_path = project_path / "labels"
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(labels_path, exist_ok=True)

            # 复制新的图片和标签到新数据集目录
            for img_path in img_paths:
                target_path = images_path / os.path.basename(img_path)
                os.makedirs(target_path.parent, exist_ok=True)
                shutil.copy(img_path, target_path)

            for label_path in label_paths:
                target_path = labels_path / os.path.basename(label_path)
                os.makedirs(target_path.parent, exist_ok=True)
                shutil.copy(label_path, target_path)

            # 更新路径为新目录的路径
            img_paths = [str(images_path / os.path.basename(img)) for img in img_paths]
            label_paths = [str(labels_path / os.path.basename(label)) for label in label_paths]

            # 划分训练集和验证集
            np.random.shuffle(img_paths)

            train_num = int(len(img_paths) * train_ratio)
            train_imgs, val_imgs = img_paths[:train_num], img_paths[train_num:]

            # 写入 train.txt 和 val.txt
            train_txt_path = project_path / 'train.txt'
            val_txt_path = project_path / 'val.txt'
            with open(train_txt_path, 'w') as f:
                for img in train_imgs:
                    f.write(str(img) + '\n')
            with open(val_txt_path, 'w') as f:
                for img in val_imgs:
                    f.write(str(img) + '\n')

            # 写入训练使用的 --data 参数 yaml 文件
            yaml_path = project_path / f"{project_name}.yaml"
            try:
                with open(yaml_path, "w") as y:
                    yaml.dump({
                        "train": str(train_txt_path),
                        "val": str(val_txt_path),
                        "nc": len(classes),
                        "names": classes
                    }, stream=y, allow_unicode=True)
            except OSError as e:
                return jsonify({"CODE": 400, "ERROR": f"写入 YAML 文件失败: {str(e)}"}), 400

    # 返回处理结果
    if trainrepeatedata:
        return jsonify({
            "CODE": 200,
            "MESSAGE": "创建数据集成功, 但以下图片已经存在于历史数据集中,已合并",
            "source": trainimages,
            "project_name": project_name,
            "Repeatedata": trainrepeatedata
        })
    else:
        return jsonify({
            "CODE": 200,
            "MESSAGE": "创建数据集成功",
            "source": trainimages,
            "project_name": project_name,
            "classes": classes
        })



@app.route('/api/TestDataset', methods=['POST'])
def test_dataset():
    params = request.form if request.form else request.json

    # 判断参数是否为空
    if not params:
        return jsonify({"CODE": 400, "ERROR": "参数不能为空"}), 400

    testdataset_required_keys = ['test_images', 'test_project_name']
    test_missing_keys = [key for key in testdataset_required_keys if key not in params]

    # 判断是否有缺失参数
    if test_missing_keys:
        misskey = ', '.join(test_missing_keys)
        return jsonify({"CODE": 400, "ERROR": f"缺失参数: {misskey}"}), 400

    testimages = params.get('test_images', '')  # 数据集路径
    test_project_name = params.get('test_project_name', '')  # 数据集名称

    # 判断图片路径是否存在
    if not os.path.exists(testimages):
        return jsonify({"CODE": 400, "ERROR": f"图片目录 {testimages} 不存在"}), 400

    # 判断数据集名称是否存在历史记录，确保数据集名称唯一
    testrepeatedata = []
    invalid_images = []  # 保存不可用图片
    test_history = [f for f in os.listdir(data_test_save_data) if os.path.isdir(os.path.join(data_test_save_data, f))]
    test_project_exists = test_project_name in test_history

    # 项目路径
    project_path = ROOT / "test_save_data" / test_project_name
    images_path = project_path / "images"

    # 如果数据集名称已经存在，执行合并操作
    if test_project_exists:
        # 获取现有的图片路径
        existing_img_paths = [str(img.name) for img in images_path.iterdir() if img.is_file()]

        # 合并新数据到历史数据集中
        for img_path in os.listdir(testimages):
            img_full_path = os.path.join(testimages, img_path)
            if not os.path.isfile(img_full_path):
                continue  # 跳过非文件

            # 验证图片是否有效
            try:
                with Image.open(img_full_path) as img:
                    img.verify()
            except (IOError, SyntaxError):
                invalid_images.append(img_path)  # 添加到无效图片列表
                os.remove(img_full_path)  # 删除不可用图片
                continue  # 跳过无效图片

            img_name = os.path.basename(img_full_path)

            # 如果图片已经存在于历史数据集中，跳过
            if img_name in existing_img_paths:
                testrepeatedata.append(img_name)  # 保存重复的图片名称
                continue  # 跳过重复的图片

            # 复制图片到历史数据集目录
            target_img_path = images_path / img_name
            os.makedirs(target_img_path.parent, exist_ok=True)
            shutil.copy(img_full_path, target_img_path)
    else:
        # 如果数据集名称不存在，则创建新的数据集
        with dataset_lock:
            os.makedirs(images_path, exist_ok=True)

            # 复制新的图片到新数据集目录
            for img_path in os.listdir(testimages):
                img_full_path = os.path.join(testimages, img_path)
                if not os.path.isfile(img_full_path):
                    continue  # 跳过非文件

                # 验证图片是否有效
                try:
                    with Image.open(img_full_path) as img:
                        img.verify()
                except (IOError, SyntaxError):
                    invalid_images.append(img_path)  # 添加到无效图片列表
                    os.remove(img_full_path)  # 删除不可用图片
                    continue  # 跳过无效图片

                target_path = images_path / os.path.basename(img_full_path)
                os.makedirs(target_path.parent, exist_ok=True)
                shutil.copy(img_full_path, target_path)

    # 获取项目的总图片数量
    total_images_count = len([img for img in images_path.iterdir() if img.is_file()])

    # 返回处理结果
    if testrepeatedata:
        return jsonify({
            "CODE": 200,
            "MESSAGE": "创建测试数据集成功, 但以下图片已经存在于历史数据集中, 已合并",
            "source": testimages,
            "project_name": test_project_name,
            "Repeatedata": testrepeatedata,
            "InvalidImages": invalid_images,  # 不可用图片
            "TotalCount": total_images_count  # 项目的总图片数量
        })
    else:
        return jsonify({
            "CODE": 200,
            "MESSAGE": "创建测试数据集成功",
            "source": testimages,
            "project_name": test_project_name,
            "InvalidImages": invalid_images,  # 不可用图片
            "TotalCount": total_images_count  # 项目的总图片数量
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=PORT)
