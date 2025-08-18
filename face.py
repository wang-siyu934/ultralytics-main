# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

# 加载 YOLOv8n 模型
model = YOLO("yolov8n.pt")

data_yaml_path = r"E:\Python\yolov8\ultralytics-main\widerface\data.yaml"

# 开始训练
results = model.train(
    data=data_yaml_path,  # 数据集配置文件
    epochs=10,  # 训练轮数
    imgsz=640,  # 输入图像大小
    batch=16,  # 批量大小
    device="cpu",  # 使用的设备（CPU 或 GPU）
    name="face_detection",  # 训练结果保存的文件夹名称
)
