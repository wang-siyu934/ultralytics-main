import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO(r"E:\Python\yolov8\ultralytics-main\runs\detect\face_detection27\weights\best.pt")



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 使用 YOLOv8 进行人臉检测
    results = model.predict(frame)
    # 在当前帧图像上绘制人脸边界框等信息方便查看
    annotated_frame = results[0].plot()
    cv2.imshow('Face Detection', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
