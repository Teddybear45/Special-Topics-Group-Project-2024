from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')


results = model.train(data='/Users/nathansun/Documents/Special-Topics-Group-Project-2024/streetview.v3i.yolov5pytorch/data.yaml',
                      epochs=10,
                      batch=8,
                      imgsz=640)