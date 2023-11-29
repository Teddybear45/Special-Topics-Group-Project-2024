from ultralytics import YOLO
from train import train

if __name__ == "__main__":
    train(YOLO("yolov8n-seg.pt"), 
          "/Users/nathansun/Documents/Special-Topics-Group-Project-2024/11:28 OVERNIGHT",
          100,
          8,)