from ultralytics import YOLO
from train import train

if __name__ == "__main__":
    train(YOLO("yolov8n-seg"), "/Users/nathansun/Documents/Special-Topics-Group-Project-2024/TEST",
          100,
          8,
          run_training=False,
          run_eval=False)

    """train(YOLO("yolov8n-seg.pt"), 
          "/Users/nathansun/Documents/Special-Topics-Group-Project-2024/11:28 OVERNIGHT",
          100,
          8,)"""