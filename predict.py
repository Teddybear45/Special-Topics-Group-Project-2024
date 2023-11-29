from ultralytics import YOLO
import numpy as np

model = YOLO('/Users/nathansun/Documents/Special-Topics-Group-Project-2024/11:28 OVERNIGHT/weights/best.pt') # ABSOLUTE path to weights

if __name__ == "__main__":
    results = model("./11:28 OVERNIGHT/prepared_dataset/test/images/Screenshot-2023-10-24-at-4-35-04-PM_png.rf.6b95a231b8caeb51973e023b74a898eb0.jpg")