from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
import shutil
from PIL import Image
import time
import tkinter as tk

def fast_post(r):
    # create keys to return as input
    keys = [0, 0, 0, 0]

    height, width = (192, 320)
    # get masks from ultralytics result
    ultralytics_mask = r.masks

    if ultralytics_mask is not None:
        # if there is a detection, we get the polygon coords from the ultralytics mask
        coords = ultralytics_mask.xyn

        # create binary mask of desired dimensions
        mask = np.zeros((height, width), dtype=np.uint8)

        # draw binary mask from ultralytics coords
        for c in coords:
            pts = np.array(c)
            unnormalized_pts = (pts * np.array((height, width)[::-1])).astype(int)
            cv2.fillPoly(mask, [unnormalized_pts], 255)

        # we look at this number of pixels above the screen to determine the vector we should be going
        horizon = 80 #TODO: tune

        # if there's nothing there, we look 10 pixels lower until either we find a row with something or we reach the bottom 
        while horizon > 10: #TODO: tune
            # check if there's something in the horizon row
            if np.size(np.where(mask[horizon - 1] != 0)) != 0:
                # get the leftmost and rightmost road pixel index, average them for the middle of the road at that point
                left = np.where(mask[horizon - 1] != 0)[0][0]
                right = np.where(mask[horizon - 1] != 0)[0][-1]
                middle = int((left + right) / 2)

                # get the angle from our current location to that point
                angle = math.atan((middle - width / 2) / horizon) * 360 / (2 * math.pi)
                break

            else:
                horizon -= 10 #TODO: tune

        if horizon <= 10:
            # if we reach the bottom we just go straight
            angle = 0.0

        if np.abs(angle) < 30.0: #TODO: tune
            # if the magnitude of the angle is small, we can speed up
            keys[0] = 1
        
        elif np.abs(angle) > 60.0: #TODO: tune
            # if the magnitude of the angle is large, we need to slow down
            keys[1] = 1

        if angle < -15.0: #TODO: tune
            # if the angle is too much to the left, we need to turn left
            keys[2] = 1
        
        elif angle > 15.0: #TODO: tune
            # if the angle is too much to the right, we need to turn right
            keys[3] = 1

    return keys    

def lazy_realtime_direction(root, canvas, dirs1d: np.ndarray):
    # clear the liens
    canvas.delete("all")
    # if the bool is true, then show the arrow, else remove it
    if dirs1d[0]:
        canvas.create_line(100, 100, 100, 50, arrow=tk.LAST)
    if dirs1d[1]:
        canvas.create_line(100, 100, 100, 150, arrow=tk.LAST)
    if dirs1d[2]:
        canvas.create_line(100, 100, 50, 100, arrow=tk.LAST)
    if dirs1d[3]:
        canvas.create_line(100, 100, 150, 100, arrow=tk.LAST)


    root.update()

def main(*args):
    # take in argument for the path to the model
    default_model = "/Users/nathansun/Documents/Special-Topics-Group-Project-2024.nosync/best.pt"


    # model = YOLO(args[0]) if os.path.exists(args[0]) and len(args) == 1 else YOLO(default_model)
    model = YOLO(default_model)

    # running the arrow interface window
    root = tk.Tk()
    root.title("Key Presses")
    root.geometry("500x500")
    # overlay transparent window top but have the lines be visible
    # root.attributes("-alpha", 0.5)

    root.attributes("-topmost", True)
    root.attributes("-transparent", True)
    # root.attributes("-alpha", 0.0)

    canvas = tk.Canvas(root, width=200, height=200)
    canvas.pack()

    # run live prediction with the laptop webcam
    results = model.predict(source="0", show=True, stream=True, imgsz=320, max_det=1, verbose=False)

    # we run the calculations and display in the tkinter interface in real time!
    for r in results:
        keys = fast_post(r)
        lazy_realtime_direction(root, canvas, keys)

if __name__ == '__main__':
    main()