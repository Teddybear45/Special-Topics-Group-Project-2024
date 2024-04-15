import numpy as np
import math
import cv2
import os
from PIL import Image
from predict import *
import shutil
from convert import frames_to_video

def create_path_plan(
    mask: np.ndarray, # np binary mask input
    max_y: int): # highest y from bottom to predict to, -1 for no max

    result = np.zeros(np.shape(mask))

    height, width = np.shape(mask)

    if max_y < 0 or max_y > height:
        max_y = height

    for i in range(max_y):
        if np.size(np.where(mask[height - 1 - i] != 0)) != 0:
            left = np.where(mask[height - 1 - i] != 0)[0][0]
            right = np.where(mask[height - 1 - i] != 0)[0][-1]
            middle = int((left + right) / 2)
            result[height - 1 - i][middle] = 1
    
    return result

def create_viewable_path( # since path only have a width of exactly 1 you cant really see it for large images
    path: np.ndarray,
    output_path: str,
    r: int): # distance on either side of path horizontally to fill

    result = np.copy(path)
    height, width = np.shape(path)

    for i in range(height):
        if np.size(np.where(path[i] != 0)) != 0:
            point_idx = np.where(path[i] != 0)[0][0]
            for j in range(max(0, point_idx - r), min(width - 1, point_idx + r)):
                result[i][j] = 1

    cv2.imwrite(output_path, result * 255)

def pred_move(
    path: np.ndarray,
    d_h: int): # how many pixels from the bottom to use as the point to go to

    height, width = np.shape(path)
    
    if d_h < 0 or d_h > height:
        d_h = height

    while d_h > 0 and not np.any(path[height - d_h] == 1):
        d_h -= 1

    if d_h == 0:
        return 0

    d_l = np.where(path[height - d_h] != 0)[0][0] - width / 2
    return math.atan(d_l / d_h) * 360 / (2 * math.pi)

def pred_move_photo(
    run_path: str,
    d_h: int,
    r: int,
    rgba: tuple
):
    mask = np.load(os.path.join(run_path, "mask.npy"))
    path = create_path_plan(mask, d_h)
    np.save(os.path.join(run_path, "path.npy"), path * 255)
    create_viewable_path(path, os.path.join(run_path, "path.jpg"), r)
    
    overlay = Image.open(os.path.join(run_path, "path.jpg"))

    overlay = overlay.convert("RGBA")
    datas = overlay.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(rgba)

    overlay.putdata(newData)

    _, _, _, mask = overlay.split()

    background = Image.open(os.path.join(run_path, "original.jpg"))
    background.paste(overlay, (0, 0), mask)


    background.save(os.path.join(run_path, "path_overlay.jpg"))

    return pred_move(path, d_h)

def pred_move_video(
    run_path: str,
    d_v: int,
    r: int,
    rgba: tuple,
    fps: int
):
    frames_dir_list = os.listdir(os.path.join(run_path, "perframeruns/"))
    frames_dir_list.sort()

    os.makedirs(os.path.join(run_path, "framepaths"), exist_ok=True)
    os.makedirs(os.path.join(run_path, "framepathoverlay"), exist_ok=True)
    print(frames_dir_list)

    angles = []

    for i in frames_dir_list:
        angles.append(pred_move_photo(os.path.join(run_path, "perframeruns", i), d_v, r, (rgba)))

        shutil.copy(os.path.join(run_path, "perframeruns", i, "path.jpg"),
                    os.path.join(run_path, "framepaths", i +  ".jpg"))
        shutil.copy(os.path.join(run_path, "perframeruns", i, "path_overlay.jpg"),
                    os.path.join(run_path, "framepathoverlay", i +  ".jpg"))
        
        frames_to_video(os.path.join(run_path, "framepaths"), os.path.join(run_path, "path.mp4"), fps)
        frames_to_video(os.path.join(run_path, "framepathoverlay"), os.path.join(run_path, "path_overlay.mp4"), fps)
    
    return angles

# def pred_move_fast(frame) {
    
# }

if __name__ == '__main__':
    #print(pred_move_photo("./cache/runs/202422716219", 100, 10, (0, 255, 0, 100)))
    #print(pred_move_video("./cache/runs/2024228201325", -1, 10, (255, 0, 0, 100), 20))
    pred_move_photo("./demoresult", -1, 10, (255, 255, 255, 255))
