import numpy as np
import math
import cv2

def create_path_plan(
    mask: np.ndarray, # np binary mask input
    max_y: int): # highest y from bottom to predict to, -1 for no max

    result = np.zeros(np.shape(mask))

    height, width = np.shape(mask)

    if max_y < 0 or max_y > height:
        max_y = height

    for i in range(max_y):
        left_it = 0
        right_it = width - 1
        middle = -1

        while left_it < width and mask[height - 1 - i][left_it] == 0:
            left_it += 1
        
        while right_it >= 0 and mask[height - 1 - i][right_it] == 0:
            right_it -= 1

        if left_it <= right_it:
            middle = int((left_it + right_it) / 2)
            result[height - 1 - i][middle] = 1
    
    return result

def viewable_path( # since path only have a width of exactly 1 you cant really see it for large images
    mask: np.ndarray,
    output_path: str,
    r: int): # distance on either side of path horizontally to fill

    result = np.copy(mask)

    for row_idx, row in enumerate(mask):
        for col_idx, value in enumerate(row):
            if value == 0:
                for i in range(max(0, col_idx - r), min(len(row), col_idx + r + 1)):
                    if row[i] == 1:
                        result[row_idx, col_idx] = 1
                        break
    
    cv2.imwrite(output_path, result * 255)

def pred_move(
    path: np.ndarray,
    d_h: int): # how many pixels from the bottom to use as the point to go to

    height, width = np.shape(mask)
    
    if d_h < 0 or d_h > height:
        d_h = height

    while d_h > 0 and not np.any(path[height - d_h] == 1):
        d_h -= 1
        print(d_h)

    d_v = np.where(path[height - d_h] == 1)[0] - width / 2
    
    return math.atan(d_h / d_v) * 360 / (2 * math.pi)

if __name__ == '__main__':
    mask = cv2.imread("./toy.jpg", cv2.IMREAD_GRAYSCALE)
    path = create_path_plan(mask, 32)
    viewable_path(path, "./toy_result.jpg", 0)
    print(pred_move(path, -1))
