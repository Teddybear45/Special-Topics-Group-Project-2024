import os

import cv2
import numpy as np
from cv2.gapi.wip.draw import Image

import matplotlib.pyplot as plt


# metric to evaluate the pixel-wise correctness of the network output
# input: path to a folder of binary gt_masks, path to a folder of predicted gt_masks
# output: dictionary of k: filename and v: pixel-wise accuracy out of 1.0



# helper
def pixel_accuracy(pred_np, gt_np):
    # convert it to a binary mask (in case it's NxMx3) to be of size (NxM)
    pred_np = np.uint8(pred_np)
    gt_np = np.uint8(gt_np)

    # convert to binary mask
    pred_np = np.where(pred_np > 0, 1, 0)
    gt_np = np.where(gt_np > 0, 1, 0)

    # add them together (so, overlap = 2)
    overlap = pred_np + gt_np

    # get the number of pixels that are the same (sum of 2)
    num_same = np.sum(overlap == 2)
    # get the total number of true pixels (sum of 1 or 2)
    num_total = np.sum(overlap > 0)

    # return the pixel accuracy
    return num_same / num_total


def get_pixel_accuracy_of_dirs(pred_dir: str, gt_dir: str):
    accuracies = []
    for pred_filename in os.listdir(pred_dir):
        if pred_filename.endswith(".npy") or pred_filename.endswith(".jpg"):
            # open the npy mask image (should be of size NxM) todo, if not make helper function to flattent out rgb
            pred_mask_path = os.path.join(pred_dir, pred_filename)
            gt_mask_path = os.path.join(gt_dir, pred_filename)

            if pred_filename.endswith(".npy"):
                pred_mask = np.load(pred_mask_path)
                gt_mask = np.load(gt_mask_path)
            elif pred_filename.endswith(".jpg"):
                pred_mask = cv2.imread(pred_mask_path)
                gt_mask = cv2.imread(gt_mask_path)
            else:
                return []

            _pixel_accuracy = pixel_accuracy(pred_mask, gt_mask)

            accuracies.append(_pixel_accuracy)
    return accuracies

if __name__ == '__main__':
    # pred dir
    pred_dir = "predicted_masks"
    gt_dir = "gt_masks"

    accuracies = get_pixel_accuracy_of_dirs(pred_dir, gt_dir)

    # scatter these accuracies in a plot
    plt.scatter(range(len(accuracies)), accuracies)
    plt.show()


