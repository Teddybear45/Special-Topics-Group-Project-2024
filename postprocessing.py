import os

import cv2
import numpy as np

# function to get rid of the small islands
# input: path to a folder of binary masks, output folder path
# output: none; folder of binary masks with small islands removed

def remove_small_islands(input_dir: str, output_dir: str, as_numpy=False):
    os.makedirs(output_dir, exist_ok=True)

    for mask_filename in os.listdir(input_dir):
        if mask_filename.endswith(".npy"):
            # open the npy mask image (should be of size NxM) todo, if not make helper function to flattent out rgb
            mask_image_path = os.path.join(input_dir, mask_filename)

            mask = np.load(mask_image_path)

            mask = np.uint8(mask)

            # get rid of small islands using cv2's connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

            # only keep the biggest island
            biggest_island_idx = np.argmax(stats[1:, 4]) + 1
            mask[labels != biggest_island_idx] = 0

            if as_numpy:
                # for analysis when doing post processing
                # save to numpy file
                processed_mask_path = os.path.join(output_dir, mask_filename)
                np.save(processed_mask_path, mask)
            else:
                # write to jpg
                processed_mask_path = os.path.join(output_dir, mask_filename.replace('.npy', '.jpg'))
                cv2.imwrite(processed_mask_path, mask)
    print("finished postprocessing")

def only_edges(input_str: str,
               output_str: str,
               edges_width: int):
    pass

if __name__ == '__main__':
    # a tester 2d binary matrix of size MxN
    test_2d_arr = np.random.randint(2, size=(1000, 1000))

    # make dirs
    os.makedirs("test_original", exist_ok=True)
    os.makedirs("test_processed", exist_ok=True)

    # save as an image to visualize
    test_2d_arr_colored = np.uint8(test_2d_arr * 255)
    cv2.imwrite("test_original/test.jpg", test_2d_arr_colored)

    # save the tester 2d array to a new directory
    np.save("test_original/test.npy", test_2d_arr)

    # test out the remove small islands function
    processed_mask = remove_small_islands("test_original", "test_processed")

    # save an image of the processed mask
    processed_mask_colored = np.uint8(processed_mask * 255)
    cv2.imwrite("test_processed/test.jpg", processed_mask_colored)



