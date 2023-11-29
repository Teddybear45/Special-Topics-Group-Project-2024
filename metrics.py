


# metric to evaluate the pixel-wise correctness of the network output
# input: path to a folder of binary masks, path to a folder of predicted masks
# output: dictionary of k: filename and v: pixel-wise accuracy out of 1.0

def pixel_accuracy(input_dir: str, output_dir: str):
