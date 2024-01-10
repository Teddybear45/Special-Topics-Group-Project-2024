import os
import numpy as np
import cv2

#converts a binary mask to the yolov5 label format
def mask_to_label(mask_path: str, label_path: str, class_id: int):
    # load the mask image and convert it to grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # find contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        with open(label_path, 'w') as label_file:
            # convert contour to polygon format
            polygon_points = contour.reshape(-1, 2)
            normalized_polygon = [(x / mask.shape[1], y / mask.shape[0]) for x, y in polygon_points]
        
            # writes the coordinates to label txt file
            label_file.write(f"{class_id} ")
            for x, y in normalized_polygon:
                label_file.write(f"{x} {y} ")
    else:
        # writes the coordinates to label txt file
        with open(label_path, 'w') as label_file:
            label_file.write(f"{class_id} ")

#converts a folder of binary masks to yolov5 label format
def mask_to_label_dir(input_dir: str, output_dir: str):
    counter = 0
    os.makedirs(output_dir, exist_ok=True)

    for mask_filename in os.listdir(input_dir):
        if mask_filename.endswith('.jpg'):
            mask_image_path = os.path.join(input_dir, mask_filename)
            label_txt_path = os.path.join(output_dir, mask_filename.replace('.jpg', '.txt'))
            mask_to_label(mask_image_path, label_txt_path, 0)
            counter += 1

    print("converted " + str(len([i.endswith("jpg") for i in os.listdir(input_dir)])) + " labels in " + input_dir + " to masks in " + output_dir)
    print("")


#converts the yolov5 label to a list of polygon coords
def get_polygon_coords(label_path: str):
    # load the yolov5 label txt file
    with open(label_path, 'r') as label_file:
        lines = label_file.readlines()

    # parse the polygon coordinates from the label file
    polygon_coords = []
    for line in lines:
        parts = line.strip().split()
        class_idx = int(parts[0])
        coords = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
        polygon_coords.append((class_idx, coords))
    
    return polygon_coords

#creates a binary mask as a numpy array from a list of polygon coords
def create_mask_image(polygon_coords: list, mask_size: tuple):
    # creates empty mask array
    mask = np.zeros(mask_size, dtype=np.uint8)
    if not polygon_coords[0] == (0, []):
        for class_idx, coords in polygon_coords:
            pts = np.array(coords)
            unnormalized_pts = (pts * np.array(mask_size[::-1])).astype(int)  # reverse mask_size for (height, width)
            cv2.fillPoly(mask, [unnormalized_pts], 255)
    return mask

def label_to_mask(label_path: str, mask_path: str, mask_size: tuple, as_numpy: bool):
    if label_path.endswith('.txt'):
        filename = os.path.basename(label_path)

        polygon_points = get_polygon_coords(label_path)
        mask = create_mask_image(polygon_points, mask_size)
        
        if as_numpy:
            # for analysis when doing post processing
            # save to numpy file
            np.save(mask_path, mask)

        else:
            # write to jpg
            cv2.imwrite(mask_path, mask)

#converts a folder of yolov5 labels to binary masks
def label_to_mask_dir(input_dir: str, output_dir: str, mask_size: tuple, as_numpy: bool):
    os.makedirs(output_dir, exist_ok=True)
    counter = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            txt_path = os.path.join(input_dir, filename)

            polygon_points = get_polygon_coords(txt_path)
            mask = create_mask_image(polygon_points, mask_size)
            
            if as_numpy:
                # for analysis when doing post processing
                # save to numpy file
                mask_path = os.path.join(output_dir, filename.replace('.txt', '.npy'))
                np.save(mask_path, mask)

            else:
                # write to jpg
                mask_path = os.path.join(output_dir, filename.replace('.txt', '.jpg'))
                cv2.imwrite(mask_path, mask)
            
    #TODO: figure out why -1 in length
    print("converted " + str(len([i.endswith(".txt") for i in os.listdir(input_dir)]) - 1) + " labels in " + input_dir + " to masks in " + output_dir)
    print("")

def video_to_frames(video_path: str, output_dir: str, fps: int):
    
    # open the video file
    cap = cv2.VideoCapture(video_path)

    # create output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # calculate the interval between frames based on the desired fps
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)

    # start extracting frames
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # save frame if it's within the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count // frame_interval:04d}.png")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

def frames_to_video(images_dir: str, output_video_path: str, fps: int):
    image_files = os.listdir(images_dir)
    image_files.sort()

    # use the first image to get dimensions
    first_image = cv2.imread(os.path.join(images_dir, image_files[0]))
    height, width, _ = first_image.shape

    # define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for i in image_files:
        frame = cv2.imread(os.path.join(images_dir, i))
        out.write(frame)

    
if __name__ == "__main__":
    print(get_polygon_coords("/Users/nathansun/Documents/Special-Topics-Group-Project-2024/emptytest.txt"))