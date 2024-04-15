from ultralytics import YOLO
import numpy as np
import cv2
import os
import shutil
from PIL import Image
import time
from convert import label_to_mask, video_to_frames, frames_to_video

# functions create a folder with the following:
# mask.jpg
# mask.npy
# edges.jpg
# edges.npy
# original.jpg
# overlay.jpg

def predict_image(model: YOLO, # must be absolute path, idk why
                 image_path: str, # path to original image
                 result_path: str, # folder to store run files
                 edges_width: int, # thickness of the edges
                 overlay_edges: bool,
                 rgba: tuple):

    #makes folder for run
    os.makedirs(result_path, exist_ok=True)
    
    #runs prediction
    result_obj = model(image_path, max_det=1)

    #extracts the coordinates from the ultralytics result obj
    result_obj = result_obj[0]
    ultralytics_mask = result_obj.masks
    print(ultralytics_mask)
    if ultralytics_mask is not None:
            coords = ultralytics_mask.xyn
             # get first and only prediction, adds to flattened list
            coords = np.array(coords)[0]
            flattened_coords = [0] #class index for label format

            for pair in coords:
                for value in pair:
                    flattened_coords.append(value)
            
            formatted_string = ' '.join(map(str, flattened_coords))
    else:
        formatted_string = '0'

    #converts to string
    with open(os.path.join(result_path, "coords.txt"), 'w') as label_file:
        label_file.write(formatted_string)

    #gets image dimensions
    #TODO: figure out if the flip should go here or in the label_to_mask functions
    image_dims = np.flip(Image.open(image_path).size)

    #saves the mask as jpg and npy
    label_to_mask(os.path.join(result_path, "coords.txt"), os.path.join(result_path, "mask.jpg"), image_dims, False)    
    label_to_mask(os.path.join(result_path, "coords.txt"), os.path.join(result_path, "mask.npy"), image_dims, True)

    #loads the mask we just madeas np array
    mask = np.load(os.path.join(result_path, "mask.npy"))

    #shift mask left and right based on edge_width, subtracting to get only edges
    left_shift_amount = int((edges_width+1)/2)
    right_shift_amount = int(edges_width/2)

    left_shift = np.hstack((mask[:, left_shift_amount:], np.zeros((np.shape(mask)[0], left_shift_amount))))
    right_shift = np.hstack((np.zeros((np.shape(mask)[0], right_shift_amount)), mask[:, :-1*right_shift_amount]))

    edges = np.absolute(np.subtract(left_shift, right_shift))

    #saves edges as npy
    np.save(os.path.join(os.path.join(result_path, "edges.npy")), edges)

    #converts to and save edges as jpg
    edges = edges.astype(np.uint8)
    edges_img = Image.fromarray(edges)
    edges_img.save(os.path.join(result_path, "edges.jpg"))

    shutil.copyfile(image_path, os.path.join(result_path, "original.jpg"))

    #open either edges or mask depending on param
    if (overlay_edges):
        overlay = Image.open(os.path.join(result_path, "edges.jpg"))
    else:
        overlay = Image.open(os.path.join(result_path, "mask.jpg"))

    #creates the overlay mask
    overlay = overlay.convert("RGBA")
    datas = overlay.getdata()

    #converts either the mask to edges to the overlay mask by setting black pixels as transparent and white ones as the defined color/opacity
    #TODO: make edges better
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(rgba)

    if ultralytics_mask is not None:
        #saves the new data to the overlay image
        overlay.putdata(newData)

        #mask for pasting overlay onto the background (og image)
        _, _, _, mask = overlay.split()

        #opens background and pastes overlay on
        background = Image.open(os.path.join(result_path, "original.jpg"))
        background.paste(overlay, (0, 0), mask)

        #saves new image (background + overlay) as overlay.jpg
        background.save(os.path.join(result_path, "overlay.jpg"))
    else:
        shutil.copyfile(os.path.join(result_path, "original.jpg"), os.path.join(result_path, "overlay.jpg"))

def predict_images_batch(model: YOLO,
                          image_paths: list, # path to original image
                          result_paths: list, # folder to store run files
                          edges_width: int, # thickness of the edges
                          overlay_edges: bool,
                          rgba: tuple):

    for i in result_paths:
        os.makedirs(i, exist_ok=True)

    results = model(image_paths, max_det=1)

    ind = 0

    for result_obj in results:
        image_path = image_paths[ind]
        result_path = result_paths[ind]

        result_obj = result_obj[0]
        ultralytics_mask = result_obj.masks
        if ultralytics_mask is not None:
            coords = ultralytics_mask.xyn
             # get first and only prediction, adds to flattened list
            coords = np.array(coords)[0]
            flattened_coords = [0] #class index for label format

            for pair in coords:
                for value in pair:
                    flattened_coords.append(value)
            
            formatted_string = ' '.join(map(str, flattened_coords))
        else:
            formatted_string = '0'

        #converts to string
        
        with open(os.path.join(result_path, "coords.txt"), 'w') as label_file:
            label_file.write(formatted_string)

        #gets image dimensions
        #TODO: figure out if the flip should go here or in the label_to_mask functions
        image_dims = np.flip(Image.open(image_path).size)

        #saves the mask as jpg and npy
        label_to_mask(os.path.join(result_path, "coords.txt"), os.path.join(result_path, "mask.jpg"), image_dims, False)    
        label_to_mask(os.path.join(result_path, "coords.txt"), os.path.join(result_path, "mask.npy"), image_dims, True)

        #loads the mask we just madeas np array
        mask = np.load(os.path.join(result_path, "mask.npy"))

        #shift mask left and right based on edge_width, subtracting to get only edges
        left_shift_amount = int((edges_width+1)/2)
        right_shift_amount = int(edges_width/2)

        left_shift = np.hstack((mask[:, left_shift_amount:], np.zeros((np.shape(mask)[0], left_shift_amount))))
        right_shift = np.hstack((np.zeros((np.shape(mask)[0], right_shift_amount)), mask[:, :-1*right_shift_amount]))

        edges = np.absolute(np.subtract(left_shift, right_shift))

        #saves edges as npy
        np.save(os.path.join(os.path.join(result_path, "edges.npy")), edges)

        #converts to and save edges as jpg
        edges = edges.astype(np.uint8)
        edges_img = Image.fromarray(edges)
        edges_img.save(os.path.join(result_path, "edges.jpg"))

        shutil.copyfile(image_path, os.path.join(result_path, "original.jpg"))

        #open either edges or mask depending on param
        if (overlay_edges):
            overlay = Image.open(os.path.join(result_path, "edges.jpg"))
        else:
            overlay = Image.open(os.path.join(result_path, "mask.jpg"))

        #creates the overlay mask
        overlay = overlay.convert("RGBA")
        datas = overlay.getdata()

        #converts either the mask to edges to the overlay mask by setting black pixels as transparent and white ones as the defined color/opacity
        #TODO: make edges better
        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(rgba)

        if ultralytics_mask is not None:
            #saves the new data to the overlay image
            overlay.putdata(newData)

            #mask for pasting overlay onto the background (og image)
            _, _, _, mask = overlay.split()

            #opens background and pastes overlay on
            background = Image.open(os.path.join(result_path, "original.jpg"))
            background.paste(overlay, (0, 0), mask)

            #saves new image (background + overlay) as overlay.jpg
            background.save(os.path.join(result_path, "overlay.jpg"))
        else:
            shutil.copyfile(os.path.join(result_path, "original.jpg"), os.path.join(result_path, "overlay.jpg"))

        ind += 1


def predict_video_as_frames(model: YOLO, # must be absolute path, idk why
                  video_path: str,
                  result_path: str,
                  edges_width: int,
                  overlay_edges: bool,
                  rgba: tuple,
                  fps: int):

    #makes folders for run
    #TODO: we probably dont need the first statement, keeping for readability
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(result_path, "frames/"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "perframeruns/"), exist_ok=True)

    os.makedirs(os.path.join(result_path, "framefolders/originals"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "framefolders/masks"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "framefolders/edges"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "framefolders/overlays"), exist_ok=True)

    video_to_frames(video_path, os.path.join(result_path, "frames/"), fps)

    frames_dir_list = os.listdir(os.path.join(result_path, "frames/"))
    frames_dir_list.sort()

    source_list = [os.path.join(result_path, "frames", i) for i in frames_dir_list]
    destination_list = [os.path.join(result_path, "perframeruns", i.replace(".png", "")) for i in frames_dir_list]
    

    """predict_images_batch(model,
                         source_list,
                         destination_list,
                         edges_width,
                         overlay_edges,
                         rgba)"""
    for i in frames_dir_list:
        predict_image(model,
                      os.path.join(result_path, "frames", i),
                      os.path.join(result_path, "perframeruns", i.replace(".png", "")),
                      edges_width,
                      overlay_edges,
                      rgba)
    for i in frames_dir_list:
        shutil.copy(os.path.join(result_path, "perframeruns", i.replace(".png", ""), "original.jpg"),
                    os.path.join(result_path, "framefolders/originals", i.replace(".png", ".jpg")))
        shutil.copy(os.path.join(result_path, "perframeruns", i.replace(".png", ""), "mask.jpg"),
                    os.path.join(result_path, "framefolders/masks", i.replace(".png", ".jpg")))
        shutil.copy(os.path.join(result_path, "perframeruns", i.replace(".png", ""), "edges.jpg"),
                    os.path.join(result_path, "framefolders/edges", i.replace(".png", ".jpg")))
        shutil.copy(os.path.join(result_path, "perframeruns", i.replace(".png", ""), "overlay.jpg"),
                    os.path.join(result_path, "framefolders/overlays", i.replace(".png", ".jpg")))

    frames_to_video(os.path.join(result_path, "framefolders/originals"), os.path.join(result_path, "original.mp4"), fps)
    frames_to_video(os.path.join(result_path, "framefolders/masks"), os.path.join(result_path, "mask.mp4"), fps)
    frames_to_video(os.path.join(result_path, "framefolders/edges"), os.path.join(result_path, "edges.mp4"), fps)
    frames_to_video(os.path.join(result_path, "framefolders/overlays"), os.path.join(result_path, "overlay.mp4"), fps)

    #adjust original video framerate to match
    
    #cleanup

count = 0
success = True
if __name__ == "__main__":
    model = YOLO("/Users/nathansun/Documents/Special-Topics-Group-Project-2024.nosync/best.pt")
    """predict_image(model,
                  "./roadpic.jpg",
                  "./predicttest",
                  30,
                  False,
                  (100, 100, 255, 100))"""
    """t0 = time.time()
    predict_video_as_frames(model,
                  "./testvid.mp4",
                  "./vidtest",
                  60,
                  False,
                  (100, 255, 100, 100),
                  10)
    t1 = time.time()
    total = t1-t0
    print(total)"""
    predict_image(model,
                  "./toy.jpg",
                  "./emptyresult",
                  20,
                  False,
                  (255, 0, 0, 100))
    """predict_images_batch(
        model,
        [os.path.join("./batch_test", i) for i in os.listdir("./batch_test")],
        [os.path.join("./batch_test_result", i) for i in os.listdir("./batch_test")],
        30,
        False,
        (100, 100, 255, 100))"""
