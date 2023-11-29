import os
import shutil
import random
import time
from PIL import Image
from torchvision.transforms import functional as F
from convert import mask_to_label_dir, label_to_mask_dir

#creates new folders of images and masks, filled with 10 crops of the original, respective images and masks
def make_crops_dir(images_dir: str, 
                   masks_dir: str,
                   output_img_dir: str,
                   output_mask_dir: str):
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    filenames = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for filename in filenames:
        for filename in filenames:
            image_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.jpg'))

            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')

            image = F.to_tensor(image)
            mask = F.to_tensor(mask)

            #TODO: make 640 pass in as param
            image_crops = F.ten_crop(image, 640)
            mask_crops = F.ten_crop(mask, 640)

            for i in range(10):
                cropped_image_path = os.path.join(output_img_dir, filename[:-4] + str(i) + filename[-4:])
                cropped_mask_path = os.path.join(output_mask_dir, filename[:-4] + str(i) + filename[-4:])

                F.to_pil_image(image_crops[i]).save(cropped_image_path)
                F.to_pil_image(mask_crops[i]).save(cropped_mask_path)

    print("used " + str(len(filenames)) + " images in " + images_dir + " to create " + str(len(filenames) * 10) + " cropped images in " + output_img_dir)
    print("used " + str(len(filenames)) + " masks in " + masks_dir + " to create " + str(len(filenames) * 10) + " cropped masks in " + output_mask_dir)
    print("")

#applies random transforms to each image and mask in their respective folders
def transform_dir(images_dir: str,
               masks_dir: str,
               output_img_dir: str,
               output_mask_dir: str):

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    filenames = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.jpg'))

            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')

            # rotate angle by random number between -20 and 20 degrees
            angle = random.randint(-20, 20)
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

            # 50% chance to adjust the brightnest to between 75% and 125% of the original
            if random.random() > 0.5:
                image = F.adjust_brightness(image, random.random() * 0.5 + 0.75)

            # 50% chance to adjust the hue by a scale of between 0.25 and 0.75
            if random.random() > 0.5:
                image = F.adjust_hue(image, random.random() / 2 - 0.25)
            
            # 50% chance to adjust the saturation to between 75% and 125% of the original
            if random.random() > 0.5:
                image = F.adjust_saturation(image, random.random() * 0.5 + 0.75)
                
            image = F.to_tensor(image)
            mask = F.to_tensor(mask)

            augmented_image_path = os.path.join(output_img_dir, filename)
            augmented_mask_path = os.path.join(output_mask_dir, filename)

            F.to_pil_image(image).save(augmented_image_path)
            F.to_pil_image(mask).save(augmented_mask_path)
    print("applied random transforms to " + str(len(filenames)) + " images in " + images_dir + " and masks in " + masks_dir + " to " + output_img_dir + " and " + output_mask_dir)
    print("")

#preprocesses a dataset from start to end - converting the labels to masks, 10x the images with crops and applying random transforms, converting the transformed masks back to labels
def prepare_dataset(input_filepath: str,
                    output_filepath: str):
    """
    REQUIREMENTS:
    image size of 640 (resulting images that go in model will have size of 600) todo make this changable
    input file tree:
    ├── data.yaml           # hyperparameters yaml file
    └── train
        └── images          # training images
            ├── 01.png
            └── 02.png
        └── labels          # training labels
            ├── 01.txt
            └── 02.txt
    └── valid
        └── images          # validation images
            ├── 03.png
            └── 04.png
        └── labels          # validation labels
            ├── 03.txt
            └── 04.txt
    └── test 
        └── images          # test images
            ├── 05.png
            └── 06.png
        └── labels          # test labels
            ├── 05.txt
            └── 06.txt
    """
    #TODO: use for loops (for split in ["train", "valid", "test"]:)

    #starts stopwatch
    start = time.time()

    #sets and checks for correct file tree
    folders = os.listdir(input_filepath)
    assert("train" in folders and "valid" in folders and "test" in folders and "data.yaml" in folders)
    
    os.makedirs(output_filepath, exist_ok=True)
    shutil.copyfile(os.path.join(input_filepath, "data.yaml"), os.path.join(output_filepath, "data.yaml"))

    #TODO: make 800 pass in as param
    # creates the masks from labels
    label_to_mask_dir(os.path.join(input_filepath, "train/labels"),
                      os.path.join(output_filepath, "train/masks_og"), (800, 800), False)
    
    label_to_mask_dir(os.path.join(input_filepath, "valid/labels"),
                      os.path.join(output_filepath, "valid/masks_og"), (800, 800), False)
    
    label_to_mask_dir(os.path.join(input_filepath, "test/labels"),
                      os.path.join(output_filepath, "test/masks_og"), (800, 800), False)
    
    # copies the images over
    shutil.copytree(os.path.join(input_filepath, "train/images"),
                    os.path.join(output_filepath, "train/images_og"))
    
    shutil.copytree(os.path.join(input_filepath, "valid/images"),
                    os.path.join(output_filepath, "valid/images_og"))
    
    shutil.copytree(os.path.join(input_filepath, "test/images"),
                    os.path.join(output_filepath, "test/images_og"))
    
    # make crops
    make_crops_dir(os.path.join(output_filepath, "train/images_og"),
                   os.path.join(output_filepath, "train/masks_og"),
                   os.path.join(output_filepath, "train/images_cropped"),
                   os.path.join(output_filepath, "train/masks_cropped"))
    
    make_crops_dir(os.path.join(output_filepath, "valid/images_og"),
                   os.path.join(output_filepath, "valid/masks_og"),
                   os.path.join(output_filepath, "valid/images_cropped"),
                   os.path.join(output_filepath, "valid/masks_cropped"))
    
    make_crops_dir(os.path.join(output_filepath, "test/images_og"),
                   os.path.join(output_filepath, "test/masks_og"),
                   os.path.join(output_filepath, "test/images_cropped"),
                   os.path.join(output_filepath, "test/masks_cropped"))
    
    # applies random transforms
    transform_dir(os.path.join(output_filepath, "train/images_cropped"),
                  os.path.join(output_filepath, "train/masks_cropped"),
                  os.path.join(output_filepath, "train/images"),
                  os.path.join(output_filepath, "train/masks"))
    
    transform_dir(os.path.join(output_filepath, "valid/images_cropped"),
                  os.path.join(output_filepath, "valid/masks_cropped"),
                  os.path.join(output_filepath, "valid/images"),
                  os.path.join(output_filepath, "valid/masks"))
    
    transform_dir(os.path.join(output_filepath, "test/images_cropped"),
                  os.path.join(output_filepath, "test/masks_cropped"),
                  os.path.join(output_filepath, "test/images"),
                  os.path.join(output_filepath, "test/masks"))
    
    # converts labels back to masks
    mask_to_label_dir(os.path.join(output_filepath, "train/masks"),
                      os.path.join(output_filepath, "train/labels"))
    
    mask_to_label_dir(os.path.join(output_filepath, "valid/masks"),
                      os.path.join(output_filepath, "valid/labels"))
    
    mask_to_label_dir(os.path.join(output_filepath, "test/masks"),
                      os.path.join(output_filepath, "test/labels"))
    
    # cleanup
    shutil.rmtree(os.path.join(output_filepath, "train/images_cropped"))
    shutil.rmtree(os.path.join(output_filepath, "train/images_og"))
    shutil.rmtree(os.path.join(output_filepath, "train/masks_cropped"))
    shutil.rmtree(os.path.join(output_filepath, "train/masks_og"))
    shutil.rmtree(os.path.join(output_filepath, "valid/images_cropped"))
    shutil.rmtree(os.path.join(output_filepath, "valid/images_og"))
    shutil.rmtree(os.path.join(output_filepath, "valid/masks_cropped"))
    shutil.rmtree(os.path.join(output_filepath, "valid/masks_og"))
    shutil.rmtree(os.path.join(output_filepath, "test/images_cropped"))
    shutil.rmtree(os.path.join(output_filepath, "test/images_og"))
    shutil.rmtree(os.path.join(output_filepath, "test/masks_cropped"))
    shutil.rmtree(os.path.join(output_filepath, "test/masks_og"))

    #ends stopwatch, returns time taken in hours
    end = time.time()
    return round((end-start)/3600, 3)

if __name__ == "__main__":
    print(prepare_dataset("./TOY", "./TOY_RESULT"))