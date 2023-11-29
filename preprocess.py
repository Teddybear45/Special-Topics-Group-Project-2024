import os
import random
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

            image_crops = F.ten_crop(image, 600)
            mask_crops = F.ten_crop(mask, 600)

            for i in range(10):
                cropped_image_path = os.path.join(output_img_dir, filename[:-4] + str(i) + filename[-4:])
                cropped_mask_path = os.path.join(output_mask_dir, filename[:-4] + str(i) + filename[-4:])


                F.to_pil_image(image_crops[i]).save(cropped_image_path)
                F.to_pil_image(mask_crops[i]).save(cropped_mask_path)
                print("transformed " + image_path + "to " + cropped_image_path)
                print("transformed " + mask_path + "to " + cropped_mask_path)

make_crops_dir("./streetview.v3i.yolov5pytorch/valid/images", 
               "./streetview.v3i.yolov5pytorch/valid/masks",
               "./streetview.v3i.yolov5pytorch/valid/crop_images",
               "./streetview.v3i.yolov5pytorch/valid/crop_masks"
               )

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
            print("transformed " + image_path + "to " + augmented_image_path)
            print("transformed " + mask_path + "to " + augmented_mask_path)

#preprocesses a dataset from start to end - converting the labels to masks, 10x the images with crops and applying random transforms, converting the transformed masks back to labels
def prepare_dataset(input_filepath: str,
                    output_filepath: str):
    pass