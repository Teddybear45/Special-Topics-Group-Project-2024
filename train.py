from ultralytics import YOLO
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from preprocess import prepare_dataset
from convert import create_mask_image, label_to_mask_dir
from postprocessing import remove_small_islands
from metrics import get_pixel_accuracy_of_dirs

def train(model: YOLO,                          # model to train
          folder: str,                          # filepath to the folder of the run, containing just the dataset in folder named "dataset"
          num_epochs: int,                      # number of epochs to train for
          batch_size: int,                      # batch size
          run_training=True,                    # if true and theres no weights present, runs training
          run_eval=True):                       # if true and theres no eval folder present, runs teseting
    
    # PREPROCESSING

    #no prepared dataset given
    if "prepared_dataset" not in os.listdir(folder):
        print("")
        print("!!! STARTING PREPROCESSING !!!")
        print("")

        # runs preprocessing
        path_to_dataset = os.path.join(folder, "dataset")
        preprocessed_dataset = os.path.join(folder, "prepared_dataset")
        preprocessing_time = prepare_dataset(path_to_dataset, preprocessed_dataset)

        print("")
        print("finished preprocessing in " + str(preprocessing_time) + " hours")

    #given prepared dataset
    else:
        print("")
        print("!!!  USING PROVIDED PREPROCESSED DATASET !!!")
        print("")

        preprocessed_dataset = os.path.join(folder, "prepared_dataset")
    
    # TRAINING

    # cleans up from past runs
    if "runs" in os.listdir("./"):
        shutil.rmtree("./runs")
    
    #not skipping training and no folder of weights or weights folder doesnt have best.pt
    if run_training and ("weights" not in os.listdir(folder) or "best.pt" not in os.listdir(os.path.join(folder, "weights"))):
        print("")
        print("!!! STARTING TRAINING !!!")
        print("")

        # starts stopwatch
        train_start = time.time()

        # runs training
        trained_model = model.train(data=os.path.join(preprocessed_dataset, "data.yaml"),
                                    epochs=num_epochs,
                                    batch=batch_size)
        
        # moves results to our directoyy for convenience, cleans up
        shutil.copytree("./runs/segment/train", os.path.join(folder, "training_output"))
        shutil.rmtree("./runs/segment")
        shutil.move(os.path.join(folder, "training_output/weights"), os.path.join(folder, "weights"))

        # ends stopwatch
        train_end = time.time()

        print("")
        print("finished training in " + str(round((train_end - train_start) / 3600, 3)) + " hours")

    #folder/weghts/best.pt already exists
    else:
        print("")
        print("!!! SKIPPED TRAINING OR USING PROVIDED TRAINED MODEL !!!")
        print("")
        if ("weights" in os.listdir(folder) and "best.pt" in os.listdir(os.path.join(folder, "weights"))):
            trained_model = YOLO(os.path.join(folder, "weights/best.pt"))
        else:
            print("NO WEIGHTS GIVEN AND SKIPPED TRAINING, ALSO SKIPPING EVAL")
            run_eval = False
        
    #EVALULATION

    #we want to run eval and there is no eval folder already there
    if run_eval and "evaluation" not in os.listdir(folder):
        print("")
        print("!!! STARTING EVALUATION !!!")
        print("")

        #runs prediction with model
        test_results = trained_model(os.path.join(folder, "prepared_dataset/test/images"), save=True, max_det=1)

        #moves model results to our folder
        shutil.copytree("./runs/segment/predict", os.path.join(folder, "evaluation/visualization"))
        shutil.rmtree("./runs/segment/predict")

        #creates folder to put the labels in
        predicted_label_path = os.path.join(folder, "evaluation/predicted_labels")
        os.mkdir(os.path.join(predicted_label_path))

        for result in test_results:
            #get normalized xy coordinates from results object
            ultralytics_mask = result.masks
            coords = ultralytics_mask.xyn

            #TODO: probably make the rest of this for loop into a helper function for readability

            # get first and only prediction, adds to flattened list
            coords = np.array(coords)[0]
            flattened_coords = [0] #class index for label format

            for pair in coords:
                for value in pair:
                    flattened_coords.append(value)

            #converts to string
            formatted_string = ' '.join(map(str, flattened_coords))
            
            #gets location to put label
            label_path = result.path
            label_path = label_path.replace("prepared_dataset/test/images", "evaluation/predicted_labels")
            label_path = label_path.replace('.jpg', '.txt')

            #writes string to label txt
            with open(label_path, 'w') as label_file:
                label_file.write(formatted_string)

        #converts folder of labels to masks
        print("")
        label_to_mask_dir(os.path.join(folder, "evaluation/predicted_labels"), os.path.join(folder, "evaluation/predicted_nps_prepost"), (640, 640), True)
        
        #postprocessing
        print("")
        remove_small_islands(os.path.join(folder, "evaluation/predicted_nps_prepost"), os.path.join(folder, "evaluation/predicted_masks"), False)

        #calculate score
        accuracies = get_pixel_accuracy_of_dirs(os.path.join(folder, "evaluation/predicted_masks"), os.path.join(folder, "prepared_dataset/test/masks"))
        score = 100 * np.average(accuracies)

        os.mkdir(os.path.join(folder, "evaluation/accuracies"))

        #makes plot
        plt.scatter(range(len(accuracies)), accuracies)
        plt.savefig(os.path.join(folder, "evaluation/accuracies/plot.png"))

        print("")
        print("!!! FINAL SCORE OF MODEL " + str(round(score, 3)) + "% !!!")

        #cleanup
        shutil.rmtree(os.path.join(folder, "evaluation/predicted_nps_prepost"))
    else:
        print("")
        print("!!! FINISHED WITHOUT EVALUATION !!!")
        print("")

    # final check for cleanup
    if "runs" in os.listdir("./"):
        shutil.rmtree("./runs")

if __name__ == "__main__":
    """train(YOLO("yolov8n-seg"), "/Users/nathansun/Documents/Special-Topics-Group-Project-2024/TEST",
          100,
          8,
          run_training=False,
          run_eval=False)"""

    train(YOLO("yolov8n-seg.pt"), 
          "/Users/nathansun/Documents/Special-Topics-Group-Project-2024/11:28 OVERNIGHT",
          100,
          8,)