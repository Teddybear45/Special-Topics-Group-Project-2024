from ultralytics import YOLO
import os
import shutil
import time
from preprocess import prepare_dataset

def train(model: YOLO,                          # model to train
          folder: str,                          # filepath to the folder of the run, containing just the dataset in folder named "dataset"
          num_epochs: int,                      # number of epochs to train for
          batch_size: int,                      # batch size
          run_testing=True):                    # if true, runs teseting, if not, skip
    
    # PREPROCESSING

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

    else:

        print("")
        print("!!!  USING PROVIDED PREPROCESSED DATASET !!!")
        print("")

        preprocessed_dataset = os.path.join(folder, "prepared_dataset")
    
    # TRAINING

    # cleans up from past runs
    if "runs" in os.listdir("./"):
        shutil.rmtree("./runs")
        
    if "weights" not in os.listdir(folder) or "best.pt" not in os.listdir(os.path.join(folder, "weights")):

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

    else:
        print("")
        print("!!! USING PROVIDED TRAINED MODEL !!!")
        print("")
        trained_model = YOLO(os.path.join(folder, "weights/best.pt"))
        
    #EVALULATION

    if run_testing:
        print("")
        print("!!! STARTING TESTING !!!")
        print("")
        test_results = trained_model(os.path.join(folder, "prepared_dataset/test/images"), save=True)
        shutil.copytree("./runs/segment/predict", os.path.join(folder, "test_visualization"))
        shutil.rmtree("./runs/segment/predict")
    else:
        print("")
        print("!!! FINISHED WITHOUT TESTING !!!")
        print("")
        
    shutil.rmtree("./runs")