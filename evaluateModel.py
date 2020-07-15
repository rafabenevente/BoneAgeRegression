import os
import pandas as pd
import cv2
import numpy as np
import modelXception
from handSeparation import GetImageForValidatation

path_to_project = os.getcwd()
path_to_datasets = os.path.join(path_to_project, "Dataset")
path_to_result = os.path.join(path_to_project, "TrainResult")
path_to_image = os.path.join(path_to_datasets, "images")

def GetMeanAndStd(path_to_train_images, result_column):
    train_df_all = pd.read_csv(path_to_train_images)
    mean = train_df_all[result_column].mean()
    std = train_df_all[result_column].std()
    return (mean, std)

def GetImages(dataset):
    images = np.array(
        [np.array(GetImageForValidatation(
                    cv2.imread(os.path.join(path_to_image, fname))))
                    for fname in dataset["fileName"].values])

    return images

test_df = pd.read_csv(os.path.join(path_to_datasets, 'test.csv'))

model = modelXception.CreateModel()
model.load_weights(os.path.join(path_to_result, "xceptionweights_14-07_11-07.h5"))
y_pred = model.predict(GetImages(test_df))
predicted = y_pred.flatten()

boneage_mean, boneage_std = GetMeanAndStd(os.path.join(path_to_datasets,"train.csv"), "boneage")
predicted_months = boneage_mean + boneage_std*(predicted)

filenames=test_df['fileName']
results=pd.DataFrame({"fileName":filenames,
                      "boneage": predicted_months})
results.to_csv(os.path.join(path_to_result,"results_from_evaluate_model_cru.csv"),index=False)

predicted_months[predicted_months < 0] = 100
predicted_months[predicted_months > 400] = 100
results=pd.DataFrame({"fileName":filenames,
                      "boneage": predicted_months})

results.to_csv(os.path.join(path_to_result,"results_from_evaluete_model_tratado.csv"),index=False)
