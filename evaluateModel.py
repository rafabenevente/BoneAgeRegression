import os
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.metrics import mean_squared_error
from handSeparation import GetImageFromValidate

def GetMeanAndStd(path_to_train_images, result_column):
    train_df_all = pd.read_csv(path_to_train_images)
    mean = train_df_all[result_column].mean()
    std = train_df_all[result_column].std()
    return (mean, std)

def GetModel(path_to_model_json, path_to_weights, loss_func, optimizer_func, metrics_func):
    json_file = open(path_to_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_to_weights)
    loaded_model.compile(loss=loss_func, optimizer=optimizer_func, metrics=metrics_func)
    return loaded_model

path_to_files = "/home/rafael/I2A2/Desafio 1 - Bone Age Regression"
path_to_datasets = os.path.join(path_to_files, "Dataset")
path_to_image = os.path.join(path_to_datasets, "images")

test_df = pd.read_csv(os.path.join(path_to_datasets, 'test.csv'))

images = np.array(
    [np.array(GetImageFromValidate(
                cv2.imread(os.path.join(path_to_image, fname))))
                            for fname in test_df["fileName"].values])

model = GetModel(os.path.join(path_to_files, "model.json"),
                  os.path.join(path_to_files, "weights_30-06_14-36.h5"),
                  "mse",
                  "adam",
                  [mean_squared_error])
y_pred = model.predict(images)
predicted = y_pred.flatten()

boneage_mean, boneage_std = GetMeanAndStd(os.path.join("/home/rafael/I2A2/Desafio 1 - Bone Age Regression/Dataset","train.csv"), "boneage")
predicted_months = boneage_mean + boneage_std*(predicted)

filenames=test_df['fileName']
results=pd.DataFrame({"fileName":filenames,
                      "boneage": predicted_months})
results.to_csv("/home/rafael/I2A2/Desafio 1 - Bone Age Regression/results_from_evaluate_model_cru.csv",index=False)

predicted_months[predicted_months < 0] = 100
predicted_months[predicted_months > 400] = 100
filenames=test_df['fileName']
results=pd.DataFrame({"fileName":filenames,
                      "boneage": predicted_months})

results.to_csv("/home/rafael/I2A2/Desafio 1 - Bone Age Regression/results_from_evaluete_model_tratado.csv",index=False)
