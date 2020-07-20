from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from dataAug import DoAugmentation
import tensorflow as tf
import cv2
import numpy as np
import modelXception
import pandas as pd
import datetime
import os


path_to_project = os.getcwd()
path_to_datasets = os.path.join(path_to_project, "Dataset")
path_to_result = os.path.join(path_to_project, "TrainResult")
path_to_image = os.path.join(path_to_datasets, "images")

def GetGenerators(size, df):
    imgGen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = imgGen.flow_from_dataframe(dataframe=df,
                                                    directory=path_to_image,
                                                    x_col="fileName",
                                                    y_col="boneage_z",
                                                    batch_size=32,
                                                    seed=2020,
                                                    shuffle=True,
                                                    class_mode="other",
                                                    color_mode="rgb",
                                                    target_size=size)

    return generator

# def ReadAndResize(path, size):
#     img = cv2.imread(path)
#     return cv2.resize(img, size)
#
# def GetImage(path, size, doAug):
#     aug = None
#     img = ReadAndResize(path, size)
#     if doAug:
#         aug = DoAugmentation(img)
#     return img, aug
#
# def GetImages(dataset, size, doAug):
#     # images = np.append([GetImage(path=os.path.join(path_to_image, fname),
#     #                             size=size,
#     #                             doAug=doAug)
#     #                     for fname in dataset["fileName"].values])
#     images = None
#     for fname in dataset["fileName"].values:
#         img, aug = GetImage(path=os.path.join(path_to_image, fname),
#                              size=size,
#                              doAug=doAug)
#         np.append(images, img)
#         if doAug:
#             np.append(images, aug)
#     return images

train_df_all = pd.read_csv(os.path.join(path_to_datasets, "train.csv"))
train_df_all["path"] = train_df_all["fileName"].map(lambda x: os.path.join(path_to_image,
                                                         '{}'.format(x)))
train_df_all["exists"] = train_df_all["path"].map(os.path.exists)
print(train_df_all["exists"].sum(), "images found of", train_df_all.shape[0], "total")

boneage_mean = train_df_all["boneage"].mean()
boneage_std = train_df_all["boneage"].std()
train_df_all["boneage_z"] = train_df_all["boneage"].map(lambda x: (x-boneage_mean)/boneage_std)
train_df_all.dropna(inplace = True)

train_df, valid_df = train_test_split(train_df_all,
                                   test_size = 0.25,
                                   random_state = 2020)
print("train", train_df.shape[0], "validation", valid_df.shape[0])


size = (299, 299)
train_generator = GetGenerators(size, train_df)
val_generator = GetGenerators(size, valid_df)

files = train_df["path"].values
tfRecor = tf.data.TFRecordDataset(filenames=files.tolist())
for elem in tfRecor.take(100):
    print(elem)
a=1
# augmented_train_batches = (
#     train_dataset
#     # Only train on a subset, so you can quickly see the effect.
#     .take(NUM_EXAMPLES)
#     .cache()
#     .shuffle(num_train_examples//4)
#     # The augmentation is added here.
#     .map(augment, num_parallel_calls=AUTOTUNE)
#     .batch(BATCH_SIZE)
#     .prefetch(AUTOTUNE)
# )
# imgs_train = GetImages(dataset=train_df, size=size, doAug=False)
# imgs_valid = GetImages(dataset=valid_df, size=size, doAug=False)

model = modelXception.CreateModel(freezeInitialLayers=22)
weight_file = "weights{}.h5".format(datetime.datetime.today().strftime("_%d-%m_%H-%M"))
weights_filepath = os.path.join(path_to_result,"/xception" + weight_file)
model_json = model.to_json()
with open(os.path.join(os.path.join(path_to_result,"xceptionModel.json")), "w") as json_file:
    json_file.write(model_json)
callbacks = [ModelCheckpoint(weights_filepath, monitor='val_loss', mode='min',
                             verbose=1, save_best_only=True),
             EarlyStopping(monitor='val_loss', patience=15),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                               verbose=0, mode='min', min_delta=0.0001,
                               cooldown=0, min_lr=0)]
history = model.fit(train_generator,
          epochs = 100,
          verbose = 1,
          validation_data = val_generator,
          validation_steps = 1,
          callbacks=callbacks)

plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.ylabel('MSE')
plt.xlabel('época')
plt.legend(['treino', 'validação'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('época')
plt.legend(['treino', 'validação'], loc = 'upper left')
plt.show()
