from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
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
                                                    directory="/drive/My Drive/I2A2/Desafios/Desafio 1 - Bone Age Regression/Dataset/images/",
                                                    x_col="fileName",
                                                    y_col="boneage_z",
                                                    batch_size=32,
                                                    seed=2020,
                                                    shuffle=True,
                                                    class_mode="other",
                                                    color_mode="rgb",
                                                    target_size=size)

    return generator

train_df_all = pd.read_csv(os.path.join(path_to_datasets, 'train.csv'))

boneage_mean = train_df_all['boneage'].mean()
boneage_std = train_df_all['boneage'].std()
train_df_all['boneage_z'] = train_df_all['boneage'].map(lambda x: (x-boneage_mean)/boneage_std)
train_df_all.dropna(inplace = True)

train_df, valid_df = train_test_split(train_df_all,
                                   test_size = 0.25,
                                   random_state = 2020)
print('train', train_df.shape[0], 'validation', valid_df.shape[0])


size = (299, 299)
train_generator = GetGenerators(size, valid_df)
val_generator = GetGenerators(size, train_df)

model = modelXception.CreateModel()
weight_file = "weights{}.h5".format(datetime.datetime.today().strftime("_%d-%m_%H-%M"))
weights_filepath = os.path.join(path_to_result,"/xception" + weight_file)
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
