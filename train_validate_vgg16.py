
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import os
import glob
import h5py
import random
import datetime
import numpy as np
import cv2

def mae_months(in_gt, in_pred):
  return mean_absolute_error((boneage_std*in_gt + boneage_mean), (boneage_std*in_pred + boneage_mean))

# """# > Load and analyse the dataset"""
dataset_dir = "/home/rafael/I2A2/Desafio 1 - Bone Age Regression/Dataset"
train_df_all = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
train_df_all['path'] = train_df_all['fileName'].map(lambda x: os.path.join(dataset_dir,
                                                         'images',
                                                         '{}'.format(x)))
train_df_all['exists'] = train_df_all['path'].map(os.path.exists)
print(train_df_all['exists'].sum(), 'images found of', train_df_all.shape[0], 'total')

boneage_mean = train_df_all['boneage'].mean()
boneage_std = train_df_all['boneage'].std()
train_df_all['boneage_z'] = train_df_all['boneage'].map(lambda x: (x-boneage_mean)/boneage_std)
train_df_all.dropna(inplace = True)
print(train_df_all.describe())

# train_df_all['boneage'].hist(figsize = (10, 5))
# pd.value_counts(train_df_all['patientSex']).plot.bar(figsize = (10, 5))

# """# > Split data for validation"""
train_df, valid_df = train_test_split(train_df_all,
                                   test_size = 0.25, 
                                   random_state = 2020)#,
                                  #  stratify = train_df['boneage'])
print('train', train_df.shape[0], 'validation', valid_df.shape[0])

size = (256, 256)

# """# > Creates the image generator for data augmentation"""
aug_gen = ImageDataGenerator(horizontal_flip = True,
                              vertical_flip = True,
                              height_shift_range = 0.15,
                              width_shift_range = 0.15,
                              rotation_range = 5,
                              shear_range = 0.01,
                              zoom_range=0.25,
                             preprocessing_function = preprocess_input,
                             rescale=1./255)

train_generator = aug_gen.flow_from_dataframe(dataframe = train_df,
                                            directory = "/home/rafael/I2A2/Desafio 1 - Bone Age Regression/Dataset/images",
                                            x_col= "fileName",
                                            y_col= "boneage_z",
                                            batch_size = 64,
                                            seed = 42,
                                            shuffle = True,
                                            color_mode = "rgb",
                                            class_mode = 'raw',
                                            target_size = size)

val_generator = aug_gen.flow_from_dataframe(dataframe = valid_df,
                            directory = "/home/rafael/I2A2/Desafio 1 - Bone Age Regression/Dataset/images",
                            x_col= "fileName",
                            y_col= "boneage_z",
                            batch_size = 64,
                            seed = 42,
                            shuffle = True,
                            color_mode = "rgb",
                            class_mode = 'raw',
                            target_size = size)

# """# > Plot examples"""

# t_x, t_y = next(train_gen)
# fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
# for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
#     c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)
#     c_ax.set_title('%2.0f months' % (c_y))
#     c_ax.axis('off')

# """# > Load and configure the model"""

pretrained_model = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))

number_of_frozen_layers = 7
for i, layer in enumerate(pretrained_model.layers):
    if i >= number_of_frozen_layers:
        break
    layer.trainable = False

# x = GlobalAveragePooling2D()(pretrained_model.output)
x = Flatten()(pretrained_model.output)
x = BatchNormalization()(x)
x = Dense(10, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='relu')(x)
x = Dense(1, activation='linear')(x)

model = Model(pretrained_model.input, x)
model.compile(loss='mse', optimizer='adam', metrics = [mae_months])
model.summary()

# """# > Train the model"""

weight_file = "weights_30-06_14-36.h5"
weights_filepath = "/home/rafael/I2A2/Desafio 1 - Bone Age Regression/" + weight_file
print(weights_filepath)

# serialize model to JSON
model_json = model.to_json()
with open("/home/rafael/I2A2/Desafio 1 - Bone Age Regression/model.json", "w") as json_file:
    json_file.write(model_json)

callbacks = [ModelCheckpoint(weights_filepath, monitor='val_loss', mode='min',
                             verbose=1, save_best_only=True),
             EarlyStopping(monitor='val_loss', patience=15),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                               verbose=0, mode='auto', min_delta=0.0001,
                               cooldown=0, min_lr=0)]
num_epochs = 100

history = model.fit(train_generator,
          epochs = num_epochs,
          verbose = 1,
          validation_data = val_generator,
          validation_steps = 1,
          callbacks=callbacks)
model.load_weights(weights_filepath)

# """# > Plot training graph"""

# plt.plot(history.history['mae_months'])
plt.plot(history.history['val_mae_months'])
plt.ylabel('acurácia')
plt.xlabel('época')
plt.legend(['treino', 'validação'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('época')
plt.legend(['treino', 'validação'], loc = 'upper left')
plt.show()

# """# > Verifying the accuracy"""

test_X, test_Y = next(val_gen.flow_from_dataframe(valid_df,
                                                  directory = "/drive/My Drive/I2A2/Desafios/Desafio 1 - Bone Age Regression/Dataset/images/",
                                                  x_col = "fileName",
                                                  y_col = "boneage_z",
                                                  target_size = size,
                                                  batch_size = 3153,
                                                  class_mode = 'other'
                                                  ))
print(test_X.shape)
print(test_Y.shape)

print(model.evaluate(test_X, test_Y)[1])

# """# > Test dataset"""
model.load_weights(weights_filepath)

test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
test_df['path'] = test_df['fileName'].map(lambda x: os.path.join(dataset_dir,
                                                         'images', 
                                                         '{}'.format(x)))
test_df['exists'] = test_df['path'].map(os.path.exists)
print(test_df['exists'].sum(), 'images found of', test_df.shape[0], 'total')

images =  np.array([np.array(cv2.resize(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB), (256,256))) for fname in test_df["path"].values])
y_pred = model.predict(images)
predicted = y_pred.flatten()
predicted_months = boneage_mean + boneage_std*(predicted)

# Remove outliers
predicted_months[predicted_months < 0] = 100
predicted_months[predicted_months > 200] = 100
filenames=test_df['fileName']
results=pd.DataFrame({"fileName":filenames,
                      "boneage": predicted_months})

results.to_csv("/home/rafael/I2A2/Desafio 1 - Bone Age Regression/results_30-06_14-36_2.csv",index=False)