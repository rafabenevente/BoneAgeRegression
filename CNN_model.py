from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.layers import *
import os


def CreateModel():
    pretrained_model = Xception(weights="imagenet", input_shape=(299,299,3), include_top=False, pooling="max")

    x = (pretrained_model.output)
    x = Dense(10, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(1, activation='linear')(x)

    model = Model(pretrained_model.input, x)

    model.compile(loss='mse', optimizer='adam', metrics=[mean_squared_error])
    return model


def SaveJSonModel(model, pathToSave):
    model_json = model.to_json()
    with open(os.path.join(pathToSave, "model.json"), "w") as json_file:
        json_file.write(model_json)