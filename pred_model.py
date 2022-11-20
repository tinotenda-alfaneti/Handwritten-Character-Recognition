from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2 as cv
#ignoring warnings
import warnings 
warnings.filterwarnings("ignore")

model = load_model('letters_pred_model.h5')


data_generator = ImageDataGenerator(rescale=1/255)
predict_data = data_generator.flow_from_directory(
    directory=r"./uploads/",
    target_size=(64, 64),
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

print(type(predict_data))