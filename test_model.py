import matplotlib.pyplot as plt
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2 as cv
import random
#ignoring warnings
import warnings 
warnings.filterwarnings("ignore")

model = load_model('letters_pred_model.h5')

path = 'uploads/'

def testModel():

    letters_labels = pd.read_csv(f'uploads/test.csv')


    rand = random.sample(range(len(letters_labels)), 10)

    test_set = pd.DataFrame(letters_labels.iloc[rand, :].values, columns=['image', 'label'])

    test_data_generator = ImageDataGenerator(rescale=1/255)
    test_data_frame = test_data_generator.flow_from_dataframe(dataframe=test_set, directory=path, x_col='image', y_col='label', 
                                                        target_size=(64, 64), class_mode='categorical', shuffle=False)

    pred = model.predict(test_data_frame)

    pred_results = pd.DataFrame(pred)

    return pred_results, test_set

def predChar():
    letters_labels = pd.read_csv(f'uploads/test.csv')
    length = len(letters_labels)
    pred_set = pd.DataFrame(letters_labels.iloc[length - 1:length].values, columns=['image', 'label'])

    pred_data_generator = ImageDataGenerator(rescale=1/255)
    pred_data_frame = pred_data_generator.flow_from_dataframe(dataframe=pred_set, directory=path, x_col='image', y_col='label', 
                                                        target_size=(64, 64), class_mode='categorical', shuffle=False)

    pred = model.predict(pred_data_frame)

    pred_results = pd.DataFrame(pred)

    return pred_results, pred_set

   
