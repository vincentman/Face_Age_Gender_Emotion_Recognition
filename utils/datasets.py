from scipy.io import loadmat
import pandas as pd
import numpy as np
import cv2
from keras.utils import np_utils



def load_imdb_or_wiki(mat_path):
    """
    images have been resized as 64*64
    """
    d = loadmat(mat_path)

    image = d["image"]
    gender = d["gender"][0]
    age = d["age"][0]
    db = d["db"][0]
    img_size = d["img_size"][0, 0]  # 64
    min_score_ = d["min_score"][0, 0]

    return image, gender, age, db, img_size, min_score_


def load_fer2013(mat_path, resize=(64, 64)):
    data = pd.read_csv(mat_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), resize)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    return faces, data['emotion']


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

