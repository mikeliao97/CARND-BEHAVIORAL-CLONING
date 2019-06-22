from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
import argparse
import os
import cv2

def load_data():
    data_df = pd.read_csv("./dataset.csv")

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, 66, 200, 3])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            images[i] = cv2.cvtColor(cv2.imread(os.path.join(data_dir, center)), cv2.COLOR_BGR2RGB) #read from cente
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

def build_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))   
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(24, (5, 5), subsample=(2,2), activation='relu'))    
    model.add(Conv2D(36, (5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, (5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def train_model(model, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.003))

    model.fit_generator(batch_generator('dataset.csv', X_train, y_train, 128, True),
                        max_q_size=1,
                        epochs=4,
                        validation_data=batch_generator('dataset.csv', X_valid, y_valid, 128, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

if __name__ == '__main__':
    dataset = load_data()
    model = build_model()
    train_model(model, *dataset)