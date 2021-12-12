from tensorflow.keras import datasets, layers, models, losses, optimizers
import numpy as np
import pandas as pd

def build_model(input_shape, keep_prob=1.0):
    model = models.Sequential()
    model.add(layers.Conv2D(24, (5, 5), strides=(2,2), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
    model.add(layers.Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=(1,1), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=(1,1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1164, activation='relu'))
    model.add(layers.Dropout(1-keep_prob))
    model.add(layers.Dense(1152, activation='relu'))
    model.add(layers.Dropout(1-keep_prob))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(1-keep_prob))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(1-keep_prob))
    model.add(layers.Dense(10))
    model.add(layers.Dropout(1-keep_prob))
    model.add(layers.Dense(1, activation='tanh'))

    
    opt = optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss=losses.mean_squared_error)
    return model

if __name__ == '__main__':
    import cv2
    from data import load_datasets
    X, Y = load_datasets()

    N_frames = 10
    X = X[:N_frames]
    Y = Y[:N_frames]

    for i in range(N_frames):
        x = X[i]
        y = Y[i]
        cv2.imshow('Example - Show image in window',x)
        print(y)
        cv2.waitKey(200)

    m = build_model(X[0].shape)
    m.summary()
    out = m.predict(X)
    print(out)

    m.fit(X,Y)