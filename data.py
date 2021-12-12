'''
Data Model
'''
import os
import cv2
import numpy as np
import pandas as pd
from preprocess import preprocess
from local_common import frame_count
from sklearn.model_selection import train_test_split

# from preprocess import preprocess
# import local_common as cm
# import model_tf2 as model

EPOCH_PATH = r'.\epochs'

def list_datasets(root_dir, n_epochs=10):
    video_paths = []
    csv_paths = []
    for i in range(1,n_epochs+1):
        vp = os.path.join(root_dir,'epoch' + str(i).zfill(2) + '_front.mkv')
        assert os.path.exists(vp)
        video_paths.append(vp)
        cp = os.path.join(root_dir,'epoch' + str(i).zfill(2) + '_steering.csv')
        assert os.path.exists(cp)
        csv_paths.append(cp)
    return video_paths, csv_paths

def build_x_y(video_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    N_frames = frame_count(video_path)
    for _ in range(N_frames):
        ret, img = cap.read()
        frames.append(preprocess(img))
    x = np.array(frames)
    y = pd.read_csv(csv_path)['wheel'].values.reshape(-1,1)
    return x,y

def build_datasets(root_dir, n_epochs=10):
    video_paths, csv_paths = list_datasets(EPOCH_PATH, n_epochs)
    X = []; Y = []
    for vp, cp in zip(video_paths, csv_paths):
        print(vp,cp)
        x,y = build_x_y(vp,cp)
        print(x.shape,y.shape)
        X.append(x)
        Y.append(y)
    return np.concatenate(X), np.concatenate(Y)

def save_datasets(X, Y, filepath='dataset.npy'):
    with open(filepath, 'wb') as f:
        np.save(f, X)
        np.save(f, Y)

def load_datasets(filepath='dataset.npy'):
    with open(filepath, 'rb') as f:
        X = np.load(f)
        Y = np.load(f) 
    return X, Y

def load_test_train_data(filepath='dataset.npy'):
    X, Y = load_datasets(filepath)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42)
    
    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    # X,Y = build_datasets(EPOCH_PATH, 2)
    # print(X.shape, Y.shape)
    # del(X)
    # del(Y)
    X, Y = load_datasets()
    print(X.shape, Y.shape)
    save_datasets(X, Y)