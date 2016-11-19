import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import csv
import matplotlib.pyplot as plt
import random

## READ DATA
df=pd.read_csv('data/test.csv')
image=[]
for im in df['Image']:
    image.append(np.fromstring(im, sep=' ')/255.0)
df['Image'] = image

## RESHAPE IMAGE TO 2D
X = np.vstack(df['Image'])
X = X.reshape(-1,96,96,1)

print("No. of Testing Samples: ", X.shape[0])

## TESTING
model=load_model('model.h5')
Y = model.predict(X)
Y = Y*96
Y = Y.astype(int)
X = X*255
X = X.astype(int)

with open("result/keypoints.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(Y)

i = random.randint(0, X.shape[0]-1)
x = X[i]
y = Y[i]
img = x.reshape(96, 96)
plt.imshow(img, cmap='gray')
plt.scatter(y[0::2], y[1::2])
#plt.scatter(truth[0::2] * 96, truth[1::2] * 96, c='r', marker='x')
plt.savefig("result/img.png")
