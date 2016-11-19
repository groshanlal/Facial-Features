import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

## READ DATA
df=pd.read_csv('data/test.csv')
image=[]
for im in df['Image']:
    image.append(np.fromstring(im, sep=' ')/255.0)
df['Image'] = image

##  DELETE INVALID DATA
df = df.dropna()

## RESHAPE IMAGE TO 2D, SHUFFLE DATA
X = np.vstack(df['Image'])
X = X.reshape(-1,96,96,1)
cols = df.columns[:-1]
Y = df[cols].values / 96.0
X, Y = shuffle(X, Y)

## TRAIN/VAL SPLIT
train_data  = X[100:]
train_label = Y[100:]
val_data  = X[:100]
val_label = Y[:100]
print("No. of Training Samples: ", train_data.shape[0])
print("No. of Validation Samples: ", val_data.shape[0])

## NEURAL NETWORK ARCHITECTURE
model = Sequential()

model.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=train_data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Dropout(0.2))


model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(30))

## LOSS FUNCTION AND GRADIEDNT DESCENT
optimizer = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

## TRAINING
earlyStopping=EarlyStopping(monitor='val_loss', patience=1, min_delta=0)
#filepath="model-{epoch:02d}-{val_loss:.6f}.h5"
filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_los')

model.fit(train_data, train_label, batch_size = 64, nb_epoch = 100,
              validation_data = (val_data, val_label), callbacks=[earlyStopping, checkpoint],
              shuffle=True)
