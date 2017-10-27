import numpy as np
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

x = pd.read_csv("X_train.dms")
y = pd.read_csv("Y_train.dms")
x_test = pd.read_csv("X_test.dms")
x_test = np.array(x_test)

x = np.array(x)
y = np.squeeze(y)
y = np_utils.to_categorical(y)

x[:,0] = (x[:,0]-np.mean(x[:,0]))/np.std(x[:,0])
x[:,1] = (x[:,1]-np.mean(x[:,1]))/np.std(x[:,1])
x[:,3] = (x[:,3]-np.mean(x[:,3]))/np.std(x[:,3])
x[:,4] = (x[:,4]-np.mean(x[:,4]))/np.std(x[:,4])
x[:,5] = (x[:,5]-np.mean(x[:,5]))/np.std(x[:,5])

model = Sequential()
model.add(Dense(input_dim=106, units=150, activation='relu'))
model.add(BatchNormalization())
for i in range(5):
	model.add(Dense(units=150, activation='relu'))
	model.add(BatchNormalization())

model.add(Dense(units=2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, validation_split=0, batch_size=1000, epochs=30)

result = model.predict(x_test)
result = result.astype(int)

f = open("nn.csv","w")

f.write("id,label\n")

for i in range(result.shape[0]):
	f.write(str(i+1)+","+str(result[i][1])+"\n")
f.close()