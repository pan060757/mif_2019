#-*-coding:utf-8-*-
'''
可变timestep
'''

from collections import deque
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from matplotlib import pyplot
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense,Masking

data_x=deque()
data_y=deque()
file= open("dataset/data_of_month.csv")
for line in file:
    lines=line.strip("\n").split(',')
    x=lines[:-2]
    y=lines[-1:]
    data_x.append(x)
    data_y.append(y)
data_x=np.array(data_x)
data_y=np.array(data_y)
print(data_x)
print(data_y)
size=int(len(data_x)*0.70)
train_x=data_x[0:size]
test_x=data_x[size:]
train_y=data_y[0:size]
test_y=data_y[size:]
train_x=sequence.pad_sequences(train_x,dtype='float32',padding='post',maxlen=12,value=0.0)
test_x=sequence.pad_sequences(train_x,12)
print(train_x)
print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(11,1)))
model.add(LSTM(1, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_x, train_y, epochs=50, batch_size=72, validation_data=(test_x, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()