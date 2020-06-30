import numpy
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,BatchNormalization,Activation
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.constraints import max_norm
from keras.utils import np_utils
from keras.datasets import cifar10

#set random seed for the purpose of reproductivty
seed = 21

#loading in the data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train=x_train/255.0
x_test=x_test/255.0

#one hot encode outputs
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
class_num=y_test.shape[1]

model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(3,32,32),activation='relu',padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=max_norm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=max_norm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(class_name))
model.add(Activation('softmax'))

epochs=25
optimizer='adam'

model.complie(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
print(model.summary())