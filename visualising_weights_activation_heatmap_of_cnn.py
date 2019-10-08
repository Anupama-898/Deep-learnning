import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
keras.backend.set_image_data_format('channels_first')

batch_size = 128
nb_classes =10
nb_epoch =20
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv  = 1024
kernel_size = 3

#input_img = Input(shape=(784,))
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols).astype('float32') / 255
#X_train = X_train.reshape(1, img_rows, img_cols,X_train.shape[0])
#X_test = X_test.reshape(1, img_rows, img_cols, X_test.shape[0])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255
X_train /= 255

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(1, img_rows, img_cols), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(img_rows, img_cols,1), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(img_rows, img_cols,1), border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=2, validation_data=(X_test, Y_test))

#model.summary()

#print(model.layers[1].get_weights()[0].shape) # Convolution2D
#print(model.layers[1].get_weights()[1].shape) # Dense


W1 = model.layers[0].get_weights()[0]
print(W1.shape)
w = W1.reshape((32, 1, 3, 3))
print(w.shape)
plt.figure(figsize=(10, 10), frameon=False)
for ind, val in enumerate(w):
  plt.subplot(6, 6, ind + 1)
  im = val.reshape((3,3))
  plt.axis("off")
  plt.imshow(im, cmap='gray',interpolation='nearest')


    
W2 = model.layers[3].get_weights()[0]
print(W2.shape)
w = W2.reshape((32, 32, 3, 3))
print(w.shape)
plt.figure(figsize=(10, 10), frameon=False)
for ind, val in enumerate(w):
  plt.subplot(6, 6, ind + 1)
  im = val.reshape((32,9))
  plt.axis("off")
  plt.imshow(im, cmap='gray',interpolation='nearest')

  
W3 = model.layers[5].get_weights()[0]
print(W3.shape)
w = W3.reshape((32, 32, 3, 3))
print(w.shape)
plt.figure(figsize=(10, 10), frameon=False)
for ind, val in enumerate(w):
 plt.subplot(6, 6, ind + 1)
 im = val.reshape((32,9))
 plt.axis("off")
 plt.imshow(im, cmap='gray',interpolation='nearest')
 
 import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
keras.backend.set_image_data_format('channels_first')

batch_size = 128
nb_classes =10
nb_epoch =20
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv  = 1024
kernel_size = 3

#input_img = Input(shape=(784,))
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols).astype('float32') / 255
#X_train = X_train.reshape(1, img_rows, img_cols,X_train.shape[0])
#X_test = X_test.reshape(1, img_rows, img_cols, X_test.shape[0])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255
X_train /= 255

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(1, img_rows, img_cols), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(img_rows, img_cols,1), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(img_rows, img_cols,1), border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=2, validation_data=(X_test, Y_test))

#model.summary()

#print(model.layers[1].get_weights()[0].shape) # Convolution2D
#print(model.layers[1].get_weights()[1].shape) # Dense


W1 = model.layers[0].get_weights()[0]
print(W1.shape)
w = W1.reshape((32, 1, 3, 3))
print(w.shape)
plt.figure(figsize=(10, 10), frameon=False)
for ind, val in enumerate(w):
  plt.subplot(6, 6, ind + 1)
  im = val.reshape((3,3))
  plt.axis("off")
  plt.imshow(im, cmap='gray',interpolation='nearest')


    
W2 = model.layers[3].get_weights()[0]
print(W2.shape)
w = W2.reshape((32, 32, 3, 3))
print(w.shape)
plt.figure(figsize=(10, 10), frameon=False)
for ind, val in enumerate(w):
  plt.subplot(6, 6, ind + 1)
  im = val.reshape((32,9))
  plt.axis("off")
  plt.imshow(im, cmap='gray',interpolation='nearest')

  
W3 = model.layers[5].get_weights()[0]
print(W3.shape)
w = W3.reshape((32, 32, 3, 3))
print(w.shape)
plt.figure(figsize=(10, 10), frameon=False)
for ind, val in enumerate(w):
 plt.subplot(6, 6, ind + 1)
 im = val.reshape((32,9))
 plt.axis("off")
 plt.imshow(im, cmap='gray',interpolation='nearest')
 
 from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
import scipy as sp

batch_size = 128
nb_classes =10
nb_epoch =20
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv  = 1024
kernel_size = 3


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255
X_train /= 255

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)




model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(img_rows, img_cols, 1), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(img_rows, img_cols,1), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(img_rows, img_cols,1), border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='relu'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=20, verbose=2, validation_data=(X_test, Y_test))
#model.summary()
# get the weights from the last layer

gap_weights = model.layers[-1].get_weights()[0]

cam_model = Model(inputs=model.layers[0].input, 
                    outputs=(model.layers[7].output, model.layers[-1].output)) 
features, results = cam_model.predict(X_train)

# check the prediction for 10 test images
for idx in range(10):   
    # get the feature map of the test image
    features_for_one_img = features[idx, :, :, :]
    #print(features_for_one_img.shape[0])
    #print(features_for_one_img.shape[1])
    # map the feature map to the original size
    height_roomout = 28 / features_for_one_img.shape[0]
    width_roomout = 28 / features_for_one_img.shape[1]
    cam_features = sp.ndimage.zoom(features_for_one_img, (height_roomout, width_roomout, 1), order=2)
    #print(cam_features.shape)  
    # get the predicted label with the maximum probability
    pred = np.argmax(results[idx])
    print(pred)
    
    # prepare the final display
    plt.figure(facecolor='white')
    
    # get the weights of class activation map
    cam_weights = gap_weights[:, pred]
    #print(cam_weights.shape)  
    # create the class activation map
    cam_output = np.dot(cam_features, cam_weights)
    
    # draw the class activation map
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    
    #buf = 'Predicted Class = ' + fashion_name[pred] + ', Probability = ' + str(results[idx][pred])
    #plt.xlabel(buf)
    #plt.imshow(t_pic[idx], alpha=0.5)
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
     
    plt.show()  